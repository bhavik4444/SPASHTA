/*
 * gtcrn_internal.h -- blob layout, layer descriptors and kernel prototypes.
 * Everything here mirrors tools/export_int8.py and tools/gtcrn_ref.py.
 */
#ifndef GTCRN_INTERNAL_H
#define GTCRN_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include "gtcrn.h"

#define GT_NAME_LEN       48
#define GT_DIR_ENTRY      88
#define GT_HEADER_BYTES   176

enum { GT_F32 = 0, GT_I8 = 1, GT_I32 = 2 };

typedef struct {
    char     name[GT_NAME_LEN];
    uint8_t  dtype;
    uint8_t  ndim;
    uint16_t rsv;
    uint32_t dims[4];
    uint32_t data_off, data_len, scale_off, n_scales;
    uint32_t rsv2;
} gt_dirent_t;   /* exactly GT_DIR_ENTRY bytes */

/* ------------------------------------------------------------------------ */
/* layer descriptors                                                         */
/* ------------------------------------------------------------------------ */

/* A quantised matmul-shaped weight: rows of int8 with one float scale each.
 * Row layout for convolutions is [out][tap][in_per_group], so the innermost
 * dot product walks contiguous memory. */
typedef struct {
    const int8_t *w;
    const float  *ws;       /* per output row */
    const float  *b;        /* float bias, never quantised */
    int           n_out;
    int           n_in;     /* per group, per tap */
    int           k;        /* frequency taps: 1 or 5 */
    int           groups;
} gt_qw_t;

typedef struct {
    gt_qw_t      wih, whh;  /* bias lives inside each */
    int          n_in, H;
} gt_gru_t;

/* GTConvBlock: SFE -> pw1 -> causal dilated depthwise -> pw2 -> BandTRA */
typedef struct {
    gt_qw_t      pc1;       /* 3*(C/2) -> C   */
    float        pc1_a;     /* PReLU alpha    */
    const float *dw_w;      /* (C,3,3) float  */
    const float *dw_b;
    float        dw_a;
    gt_qw_t      pc2;       /* C -> C/2       */
    gt_gru_t     tra;       /* C/2 -> 2*(C/2) */
    gt_qw_t      tra_fc;    /* 2*(C/2) -> C/2 */
    int          dilation;
    float       *hist;      /* (2*dilation+1) frames of [C][Wb]  */
    int          hist_n, hist_pos, hist_primed;
    float       *tra_h;     /* [tra_bands][H] */
} gt_block_t;

typedef struct {
    gt_gru_t     ia[2][2];  /* [grouped rnn][0=fwd, 1=bwd] over frequency */
    gt_qw_t      ia_fc;
    const float *ia_lnw, *ia_lnb;
    gt_gru_t     ie[2];     /* over time */
    gt_qw_t      ie_fc;
    const float *ie_lnw, *ie_lnb;
    float       *inter_h;   /* [bn_width][C] carried across frames */
} gt_dp_t;

/* ------------------------------------------------------------------------ */
#define GT_MAX_ALLOC 128


/* One pipeline frame exchanged between Core 0 and Core 1. */
struct gtcrn_frame {
    int C, Wb, F, MW, nd;
    float *re;
    float *im;
    float *enc_out0;
    float **enc_out;
    float *out_re;
    float *out_im;
    void *storage;
};

struct gtcrn {
    gtcrn_cfg_t  cfg;
    void        *allocs[GT_MAX_ALLOC];
    int          n_allocs;
    const uint8_t *base;          /* blob data section */
    const gt_dirent_t *dir;
    int          n_tensors;

    /* graph */
    gt_qw_t      enc_c1, enc_c2, dec_c1, head;
    float        enc_c1_a, enc_c2_a, dec_c1_a;
    gt_block_t  *enc, *dec;       /* n_dil each */
    gt_dp_t     *dp;              /* n_dpgrnn   */

    /* ERB, stored as contiguous runs */
    const int32_t *bm_start, *bm_len, *bs_start, *bs_len;
    const float   *bm_w, *bs_w;
    int32_t       *bm_off, *bs_off;

    /* ---- feature front-end state ---- */
    float  *lvl_ring;     int lvl_pos;      /* [level_frames]            */
    float  *smo_ring;     int smo_pos;      /* [smooth_frames][n_freqs]  */
    float  *nf_ring;      int nf_pos;       /* [nf_frames][n_freqs]      */
    int     front_primed;

    /* ---- deep filter history (masked spectrum, pre-residual) ---- */
    float  *df_r, *df_i;  int df_pos;       /* [df_order][df_bins]       */

    /* ---- STFT / overlap-add ---- */
    float  *win;          /* [n_fft] sqrt-Hann                          */
    float  *ana;          /* [n_fft] sliding analysis buffer            */
    float  *ola;          /* [n_fft] overlap-add accumulator            */
    float  *fft_re, *fft_im;
    float  *tw_re, *tw_im;      /* [n_fft/2] twiddles                   */
    uint16_t *brev;             /* [n_fft] bit-reversal table           */

    /* ---- per-frame scratch ---- */
    float  *pwr, *magbuf, *psmooth, *nfloor;   /* [n_freqs] each  */
    float  *feat;            /* [4][n_freqs]      */
    float  *erbed;           /* [4][width]        */
    float  *sfe_in;          /* [12][width]       */
    float  *enc_out0;        /* [C][mid_width]    */
    float **enc_out;         /* n_dil+1 buffers of [C][bn_width] */
    float  *xa, *xb;         /* [C][bn_width]     */
    float  *sfe_h;           /* [3*C/2][bn_width] */
    float  *hid, *hid2;      /* [C][bn_width]     */
    float  *half;            /* [C/2][bn_width]   */
    float  *mid;             /* [C][mid_width]    */
    float  *headbuf;         /* [n_out][width]    */
    float  *maskbuf;         /* [3][n_freqs]      */
    float  *er, *ei;         /* [n_freqs]         */
    float  *dfr, *dfi;       /* [df_bins]         */
    float  *dpa, *dpb, *dpc; /* [bn_width][C]     */
    float  *tra_zt, *tra_gain, *gru_h;
    float  *gruscratch;      /* 6 * H_max         */
    int8_t *q8;              /* quantisation scratch */

    int     mid_width;
    size_t  mem_int, mem_ps;
    uint32_t last_us;
};

/* ------------------------------------------------------------------------ */
/* kernels (gtcrn_ops.c)                                                     */
/* ------------------------------------------------------------------------ */
void  gt_quant(const float *x, int n, int8_t *q, float *scale);

/* Quantise [C][F] into a transposed, zero-padded [pad + F + pad][C] buffer so
 * the frequency convolutions can index neighbours without bounds tests. */
void  gt_quant_tpad(const float *x, int C, int F, int pad, int8_t *q, float *scale);

int32_t gt_dot_i8(const int8_t *a, const int8_t *b, int n);

void  gt_conv1x5_s2(const gt_qw_t *L, const float *x, int Cin, int Fin,
                    float *y, int Fout, int8_t *scratch);
void  gt_deconv1x5_s2(const gt_qw_t *L, const float *x, int Cin, int Fin,
                      float *y, int Fout, int8_t *scratch);
void  gt_conv1x1(const gt_qw_t *L, const float *x, int Cin, int F,
                 float *y, int8_t *scratch);
void  gt_qlinear_m(const gt_qw_t *L, const float *x, int rows, int n_in,
                   float *y, int8_t *scratch);
void  gt_linear_v(const gt_qw_t *L, const float *x, float *y, int8_t *scratch);

void  gt_depthwise33(const float *t0, const float *t1, const float *t2,
                     int C, int F, const float *w, const float *b, float *y);

void  gt_prelu(float *x, int n, float a);
float gt_sigmoid(float x);
float gt_tanh(float x);
void  gt_layernorm(float *x, int n, const float *w, const float *b);
void  gt_gru_step(const gt_gru_t *g, const float *x, float *h,
                  float *scratch, int8_t *q8);

void  gt_sfe3(const float *x, int C, int F, float *y);

/* radix-2 complex FFT, in place; inverse divides by n */
void  gt_fft_init(int n, float *tw_re, float *tw_im, uint16_t *brev);
void  gt_fft(float *re, float *im, int n, const float *tw_re, const float *tw_im,
             const uint16_t *brev, int inverse);

#endif /* GTCRN_INTERNAL_H */
