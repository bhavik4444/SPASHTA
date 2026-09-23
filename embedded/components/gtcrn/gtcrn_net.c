/*
 * gtcrn_net.c -- blob loading, memory plan, and the frame-by-frame forward pass.
 *
 * Read this next to tools/gtcrn_ref.py. The function order is the same and the
 * variable names are the same, deliberately: when something sounds wrong on the
 * board, you want to be able to put the two side by side and diff them by eye.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gtcrn_internal.h"

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_timer.h"
#else
#include <time.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================= */
/* allocation                                                                */
/* ========================================================================= */
static void *track(gtcrn_t *g, void *p)
{
    if (p && g->n_allocs < GT_MAX_ALLOC) g->allocs[g->n_allocs++] = p;
    return p;
}

/* Weights and hot scratch want internal SRAM: the S3's cache is small and
 * pulling 150 kB of coefficients over the SPI bus every frame is the single
 * easiest way to make this three times slower than it needs to be. */
static void *alloc_fast(gtcrn_t *g, size_t n)
{
    void *p = NULL;
#ifdef ESP_PLATFORM
    p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
#else
    p = malloc(n);
#endif
    if (p) { memset(p, 0, n); g->mem_int += n; }
    return track(g, p);
}

/* History rings are big and touched a few kilobytes per frame -- PSRAM is the
 * right home for them. */
static void *alloc_big(gtcrn_t *g, size_t n)
{
    void *p = NULL;
#ifdef ESP_PLATFORM
    p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
    if (!p) p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#else
    p = malloc(n);
#endif
    if (p) { memset(p, 0, n); g->mem_ps += n; }
    return track(g, p);
}

#define FAST(g, n)  ((float *)alloc_fast((g), (size_t)(n) * sizeof(float)))
#define BIG(g, n)   ((float *)alloc_big((g),  (size_t)(n) * sizeof(float)))

/* ========================================================================= */
/* blob lookup                                                               */
/* ========================================================================= */
static const gt_dirent_t *find(const gtcrn_t *g, const char *name)
{
    for (int i = 0; i < g->n_tensors; ++i)
        if (strncmp(g->dir[i].name, name, GT_NAME_LEN) == 0) return &g->dir[i];
    return NULL;
}

static const void *tdata(const gtcrn_t *g, const gt_dirent_t *e)
{
    return (const void *)(g->base + e->data_off);
}

static int load_qw(gtcrn_t *g, gt_qw_t *L, const char *pfx, const char *wsuf,
                   const char *bsuf, int k, int groups)
{
    char nm[GT_NAME_LEN];
    snprintf(nm, sizeof nm, "%s%s", pfx, wsuf);
    const gt_dirent_t *e = find(g, nm);
    if (!e || e->dtype != GT_I8 || e->ndim != 2) return -1;
    L->w      = (const int8_t *)tdata(g, e);
    L->ws     = (const float *)(g->base + e->scale_off);
    L->n_out  = (int)e->dims[0];
    L->k      = k;
    L->groups = groups;
    L->n_in   = (int)e->dims[1] / k;

    snprintf(nm, sizeof nm, "%s%s", pfx, bsuf);
    e = find(g, nm);
    if (!e || e->dtype != GT_F32) return -1;
    L->b = (const float *)tdata(g, e);
    return 0;
}

static int load_gru(gtcrn_t *g, gt_gru_t *G, const char *pfx)
{
    if (load_qw(g, &G->wih, pfx, "wih", "bih", 1, 1)) return -1;
    if (load_qw(g, &G->whh, pfx, "whh", "bhh", 1, 1)) return -1;
    G->H    = G->wih.n_out / 3;
    G->n_in = G->wih.n_in;
    return 0;
}

static const float *load_f32(gtcrn_t *g, const char *pfx, const char *suf)
{
    char nm[GT_NAME_LEN];
    snprintf(nm, sizeof nm, "%s%s", pfx, suf);
    const gt_dirent_t *e = find(g, nm);
    if (!e || e->dtype != GT_F32) return NULL;
    return (const float *)tdata(g, e);
}

static const int32_t *load_i32(gtcrn_t *g, const char *name, int *n)
{
    const gt_dirent_t *e = find(g, name);
    if (!e || e->dtype != GT_I32) return NULL;
    if (n) *n = (int)e->dims[0];
    return (const int32_t *)tdata(g, e);
}

/* ========================================================================= */
/* feature front-end                                                         */
/* ========================================================================= */
/*
 * Level tracking and minimum statistics, exactly as FeatureFront does them.
 *
 * One subtlety worth stating because it is invisible and matters: the training
 * code left-pads these causal windows in REPLICATE mode, so at the start of a
 * stream the very first frame is treated as if it had already been there for
 * three seconds. Priming the rings with the first frame reproduces that. Fill
 * them with zeros instead and the noise floor sits at zero for the first second
 * and a half, the SNR feature saturates at its ceiling, and the model passes
 * noise straight through until the window fills.
 */
static void feature_front(gtcrn_t *g, const float *re, const float *im,
                          const float *mag)
{
    const gtcrn_cfg_t *c = &g->cfg;
    const int F = c->n_freqs;

    double bp = 0.0;
    for (int f = 0; f < F; ++f) {
        g->pwr[f] = mag[f] * mag[f];
        bp += g->pwr[f];
    }
    bp /= (double)F;

    if (!g->front_primed) {
        g->front_primed = 1;
        for (int i = 0; i < c->level_frames; ++i) g->lvl_ring[i] = (float)bp;
        for (int i = 0; i < c->smooth_frames; ++i)
            memcpy(g->smo_ring + (size_t)i * F, g->pwr, (size_t)F * sizeof(float));
        memcpy(g->psmooth, g->pwr, (size_t)F * sizeof(float));
        for (int i = 0; i < c->nf_frames; ++i)
            memcpy(g->nf_ring + (size_t)i * F, g->psmooth, (size_t)F * sizeof(float));
        g->lvl_pos = g->smo_pos = g->nf_pos = 0;
    } else {
        g->lvl_ring[g->lvl_pos] = (float)bp;
        g->lvl_pos = (g->lvl_pos + 1) % c->level_frames;

        memcpy(g->smo_ring + (size_t)g->smo_pos * F, g->pwr, (size_t)F * sizeof(float));
        g->smo_pos = (g->smo_pos + 1) % c->smooth_frames;

        const float inv_s = 1.0f / (float)c->smooth_frames;
        for (int f = 0; f < F; ++f) {
            float a = 0.0f;
            for (int i = 0; i < c->smooth_frames; ++i)
                a += g->smo_ring[(size_t)i * F + f];
            g->psmooth[f] = a * inv_s;
        }
        memcpy(g->nf_ring + (size_t)g->nf_pos * F, g->psmooth, (size_t)F * sizeof(float));
        g->nf_pos = (g->nf_pos + 1) % c->nf_frames;
    }

    double la = 0.0;
    for (int i = 0; i < c->level_frames; ++i) la += g->lvl_ring[i];
    la /= (double)c->level_frames;
    const float level = sqrtf((float)(la > 1e-12 ? la : 1e-12));
    const float inv_level = 1.0f / level;

    /* running minimum over the noise-floor window. Brute force: 96 x 257
     * comparisons is ~1.5 M/s at 62.5 fps, under 1 % of the frame budget, and
     * an exact answer beats a clever monotonic deque that is subtly wrong. */
    for (int f = 0; f < F; ++f) {
        float m = g->nf_ring[f];
        for (int i = 1; i < c->nf_frames; ++i) {
            float v = g->nf_ring[(size_t)i * F + f];
            if (v < m) m = v;
        }
        g->nfloor[f] = m * c->nf_bias;
    }

    float *f0 = g->feat;
    float *f1 = g->feat + F;
    float *f2 = g->feat + 2 * F;
    float *f3 = g->feat + 3 * F;

    for (int f = 0; f < F; ++f) {
        float ps = g->psmooth[f] > 1e-12f ? g->psmooth[f] : 1e-12f;
        float nf = g->nfloor[f]  > 1e-12f ? g->nfloor[f]  : 1e-12f;
        float snr = log10f(ps / nf) * 0.5f;
        if (snr < c->snr_lo) snr = c->snr_lo;
        else if (snr > c->snr_hi) snr = c->snr_hi;
        f3[f] = snr;

        float mag_n = mag[f] * inv_level;
        if (mag_n < 1e-6f) mag_n = 1e-6f;
        float comp = powf(mag_n, c->compress);
        float gain = comp / mag_n;
        f0[f] = comp;
        f1[f] = re[f] * inv_level * gain;
        f2[f] = im[f] * inv_level * gain;
    }
}

/* ========================================================================= */
/* ERB band merge / split (stored as contiguous runs)                        */
/* ========================================================================= */
static void erb_bm(gtcrn_t *g, const float *x, float *y)
{
    const gtcrn_cfg_t *c = &g->cfg;
    const int F = c->n_freqs, W = c->width, e1 = c->erb1;
    for (int ch = 0; ch < 4; ++ch) {
        const float *xr = x + (size_t)ch * F;
        float *yr = y + (size_t)ch * W;
        memcpy(yr, xr, (size_t)e1 * sizeof(float));
        for (int i = 0; i < c->erb2; ++i) {
            const int n = g->bm_len[i];
            const float *w = g->bm_w + g->bm_off[i];
            const float *s = xr + e1 + g->bm_start[i];
            float acc = 0.0f;
            for (int j = 0; j < n; ++j) acc += s[j] * w[j];
            yr[e1 + i] = acc;
        }
    }
}

static void erb_bs(gtcrn_t *g, const float *x, int stride_in, float *y)
{
    const gtcrn_cfg_t *c = &g->cfg;
    const int F = c->n_freqs, e1 = c->erb1, nhi = F - e1;
    for (int ch = 0; ch < 3; ++ch) {
        const float *xr = x + (size_t)ch * stride_in;
        float *yr = y + (size_t)ch * F;
        memcpy(yr, xr, (size_t)e1 * sizeof(float));
        for (int i = 0; i < nhi; ++i) {
            const int n = g->bs_len[i];
            const float *w = g->bs_w + g->bs_off[i];
            const float *s = xr + e1 + g->bs_start[i];
            float acc = 0.0f;
            for (int j = 0; j < n; ++j) acc += s[j] * w[j];
            yr[e1 + i] = acc;
        }
    }
}

/* ========================================================================= */
/* GTConvBlock                                                               */
/* ========================================================================= */
static float *hist_at(gt_block_t *B, int back, int frame_len)
{
    int idx = B->hist_pos - 1 - back;
    while (idx < 0) idx += B->hist_n;
    return B->hist + (size_t)(idx % B->hist_n) * frame_len;
}

static void hist_push(gt_block_t *B, const float *x, int frame_len)
{
    memcpy(B->hist + (size_t)B->hist_pos * frame_len, x,
           (size_t)frame_len * sizeof(float));
    B->hist_pos = (B->hist_pos + 1) % B->hist_n;
}

/*
 * Band-wise temporal recurrent attention. The frequency axis is zero-padded up
 * to a multiple of tra_bands before the per-band energies are taken -- the
 * trailing band therefore sees partly (or entirely) zeros, and its GRU input is
 * genuinely zero for bn_width=33 with 8 bands. That looks like a bug and is not:
 * the trained weights were fitted with exactly this padding.
 */
static void band_tra(gtcrn_t *g, gt_block_t *B, float *x)
{
    const gtcrn_cfg_t *c = &g->cfg;
    const int half = c->base_channels / 2, Wb = c->bn_width, nb = c->tra_bands;
    const int pad = (nb - (Wb % nb)) % nb;
    const int bw = (Wb + pad) / nb;
    const int H = B->tra.H;
    const float inv_bw = 1.0f / (float)bw;

    for (int b = 0; b < nb; ++b) {
        for (int ch = 0; ch < half; ++ch) {
            const float *xr = x + (size_t)ch * Wb;
            float acc = 0.0f;
            for (int j = 0; j < bw; ++j) {
                const int idx = b * bw + j;
                if (idx < Wb) acc += xr[idx] * xr[idx];
            }
            g->tra_zt[ch] = acc * inv_bw;
        }
        gt_gru_step(&B->tra, g->tra_zt, B->tra_h + (size_t)b * H,
                    g->gruscratch, g->q8);
        gt_linear_v(&B->tra_fc, B->tra_h + (size_t)b * H, g->tra_gain, g->q8);
        for (int ch = 0; ch < half; ++ch)
            g->tra_gain[ch] = gt_sigmoid(g->tra_gain[ch]);
        for (int ch = 0; ch < half; ++ch) {
            float *xr = x + (size_t)ch * Wb;
            const float gn = g->tra_gain[ch];
            for (int j = 0; j < bw; ++j) {
                const int idx = b * bw + j;
                if (idx < Wb) xr[idx] *= gn;
            }
        }
    }
}

static void block_forward(gtcrn_t *g, gt_block_t *B, const float *x, float *out)
{
    const gtcrn_cfg_t *c = &g->cfg;
    const int C = c->base_channels, half = C / 2, Wb = c->bn_width;
    const int frame_len = C * Wb;
    const float *x2 = x + (size_t)half * Wb;

    gt_sfe3(x, half, Wb, g->sfe_h);                       /* 3*half x Wb */
    gt_conv1x1(&B->pc1, g->sfe_h, 3 * half, Wb, g->hid, g->q8);
    gt_prelu(g->hid, C * Wb, B->pc1_a);

    /* causal dilated depthwise: taps at t-2d, t-d, t.  The time padding in the
     * model is zeros (F.pad default), so the ring starts zeroed -- do not
     * replicate here, unlike the feature front-end. */
    hist_push(B, g->hid, frame_len);
    const float *t0 = hist_at(B, 2 * B->dilation, frame_len);
    const float *t1 = hist_at(B, B->dilation, frame_len);
    gt_depthwise33(t0, t1, g->hid, C, Wb, B->dw_w, B->dw_b, g->hid2);
    gt_prelu(g->hid2, C * Wb, B->dw_a);

    gt_conv1x1(&B->pc2, g->hid2, C, Wb, g->half, g->q8);  /* half x Wb */
    band_tra(g, B, g->half);

    /* ShuffleNet interleave: out[2i] = processed, out[2i+1] = carried through */
    for (int i = 0; i < half; ++i) {
        memcpy(out + (size_t)(2 * i) * Wb, g->half + (size_t)i * Wb,
               (size_t)Wb * sizeof(float));
        memcpy(out + (size_t)(2 * i + 1) * Wb, x2 + (size_t)i * Wb,
               (size_t)Wb * sizeof(float));
    }
}

/* ========================================================================= */
/* DPGRNN                                                                    */
/* ========================================================================= */
static void dpgrnn(gtcrn_t *g, gt_dp_t *D, float *x)
{
    const gtcrn_cfg_t *c = &g->cfg;
    const int C = c->base_channels, Wb = c->bn_width;
    const int h2 = C / 2, Hb = C / 4;
    const int n = Wb * C;

    for (int f = 0; f < Wb; ++f)
        for (int ch = 0; ch < C; ++ch)
            g->dpa[f * C + ch] = x[(size_t)ch * Wb + f];

    /* ---- intra: bidirectional across frequency, state reset every frame ----
     * Frequency has no causality constraint, so this pass is allowed to look
     * "ahead" in frequency. It must NOT carry state across frames. */
    for (int r = 0; r < 2; ++r) {
        const int off = r * h2;
        memset(g->gru_h, 0, (size_t)Hb * sizeof(float));
        for (int f = 0; f < Wb; ++f) {
            gt_gru_step(&D->ia[r][0], g->dpa + f * C + off, g->gru_h,
                        g->gruscratch, g->q8);
            memcpy(g->dpb + f * C + off, g->gru_h, (size_t)Hb * sizeof(float));
        }
        memset(g->gru_h, 0, (size_t)Hb * sizeof(float));
        for (int f = Wb - 1; f >= 0; --f) {
            gt_gru_step(&D->ia[r][1], g->dpa + f * C + off, g->gru_h,
                        g->gruscratch, g->q8);
            memcpy(g->dpb + f * C + off + Hb, g->gru_h, (size_t)Hb * sizeof(float));
        }
    }
    gt_qlinear_m(&D->ia_fc, g->dpb, Wb, C, g->dpc, g->q8);
    gt_layernorm(g->dpc, n, D->ia_lnw, D->ia_lnb);
    for (int i = 0; i < n; ++i) g->dpa[i] += g->dpc[i];    /* intra_out */

    /* ---- inter: one GRU step per frequency, state carried across frames ---- */
    for (int f = 0; f < Wb; ++f) {
        float *st = D->inter_h + (size_t)f * C;
        gt_gru_step(&D->ie[0], g->dpa + f * C, st, g->gruscratch, g->q8);
        gt_gru_step(&D->ie[1], g->dpa + f * C + h2, st + h2, g->gruscratch, g->q8);
        memcpy(g->dpb + f * C, st, (size_t)C * sizeof(float));
    }
    gt_qlinear_m(&D->ie_fc, g->dpb, Wb, C, g->dpc, g->q8);
    gt_layernorm(g->dpc, n, D->ie_lnw, D->ie_lnb);
    for (int i = 0; i < n; ++i) g->dpa[i] += g->dpc[i];

    for (int ch = 0; ch < C; ++ch)
        for (int f = 0; f < Wb; ++f)
            x[(size_t)ch * Wb + f] = g->dpa[f * C + ch];
}

/* ========================================================================= */
/* one spectral frame                                                        */
/* ========================================================================= */
static void gtcrn_encode_frame(gtcrn_t *g, const float *re, const float *im)
{
    const gtcrn_cfg_t *c = &g->cfg;
    const int F = c->n_freqs, W = c->width, Wb = c->bn_width;
    const int C = c->base_channels, MW = g->mid_width;
    const int nd = c->n_dil;

    for (int f = 0; f < F; ++f)
        g->pwr[f] = sqrtf(re[f] * re[f] + im[f] * im[f] + 1e-12f);

    memcpy(g->magbuf, g->pwr, (size_t)F * sizeof(float));
    feature_front(g, re, im, g->magbuf);

    /* ---- encoder ---- */
    erb_bm(g, g->feat, g->erbed);
    gt_sfe3(g->erbed, 4, W, g->sfe_in);
    gt_conv1x5_s2(&g->enc_c1, g->sfe_in, 12, W, g->enc_out0, MW, g->q8);
    gt_prelu(g->enc_out0, C * MW, g->enc_c1_a);
    gt_conv1x5_s2(&g->enc_c2, g->enc_out0, C, MW, g->enc_out[0], Wb, g->q8);
    gt_prelu(g->enc_out[0], C * Wb, g->enc_c2_a);
    for (int i = 0; i < nd; ++i)
        block_forward(g, &g->enc[i], g->enc_out[i], g->enc_out[i + 1]);
}

static void gtcrn_backend_inplace(gtcrn_t *g, const float *re, const float *im)
{
    const gtcrn_cfg_t *c = &g->cfg;
    const int F = c->n_freqs, W = c->width, Wb = c->bn_width;
    const int C = c->base_channels, MW = g->mid_width;
    const int nd = c->n_dil;

    /* ---- bottleneck ---- */
    memcpy(g->xa, g->enc_out[nd], (size_t)C * Wb * sizeof(float));
    for (int k = 0; k < c->n_dpgrnn; ++k)
        dpgrnn(g, &g->dp[k], g->xa);

    /* ---- decoder, with mirrored skip connections ---- */
    for (int i = 0; i < nd; ++i) {
        const float *skip = g->enc_out[nd - i];
        for (int j = 0; j < C * Wb; ++j)
            g->xb[j] = g->xa[j] + skip[j];
        block_forward(g, &g->dec[i], g->xb, g->xa);
    }
    for (int j = 0; j < C * Wb; ++j)
        g->xb[j] = g->xa[j] + g->enc_out[0][j];
    gt_deconv1x5_s2(&g->dec_c1, g->xb, C, Wb, g->mid, MW, g->q8);
    gt_prelu(g->mid, C * MW, g->dec_c1_a);
    for (int j = 0; j < C * MW; ++j)
        g->mid[j] += g->enc_out0[j];
    gt_deconv1x5_s2(&g->head, g->mid, C, MW, g->headbuf, W, g->q8);

    /* ---- bounded, phase-decoupled mask ---- */
    erb_bs(g, g->headbuf, W, g->maskbuf);
    const float span = c->mask_max - c->mask_min;
    const float *m0 = g->maskbuf;
    const float *m1 = g->maskbuf + F;
    const float *m2 = g->maskbuf + 2 * F;
    for (int f = 0; f < F; ++f) {
        const float mm = c->mask_min + span * gt_sigmoid(m0[f]);
        const float pr = 1.0f + m1[f], pi = m2[f];
        const float pn = sqrtf(pr * pr + pi * pi + 1e-8f);
        const float mr = mm * pr / pn, mi = mm * pi / pn;
        g->er[f] = re[f] * mr - im[f] * mi;
        g->ei[f] = re[f] * mi + im[f] * mr;
    }

    /* ---- deep filter residual over low band ---- */
    const int K = c->df_order, Fd = c->df_bins;
    if (K > 0 && Fd > 0) {
        memcpy(g->df_r + (size_t)g->df_pos * Fd, g->er, (size_t)Fd * sizeof(float));
        memcpy(g->df_i + (size_t)g->df_pos * Fd, g->ei, (size_t)Fd * sizeof(float));
        g->df_pos = (g->df_pos + 1) % K;

        memset(g->dfr, 0, (size_t)Fd * sizeof(float));
        memset(g->dfi, 0, (size_t)Fd * sizeof(float));
        for (int k = 0; k < K; ++k) {
            int idx = g->df_pos - 1 - k;
            while (idx < 0) idx += K;
            const float *lr = g->df_r + (size_t)idx * Fd;
            const float *li = g->df_i + (size_t)idx * Fd;
            const float *cr = g->headbuf + (size_t)(3 + 2 * k) * W;
            const float *ci = g->headbuf + (size_t)(3 + 2 * k + 1) * W;
            for (int f = 0; f < Fd; ++f) {
                g->dfr[f] += cr[f] * lr[f] - ci[f] * li[f];
                g->dfi[f] += cr[f] * li[f] + ci[f] * lr[f];
            }
        }
        for (int f = 0; f < Fd; ++f) {
            g->er[f] += g->dfr[f];
            g->ei[f] += g->dfi[f];
        }
    }
}

static void gtcrn_frame(gtcrn_t *g, const float *re, const float *im)
{
    gtcrn_encode_frame(g, re, im);
    gtcrn_backend_inplace(g, re, im);
}

/* ========================================================================= */
/* public API                                                                */
/* ========================================================================= */
static int parse_header(gtcrn_t *g, const uint8_t *b, size_t len)
{
    if (len < GT_HEADER_BYTES || memcmp(b, "GTCRNQ8\0", 8) != 0) return -1;
    uint32_t ver, n_t, dir_off, data_off, total;
    memcpy(&ver, b + 8, 4);
    memcpy(&n_t, b + 12, 4);
    memcpy(&dir_off, b + 16, 4);
    memcpy(&data_off, b + 20, 4);
    memcpy(&total, b + 24, 4);
    if (ver != 1 || total > len) return -1;

    const uint32_t *u = (const uint32_t *)(b + 32);
    const int32_t  *s = (const int32_t *)(b + 32 + 48);
    const uint32_t *v = (const uint32_t *)(b + 32 + 80);
    const float    *f = (const float *)(b + 32 + 92);
    const uint32_t *w = (const uint32_t *)(b + 32 + 104);
    const float    *y = (const float *)(b + 32 + 116);

    gtcrn_cfg_t *c = &g->cfg;
    c->sample_rate = u[0]; c->n_fft = u[1]; c->hop_length = u[2];
    c->n_freqs = u[3]; c->erb1 = u[4]; c->erb2 = u[5];
    c->width = u[6]; c->bn_width = u[7]; c->base_channels = u[8];
    c->n_dpgrnn = u[9]; c->tra_bands = u[10]; c->n_dil = u[11];
    for (int i = 0; i < 8; ++i) c->dil[i] = s[i];
    c->df_order = v[0]; c->df_bins = v[1]; c->n_out = v[2];
    c->mask_max = f[0]; c->mask_min = f[1]; c->compress = f[2];
    c->level_frames = w[0]; c->smooth_frames = w[1]; c->nf_frames = w[2];
    c->nf_bias = y[0]; c->snr_lo = y[1]; c->snr_hi = y[2]; c->mix_rms = y[3];

    g->dir = (const gt_dirent_t *)(b + dir_off);
    g->base = b + data_off;
    g->n_tensors = (int)n_t;
    return 0;
}

static int build_erb(gtcrn_t *g)
{
    int n;
    g->bm_start = load_i32(g, "erb.bm.start", &n);
    g->bm_len   = load_i32(g, "erb.bm.len", NULL);
    g->bs_start = load_i32(g, "erb.bs.start", NULL);
    g->bs_len   = load_i32(g, "erb.bs.len", NULL);
    const gt_dirent_t *e = find(g, "erb.bm.w");
    if (!e) return -1;
    g->bm_w = (const float *)tdata(g, e);
    e = find(g, "erb.bs.w");
    if (!e) return -1;
    g->bs_w = (const float *)tdata(g, e);
    if (!g->bm_start || !g->bm_len || !g->bs_start || !g->bs_len) return -1;

    const int nhi = g->cfg.n_freqs - g->cfg.erb1;
    g->bm_off = (int32_t *)alloc_fast(g, (size_t)g->cfg.erb2 * sizeof(int32_t));
    g->bs_off = (int32_t *)alloc_fast(g, (size_t)nhi * sizeof(int32_t));
    if (!g->bm_off || !g->bs_off) return -1;
    int32_t acc = 0;
    for (int i = 0; i < g->cfg.erb2; ++i) { g->bm_off[i] = acc; acc += g->bm_len[i]; }
    acc = 0;
    for (int i = 0; i < nhi; ++i) { g->bs_off[i] = acc; acc += g->bs_len[i]; }
    return 0;
}

static int build_block(gtcrn_t *g, gt_block_t *B, const char *pfx, int dilation)
{
    char p[GT_NAME_LEN];
    const int C = g->cfg.base_channels, Wb = g->cfg.bn_width;

    snprintf(p, sizeof p, "%spc1.", pfx);
    if (load_qw(g, &B->pc1, p, "w", "b", 1, 1)) return -1;
    const float *a = load_f32(g, p, "a");
    if (!a) return -1;
    B->pc1_a = a[0];

    snprintf(p, sizeof p, "%sdw.", pfx);
    B->dw_w = load_f32(g, p, "w");
    B->dw_b = load_f32(g, p, "b");
    a = load_f32(g, p, "a");
    if (!B->dw_w || !B->dw_b || !a) return -1;
    B->dw_a = a[0];

    snprintf(p, sizeof p, "%spc2.", pfx);
    if (load_qw(g, &B->pc2, p, "w", "b", 1, 1)) return -1;

    snprintf(p, sizeof p, "%stra.", pfx);
    if (load_gru(g, &B->tra, p)) return -1;
    snprintf(p, sizeof p, "%stra.fc.", pfx);
    if (load_qw(g, &B->tra_fc, p, "w", "b", 1, 1)) return -1;

    B->dilation   = dilation;
    B->hist_n     = 2 * dilation + 1;
    B->hist_pos   = 0;
    B->hist       = BIG(g, (size_t)B->hist_n * C * Wb);
    B->tra_h      = FAST(g, (size_t)g->cfg.tra_bands * B->tra.H);
    return (B->hist && B->tra_h) ? 0 : -1;
}

static int build_dp(gtcrn_t *g, gt_dp_t *D, int k)
{
    char p[GT_NAME_LEN];
    const int C = g->cfg.base_channels, Wb = g->cfg.bn_width;
    static const char *rn[2] = {"ia1.", "ia2."};
    static const char *dr[2] = {"f.", "b."};

    for (int r = 0; r < 2; ++r)
        for (int d = 0; d < 2; ++d) {
            snprintf(p, sizeof p, "dp%d.%s%s", k, rn[r], dr[d]);
            if (load_gru(g, &D->ia[r][d], p)) return -1;
        }
    snprintf(p, sizeof p, "dp%d.ia.fc.", k);
    if (load_qw(g, &D->ia_fc, p, "w", "b", 1, 1)) return -1;
    snprintf(p, sizeof p, "dp%d.ia.ln.", k);
    D->ia_lnw = load_f32(g, p, "w");
    D->ia_lnb = load_f32(g, p, "b");

    for (int r = 0; r < 2; ++r) {
        snprintf(p, sizeof p, "dp%d.ie%d.", k, r + 1);
        if (load_gru(g, &D->ie[r], p)) return -1;
    }
    snprintf(p, sizeof p, "dp%d.ie.fc.", k);
    if (load_qw(g, &D->ie_fc, p, "w", "b", 1, 1)) return -1;
    snprintf(p, sizeof p, "dp%d.ie.ln.", k);
    D->ie_lnw = load_f32(g, p, "w");
    D->ie_lnb = load_f32(g, p, "b");
    if (!D->ia_lnw || !D->ia_lnb || !D->ie_lnw || !D->ie_lnb) return -1;

    D->inter_h = FAST(g, (size_t)Wb * C);
    return D->inter_h ? 0 : -1;
}

gtcrn_t *gtcrn_create(const void *blob, size_t blob_len)
{
    gtcrn_t *g = (gtcrn_t *)calloc(1, sizeof(gtcrn_t));
    if (!g) return NULL;
    if (parse_header(g, (const uint8_t *)blob, blob_len)) { free(g); return NULL; }

    gtcrn_cfg_t *c = &g->cfg;
    const int F = c->n_freqs, W = c->width, Wb = c->bn_width;
    const int C = c->base_channels, nd = c->n_dil, N = c->n_fft;
    g->mid_width = (W + 4 - 5) / 2 + 1;
    const int MW = g->mid_width;

    /* ---- top-level layers ---- */
    if (load_qw(g, &g->enc_c1, "enc.c1.", "w", "b", 5, 1)) goto fail;
    if (load_qw(g, &g->enc_c2, "enc.c2.", "w", "b", 5, 2)) goto fail;
    if (load_qw(g, &g->dec_c1, "dec.c1.", "w", "b", 5, 2)) goto fail;
    if (load_qw(g, &g->head,   "dec.head.", "w", "b", 5, 1)) goto fail;
    {
        const float *a1 = load_f32(g, "enc.c1.", "a");
        const float *a2 = load_f32(g, "enc.c2.", "a");
        const float *a3 = load_f32(g, "dec.c1.", "a");
        if (!a1 || !a2 || !a3) goto fail;
        g->enc_c1_a = a1[0];
        g->enc_c2_a = a2[0];
        g->dec_c1_a = a3[0];
    }
    if (build_erb(g)) goto fail;

    g->enc = (gt_block_t *)alloc_fast(g, (size_t)nd * sizeof(gt_block_t));
    g->dec = (gt_block_t *)alloc_fast(g, (size_t)nd * sizeof(gt_block_t));
    g->dp  = (gt_dp_t *)alloc_fast(g, (size_t)c->n_dpgrnn * sizeof(gt_dp_t));
    if (!g->enc || !g->dec || !g->dp) goto fail;

    for (int i = 0; i < nd; ++i) {
        char pfx[GT_NAME_LEN];
        snprintf(pfx, sizeof pfx, "enc.g%d.", i);
        if (build_block(g, &g->enc[i], pfx, c->dil[i])) goto fail;
        snprintf(pfx, sizeof pfx, "dec.g%d.", i);
        if (build_block(g, &g->dec[i], pfx, c->dil[nd - 1 - i])) goto fail;
    }
    for (int k = 0; k < c->n_dpgrnn; ++k)
        if (build_dp(g, &g->dp[k], k)) goto fail;

    /* ---- state ---- */
    g->lvl_ring = FAST(g, c->level_frames);
    g->smo_ring = FAST(g, (size_t)c->smooth_frames * F);
    g->nf_ring  = BIG(g, (size_t)c->nf_frames * F);
    g->df_r     = FAST(g, (size_t)(c->df_order > 0 ? c->df_order : 1) * c->df_bins);
    g->df_i     = FAST(g, (size_t)(c->df_order > 0 ? c->df_order : 1) * c->df_bins);

    /* ---- STFT ---- */
    g->win    = FAST(g, N);
    g->ana    = FAST(g, N);
    g->ola    = FAST(g, N);
    g->fft_re = FAST(g, N);
    g->fft_im = FAST(g, N);
    g->tw_re  = FAST(g, N / 2);
    g->tw_im  = FAST(g, N / 2);
    g->brev   = (uint16_t *)alloc_fast(g, (size_t)N * sizeof(uint16_t));
    if (!g->win || !g->brev) goto fail;
    gt_fft_init(N, g->tw_re, g->tw_im, g->brev);
    for (int i = 0; i < N; ++i) {
        float h = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)N);
        g->win[i] = sqrtf(h > 0.0f ? h : 0.0f);
    }

    /* ---- scratch ---- */
    g->pwr      = FAST(g, F);
    g->magbuf   = FAST(g, F);
    g->psmooth  = FAST(g, F);
    g->nfloor   = FAST(g, F);
    g->feat     = FAST(g, 4 * F);
    g->erbed    = FAST(g, 4 * W);
    g->sfe_in   = FAST(g, 12 * W);
    g->enc_out0 = FAST(g, (size_t)C * MW);
    g->enc_out  = (float **)alloc_fast(g, (size_t)(nd + 1) * sizeof(float *));
    if (!g->enc_out) goto fail;
    for (int i = 0; i <= nd; ++i) g->enc_out[i] = FAST(g, (size_t)C * Wb);
    g->xa       = FAST(g, (size_t)C * Wb);
    g->xb       = FAST(g, (size_t)C * Wb);
    g->sfe_h    = FAST(g, (size_t)(3 * (C / 2)) * Wb);
    g->hid      = FAST(g, (size_t)C * Wb);
    g->hid2     = FAST(g, (size_t)C * Wb);
    g->half     = FAST(g, (size_t)(C / 2) * Wb);
    g->mid      = FAST(g, (size_t)C * MW);
    g->headbuf  = FAST(g, (size_t)c->n_out * W);
    g->maskbuf  = FAST(g, 3 * F);
    g->er       = FAST(g, F);
    g->ei       = FAST(g, F);
    g->dfr      = FAST(g, c->df_bins > 0 ? c->df_bins : 1);
    g->dfi      = FAST(g, c->df_bins > 0 ? c->df_bins : 1);
    g->dpa      = FAST(g, (size_t)Wb * C);
    g->dpb      = FAST(g, (size_t)Wb * C);
    g->dpc      = FAST(g, (size_t)Wb * C);
    g->tra_zt   = FAST(g, C);
    g->tra_gain = FAST(g, C);
    g->gru_h    = FAST(g, C);
    g->gruscratch = FAST(g, 12 * C);          /* 6*H, H <= 2*C                */

    /* worst-case quantisation scratch: the widest transposed+padded frame */
    size_t q = (size_t)(W + 4) * 12;
    size_t q2 = (size_t)(MW + 4) * C;
    size_t q3 = (size_t)(Wb + 4) * (3 * (C / 2));
    size_t q4 = (size_t)Wb * C;
    if (q2 > q) q = q2;
    if (q3 > q) q = q3;
    if (q4 > q) q = q4;
    g->q8 = (int8_t *)alloc_fast(g, q + 64);

    if (!g->q8 || !g->dpc || !g->headbuf || !g->nf_ring) goto fail;
    gtcrn_reset(g);
    return g;

fail:
    gtcrn_destroy(g);
    return NULL;
}

void gtcrn_destroy(gtcrn_t *g)
{
    if (!g) return;
    for (int i = 0; i < g->n_allocs; ++i) free(g->allocs[i]);
    free(g);
}

const gtcrn_cfg_t *gtcrn_config(const gtcrn_t *g) { return &g->cfg; }
int gtcrn_latency_samples(const gtcrn_t *g) { return g->cfg.hop_length; }
size_t gtcrn_mem_internal(const gtcrn_t *g) { return g->mem_int; }
size_t gtcrn_mem_psram(const gtcrn_t *g) { return g->mem_ps; }
uint32_t gtcrn_last_frame_us(const gtcrn_t *g) { return g->last_us; }
void gtcrn_set_mask_min(gtcrn_t *g, float v) { g->cfg.mask_min = v; }

void gtcrn_reset(gtcrn_t *g)
{
    const gtcrn_cfg_t *c = &g->cfg;
    const int C = c->base_channels, Wb = c->bn_width, F = c->n_freqs;

    g->front_primed = 0;
    g->lvl_pos = g->smo_pos = g->nf_pos = g->df_pos = 0;
    memset(g->lvl_ring, 0, (size_t)c->level_frames * sizeof(float));
    memset(g->smo_ring, 0, (size_t)c->smooth_frames * F * sizeof(float));
    memset(g->nf_ring, 0, (size_t)c->nf_frames * F * sizeof(float));
    if (c->df_order > 0 && c->df_bins > 0) {
        memset(g->df_r, 0, (size_t)c->df_order * c->df_bins * sizeof(float));
        memset(g->df_i, 0, (size_t)c->df_order * c->df_bins * sizeof(float));
    }
    memset(g->ana, 0, (size_t)c->n_fft * sizeof(float));
    memset(g->ola, 0, (size_t)c->n_fft * sizeof(float));

    for (int i = 0; i < c->n_dil; ++i) {
        gt_block_t *B[2] = {&g->enc[i], &g->dec[i]};
        for (int s = 0; s < 2; ++s) {
            memset(B[s]->hist, 0, (size_t)B[s]->hist_n * C * Wb * sizeof(float));
            B[s]->hist_pos = 0;
            memset(B[s]->tra_h, 0, (size_t)c->tra_bands * B[s]->tra.H * sizeof(float));
        }
    }
    for (int k = 0; k < c->n_dpgrnn; ++k)
        memset(g->dp[k].inter_h, 0, (size_t)Wb * C * sizeof(float));
}

gtcrn_frame_t *gtcrn_frame_create(const gtcrn_t *g)
{
    if (!g) return NULL;

    const int C = g->cfg.base_channels;
    const int Wb = g->cfg.bn_width;
    const int F = g->cfg.n_freqs;
    const int MW = g->mid_width;
    const int nd = g->cfg.n_dil;

    gtcrn_frame_t *f = (gtcrn_frame_t *)calloc(1, sizeof(*f));
    if (!f) return NULL;

    f->C = C; f->Wb = Wb; f->F = F; f->MW = MW; f->nd = nd;

    const size_t n_reim = (size_t)2 * F;
    const size_t n_enc0 = (size_t)C * MW;
    const size_t n_enc = (size_t)(nd + 1) * C * Wb;
    const size_t n_out = (size_t)2 * F;
    const size_t total = n_reim + n_enc0 + n_enc + n_out;

#ifdef ESP_PLATFORM
    f->storage = heap_caps_malloc(total * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!f->storage)
        f->storage = heap_caps_malloc(total * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#else
    f->storage = malloc(total * sizeof(float));
#endif
    if (!f->storage) {
        free(f);
        return NULL;
    }
    memset(f->storage, 0, total * sizeof(float));

    float *p = (float *)f->storage;
    f->re = p; p += F;
    f->im = p; p += F;
    f->enc_out0 = p; p += n_enc0;

    f->enc_out = (float **)calloc((size_t)(nd + 1), sizeof(float *));
    if (!f->enc_out) {
        free(f->storage);
        free(f);
        return NULL;
    }
    for (int i = 0; i <= nd; ++i) {
        f->enc_out[i] = p;
        p += (size_t)C * Wb;
    }

    f->out_re = p; p += F;
    f->out_im = p;
    return f;
}

void gtcrn_frame_destroy(gtcrn_frame_t *f)
{
    if (!f) return;
    free(f->enc_out);
    free(f->storage);
    free(f);
}

void gtcrn_encode_hop(gtcrn_t *g, const float *in, gtcrn_frame_t *f)
{
    if (!g || !in || !f) return;

    const gtcrn_cfg_t *c = &g->cfg;
    const int N = c->n_fft, hop = c->hop_length, F = c->n_freqs;

    memmove(g->ana, g->ana + hop, (size_t)(N - hop) * sizeof(float));
    memcpy(g->ana + N - hop, in, (size_t)hop * sizeof(float));

    for (int i = 0; i < N; ++i) {
        g->fft_re[i] = g->ana[i] * g->win[i];
        g->fft_im[i] = 0.0f;
    }
    gt_fft(g->fft_re, g->fft_im, N, g->tw_re, g->tw_im, g->brev, 0);

    memcpy(f->re, g->fft_re, (size_t)F * sizeof(float));
    memcpy(f->im, g->fft_im, (size_t)F * sizeof(float));

    gtcrn_encode_frame(g, f->re, f->im);

    const int C = c->base_channels, Wb = c->bn_width, MW = g->mid_width, nd = c->n_dil;
    memcpy(f->enc_out0, g->enc_out0, (size_t)C * MW * sizeof(float));
    for (int i = 0; i <= nd; ++i)
        memcpy(f->enc_out[i], g->enc_out[i], (size_t)C * Wb * sizeof(float));
}

void gtcrn_backend_frame(gtcrn_t *g, gtcrn_frame_t *f)
{
    if (!g || !f) return;

    const gtcrn_cfg_t *c = &g->cfg;
    const int C = c->base_channels, Wb = c->bn_width, MW = g->mid_width, nd = c->n_dil;

    memcpy(g->enc_out0, f->enc_out0, (size_t)C * MW * sizeof(float));
    for (int i = 0; i <= nd; ++i)
        memcpy(g->enc_out[i], f->enc_out[i], (size_t)C * Wb * sizeof(float));

    gtcrn_backend_inplace(g, f->re, f->im);
    memcpy(f->out_re, g->er, (size_t)f->F * sizeof(float));
    memcpy(f->out_im, g->ei, (size_t)f->F * sizeof(float));
}

void gtcrn_synthesize_hop(gtcrn_t *g, const gtcrn_frame_t *f, float *out)
{
    if (!g || !f || !out) return;

    const gtcrn_cfg_t *c = &g->cfg;
    const int N = c->n_fft, hop = c->hop_length, F = c->n_freqs;

    for (int k = 0; k < F; ++k) {
        g->fft_re[k] = f->out_re[k];
        g->fft_im[k] = f->out_im[k];
    }
    for (int k = F; k < N; ++k) {
        g->fft_re[k] = f->out_re[N - k];
        g->fft_im[k] = -f->out_im[N - k];
    }
    gt_fft(g->fft_re, g->fft_im, N, g->tw_re, g->tw_im, g->brev, 1);

    for (int i = 0; i < N; ++i)
        g->ola[i] += g->fft_re[i] * g->win[i];

    memcpy(out, g->ola, (size_t)hop * sizeof(float));
    memmove(g->ola, g->ola + hop, (size_t)(N - hop) * sizeof(float));
    memset(g->ola + N - hop, 0, (size_t)hop * sizeof(float));
}

void gtcrn_process_hop(gtcrn_t *g, const float *in, float *out)
{
    const gtcrn_cfg_t *c = &g->cfg;
    const int N = c->n_fft, hop = c->hop_length, F = c->n_freqs;

#ifdef ESP_PLATFORM
    const int64_t t0 = esp_timer_get_time();
#endif

    /* slide the analysis window. The buffer starts zeroed, which reproduces
     * torch.stft(center=True) with zero padding -- the training code reflects
     * instead, but that only differs over the first half-frame of a stream. */
    memmove(g->ana, g->ana + hop, (size_t)(N - hop) * sizeof(float));
    memcpy(g->ana + N - hop, in, (size_t)hop * sizeof(float));

    for (int i = 0; i < N; ++i) { g->fft_re[i] = g->ana[i] * g->win[i]; g->fft_im[i] = 0.0f; }
    gt_fft(g->fft_re, g->fft_im, N, g->tw_re, g->tw_im, g->brev, 0);

    gtcrn_frame(g, g->fft_re, g->fft_im);

    for (int f = 0; f < F; ++f) { g->fft_re[f] = g->er[f]; g->fft_im[f] = g->ei[f]; }
    for (int f = F; f < N; ++f) { g->fft_re[f] = g->er[N - f]; g->fft_im[f] = -g->ei[N - f]; }
    gt_fft(g->fft_re, g->fft_im, N, g->tw_re, g->tw_im, g->brev, 1);

    /* analysis and synthesis windows multiply to a periodic Hann, which sums to
     * exactly 1.0 at 50 % overlap, so no window-sum normalisation is needed. */
    for (int i = 0; i < N; ++i) g->ola[i] += g->fft_re[i] * g->win[i];
    memcpy(out, g->ola, (size_t)hop * sizeof(float));
    memmove(g->ola, g->ola + hop, (size_t)(N - hop) * sizeof(float));
    memset(g->ola + N - hop, 0, (size_t)hop * sizeof(float));

#ifdef ESP_PLATFORM
    g->last_us = (uint32_t)(esp_timer_get_time() - t0);
#endif
}
