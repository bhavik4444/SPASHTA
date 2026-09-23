/*
 * gtcrn.h -- streaming GTCRN-DF speech enhancement for ESP32-S3.
 *
 * A direct C implementation of the PyTorch model, not a converted graph. It
 * consumes one hop of samples and produces one hop, holding all recurrent and
 * convolutional history internally, so memory is O(1) in stream length and an
 * hour of audio costs exactly as much RAM as a second does.
 *
 * Numerics: int8 weights with per-output-row scales and dynamically quantised
 * activations for every matmul-shaped layer; float32 everywhere the model is
 * sensitive (feature front-end, depthwise kernels, mask, deep filter, STFT).
 * tools/gtcrn_ref.py is the bit-level specification.
 *
 * Latency: exactly one hop (16 ms at 16 kHz / hop 256). The model itself is
 * causal; nothing here looks ahead.
 *
 * Thread-safety: a gtcrn_t is not re-entrant. One instance per stream, used
 * from one task.
 */
#ifndef GTCRN_H
#define GTCRN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int   sample_rate;
    int   n_fft;
    int   hop_length;
    int   n_freqs;          /* n_fft/2 + 1                                  */
    int   erb1;             /* linear bins kept at full resolution          */
    int   erb2;             /* compressed ERB bands above that              */
    int   width;            /* erb1 + erb2, the encoder input width         */
    int   bn_width;         /* width after two stride-2 convs               */
    int   base_channels;
    int   n_dpgrnn;
    int   tra_bands;
    int   n_dil;
    int   dil[8];
    int   df_order;         /* deep filter taps                             */
    int   df_bins;          /* low bins the deep filter covers              */
    int   n_out;            /* decoder head channels = 3 + 2*df_order       */
    float mask_max, mask_min, compress;
    int   level_frames, smooth_frames, nf_frames;
    float nf_bias, snr_lo, snr_hi;
    float mix_rms;          /* loudness the model was trained at            */
} gtcrn_cfg_t;

typedef struct gtcrn gtcrn_t;

/* Create a runtime from an exported blob. The blob must stay valid for the
 * lifetime of the handle (embedding it in flash is fine -- it is read-only and
 * memory-mapped). Returns NULL on a bad blob or on allocation failure. */
gtcrn_t *gtcrn_create(const void *blob, size_t blob_len);
void     gtcrn_destroy(gtcrn_t *g);

const gtcrn_cfg_t *gtcrn_config(const gtcrn_t *g);

/* Clear every piece of history. Call between unrelated streams; do NOT call it
 * inside one stream, or the noise-floor tracker restarts from scratch and the
 * first second of output degrades. */
void gtcrn_reset(gtcrn_t *g);

/* Mask magnitude floor. 0.0 = full suppression in the pauses; ~0.02 (-34 dB)
 * leaves a faint noise bed, which some listeners prefer. Free to change at any
 * time -- it has no learnable parameters behind it. */
void gtcrn_set_mask_min(gtcrn_t *g, float v);

/* Push exactly cfg->hop_length samples, get exactly cfg->hop_length samples.
 * in and out may alias. The output lags the input by one hop. */
void gtcrn_process_hop(gtcrn_t *g, const float *in, float *out);

/* Samples of algorithmic delay between the input and output streams. */
int gtcrn_latency_samples(const gtcrn_t *g);

/* Bytes of working memory the handle owns, for logging a memory report. */
size_t gtcrn_mem_internal(const gtcrn_t *g);
size_t gtcrn_mem_psram(const gtcrn_t *g);

/* Rough per-frame cost, filled in by gtcrn_process_hop when profiling is on. */
uint32_t gtcrn_last_frame_us(const gtcrn_t *g);

/* Two-core pipeline API. A frame object is owned by the caller and can be
 * recycled after gtcrn_synthesize_hop() returns. */
typedef struct gtcrn_frame gtcrn_frame_t;
gtcrn_frame_t *gtcrn_frame_create(const gtcrn_t *g);
void gtcrn_frame_destroy(gtcrn_frame_t *f);
void gtcrn_encode_hop(gtcrn_t *g, const float *in, gtcrn_frame_t *f);
void gtcrn_backend_frame(gtcrn_t *g, gtcrn_frame_t *f);
void gtcrn_synthesize_hop(gtcrn_t *g, const gtcrn_frame_t *f, float *out);

#ifdef __cplusplus
}
#endif
#endif /* GTCRN_H */
