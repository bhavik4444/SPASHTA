/*
 * gtcrn_ops.c -- the numeric core.
 *
 * Every routine here has a twin in tools/gtcrn_ref.py. Where the two could
 * plausibly disagree -- rounding at exact .5, the tap parity of the transposed
 * convolution, the order of the GRU gates -- the comment says which convention
 * is in force. Those are the places where a "working" port silently loses 3 dB.
 */
#include <math.h>
#include <string.h>

#include "gtcrn_internal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* -------------------------------------------------------------------------
 * int8 dot product
 *
 * ESP32-S3 uses its 128-bit SIMD MAC instruction for n >= 16.
 * The SIMD routine returns the same raw int32 dot-product result expected
 * by the rest of this runtime.
 *
 * For small vectors, plain C is used because the SIMD setup overhead is
 * not worthwhile.
 * ------------------------------------------------------------------------- */
#if defined(CONFIG_IDF_TARGET_ESP32S3)
extern int32_t gt_dot_i8_s3(
    const int8_t *a,
    const int8_t *b,
    int n
);
#endif

int32_t gt_dot_i8(const int8_t *a, const int8_t *b, int n)
{
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    /*
     * Large dot products use the S3 SIMD MAC kernel.
     * This is the hot path for the convolution/linear layers.
     */
    if (n >= 16)
        return gt_dot_i8_s3(a, b, n);
#endif

    /*
     * The RT-lite model has many tiny GRU dot products with n=12, 8 and 4.
     * Keep these completely unrolled so the compiler emits straight-line
     * multiply/add code rather than a loop with counter/branch overhead.
     */
    if (n == 12) {
        return
            (int32_t)a[0]  * (int32_t)b[0]  +
            (int32_t)a[1]  * (int32_t)b[1]  +
            (int32_t)a[2]  * (int32_t)b[2]  +
            (int32_t)a[3]  * (int32_t)b[3]  +
            (int32_t)a[4]  * (int32_t)b[4]  +
            (int32_t)a[5]  * (int32_t)b[5]  +
            (int32_t)a[6]  * (int32_t)b[6]  +
            (int32_t)a[7]  * (int32_t)b[7]  +
            (int32_t)a[8]  * (int32_t)b[8]  +
            (int32_t)a[9]  * (int32_t)b[9]  +
            (int32_t)a[10] * (int32_t)b[10] +
            (int32_t)a[11] * (int32_t)b[11];
    }

    if (n == 8) {
        return
            (int32_t)a[0] * (int32_t)b[0] +
            (int32_t)a[1] * (int32_t)b[1] +
            (int32_t)a[2] * (int32_t)b[2] +
            (int32_t)a[3] * (int32_t)b[3] +
            (int32_t)a[4] * (int32_t)b[4] +
            (int32_t)a[5] * (int32_t)b[5] +
            (int32_t)a[6] * (int32_t)b[6] +
            (int32_t)a[7] * (int32_t)b[7];
    }

    if (n == 4) {
        return
            (int32_t)a[0] * (int32_t)b[0] +
            (int32_t)a[1] * (int32_t)b[1] +
            (int32_t)a[2] * (int32_t)b[2] +
            (int32_t)a[3] * (int32_t)b[3];
    }

    int32_t sum = 0;

    for (int i = 0; i < n; ++i)
        sum += (int32_t)a[i] * (int32_t)b[i];

    return sum;
}

/* Dynamic symmetric per-tensor quantisation. 127 levels, never -128, so the
 * codes stay negatable. Rounding is half-away-from-zero to match the exporter;
 * NumPy's default banker's rounding would disagree on exact ties. */
void gt_quant(const float *x, int n, int8_t *q, float *scale)
{
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = fabsf(x[i]);
        if (v > amax) amax = v;
    }
    float s = (amax < 1e-12f) ? 1.0f : (amax / 127.0f);
    float inv = 1.0f / s;
    for (int i = 0; i < n; ++i) {
        float v = x[i] * inv;
        int t = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
        if (t > 127) t = 127; else if (t < -127) t = -127;
        q[i] = (int8_t)t;
    }
    *scale = s;
}

/* [C][F] float -> [pad + F + pad][C] int8, pad rows zeroed. Transposing here
 * is what lets every convolution below run its dot product over contiguous
 * input channels. */
void gt_quant_tpad(const float *x, int C, int F, int pad,
                   int8_t *q, float *scale)
{
    const int n = C * F;
    float amax = 0.0f;

    for (int i = 0; i < n; ++i) {
        float v = fabsf(x[i]);
        if (v > amax) amax = v;
    }

    float s = (amax < 1e-12f) ? 1.0f : (amax / 127.0f);
    float inv = 1.0f / s;

    memset(q, 0, (size_t)(F + 2 * pad) * C);

    for (int c = 0; c < C; ++c) {
        const float *xr = x + (size_t)c * F;
        int8_t *dst = q + (size_t)pad * C + c;

        for (int f = 0; f < F; ++f) {
            float v = xr[f] * inv;
            int t = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));

            if (t > 127) t = 127;
            else if (t < -127) t = -127;

            dst[(size_t)f * C] = (int8_t)t;
        }
    }

    *scale = s;
}

/* -------------------------------------------------------------------------
 * convolutions over the frequency axis
 * ------------------------------------------------------------------------- */

/* Conv2d((1,5), stride (1,2), padding (0,2)).
 * Output bin f reads input bins 2f-2 .. 2f+2, which is padded rows
 * 2f .. 2f+4.
 */
void gt_conv1x5_s2(const gt_qw_t *L, const float *x, int Cin, int Fin,
                   float *y, int Fout, int8_t *q)
{
    const int g = L->groups;
    const int in_pg = L->n_in;
    const int out_pg = L->n_out / g;
    const int rowlen = 5 * in_pg;

    float sx;

    gt_quant_tpad(x, Cin, Fin, 2, q, &sx);

    for (int gi = 0; gi < g; ++gi) {
        for (int o = 0; o < out_pg; ++o) {
            const int oc = gi * out_pg + o;
            const int8_t *w = L->w + (size_t)oc * rowlen;
            const float sc = L->ws[oc] * sx;
            const float bi = L->b[oc];
            float *yr = y + (size_t)oc * Fout;

            for (int f = 0; f < Fout; ++f) {
                const int8_t *p =
                    q + (size_t)(2 * f) * Cin + gi * in_pg;

                int32_t acc = 0;

                for (int k = 0; k < 5; ++k) {
                    acc += gt_dot_i8(
                        w + k * in_pg,
                        p + (size_t)k * Cin,
                        in_pg
                    );
                }

                yr[f] = (float)acc * sc + bi;
            }
        }
    }
}

/* ConvTranspose2d((1,5), stride (1,2), padding (0,2)).
 *
 * Output bin j is fed only by taps k with (j + 2 - k) even -- so even bins
 * see three taps and odd bins see two. That asymmetry IS the upsampling.
 * Treating it as an ordinary strided conv gives a plausible-looking spectrum
 * with a comb pattern on it.
 */
void gt_deconv1x5_s2(const gt_qw_t *L, const float *x, int Cin, int Fin,
                     float *y, int Fout, int8_t *q)
{
    const int g = L->groups;
    const int in_pg = L->n_in;
    const int out_pg = L->n_out / g;
    const int rowlen = 5 * in_pg;

    float sx;

    gt_quant_tpad(x, Cin, Fin, 1, q, &sx);

    for (int gi = 0; gi < g; ++gi) {
        for (int o = 0; o < out_pg; ++o) {
            const int oc = gi * out_pg + o;
            const int8_t *w = L->w + (size_t)oc * rowlen;
            const float sc = L->ws[oc] * sx;
            const float bi = L->b[oc];
            float *yr = y + (size_t)oc * Fout;

            for (int j = 0; j < Fout; ++j) {
                int32_t acc = 0;

                for (int k = (j & 1); k < 5; k += 2) {
                    const int idx = (j + 2 - k) >> 1;

                    if (idx < 0 || idx >= Fin)
                        continue;

                    acc += gt_dot_i8(
                        w + k * in_pg,
                        q + (size_t)(idx + 1) * Cin + gi * in_pg,
                        in_pg
                    );
                }

                yr[j] = (float)acc * sc + bi;
            }
        }
    }
}

/* 1x1 point convolution:
 * [Cin][F] -> [Cout][F], one activation scale per frame.
 */
void gt_conv1x1(const gt_qw_t *L, const float *x, int Cin, int F,
                float *y, int8_t *q)
{
    float sx;

    gt_quant_tpad(x, Cin, F, 0, q, &sx);

    for (int o = 0; o < L->n_out; ++o) {
        const int8_t *w = L->w + (size_t)o * Cin;
        const float sc = L->ws[o] * sx;
        const float bi = L->b[o];
        float *yr = y + (size_t)o * F;

        for (int f = 0; f < F; ++f) {
            yr[f] =
                (float)gt_dot_i8(
                    w,
                    q + (size_t)f * Cin,
                    Cin
                ) * sc + bi;
        }
    }
}

/* y[rows][n_out] = x[rows][n_in] * W^T + b,
 * one scale over the whole input.
 */
void gt_qlinear_m(const gt_qw_t *L, const float *x, int rows, int n_in,
                  float *y, int8_t *q)
{
    float sx;

    gt_quant(x, rows * n_in, q, &sx);

    for (int r = 0; r < rows; ++r) {
        const int8_t *xr = q + (size_t)r * n_in;
        float *yr = y + (size_t)r * L->n_out;

        for (int o = 0; o < L->n_out; ++o) {
            yr[o] =
                (float)gt_dot_i8(
                    L->w + (size_t)o * n_in,
                    xr,
                    n_in
                ) * (L->ws[o] * sx) + L->b[o];
        }
    }
}

void gt_linear_v(const gt_qw_t *L, const float *x,
                 float *y, int8_t *q)
{
    gt_qlinear_m(L, x, 1, L->n_in, y, q);
}

/* Depthwise 3x3 over (time, frequency). t0/t1/t2 are the frames at t-2d,
 * t-d and t; the exporter already flipped the decoder kernels so both encoder
 * and decoder use this same order. Frequency edges are zero-padded. Kept in
 * float: nine MACs per output, so int8 would buy nothing and cost accuracy.
 */
void gt_depthwise33(const float *t0, const float *t1, const float *t2,
                    int C, int F, const float *w,
                    const float *b, float *y)
{
    const float *taps[3];

    taps[0] = t0;
    taps[1] = t1;
    taps[2] = t2;

    for (int c = 0; c < C; ++c) {
        const float *wc = w + (size_t)c * 9;
        float *yr = y + (size_t)c * F;
        const float bi = b[c];

        for (int f = 0; f < F; ++f)
            yr[f] = bi;

        for (int kt = 0; kt < 3; ++kt) {
            const float *src = taps[kt] + (size_t)c * F;

            const float a0 = wc[kt * 3 + 0];    /* f-1 */
            const float a1 = wc[kt * 3 + 1];    /* f   */
            const float a2 = wc[kt * 3 + 2];    /* f+1 */

            yr[0] += a1 * src[0] + a2 * src[1];

            for (int f = 1; f < F - 1; ++f) {
                yr[f] +=
                    a0 * src[f - 1] +
                    a1 * src[f] +
                    a2 * src[f + 1];
            }

            yr[F - 1] +=
                a0 * src[F - 2] +
                a1 * src[F - 1];
        }
    }
}

/* -------------------------------------------------------------------------
 * activations and normalisation
 * ------------------------------------------------------------------------- */

float gt_sigmoid(float x)
{
    if (x >= 30.0f)
        return 1.0f;

    if (x <= -30.0f)
        return 0.0f;

    return 1.0f / (1.0f + expf(-x));
}

float gt_tanh(float x)
{
    return tanhf(x);
}

void gt_prelu(float *x, int n, float a)
{
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0.0f)
            x[i] *= a;
    }
}

/* nn.LayerNorm((width, hidden)): mean and variance over the whole plane,
 * not per row. eps is 1e-8, as in the model. The mean accumulates in double
 * because n is ~1000 and float summation drifts enough to shift the gain.
 */
void gt_layernorm(float *x, int n, const float *w, const float *b)
{
    double m = 0.0;

    for (int i = 0; i < n; ++i)
        m += x[i];

    m /= n;

    double v = 0.0;

    for (int i = 0; i < n; ++i) {
        double d = (double)x[i] - m;
        v += d * d;
    }

    v /= n;

    const float mu = (float)m;
    const float inv = 1.0f / sqrtf((float)v + 1e-8f);

    for (int i = 0; i < n; ++i) {
        x[i] = (x[i] - mu) * inv * w[i] + b[i];
    }
}

/* PyTorch GRU, gate order [r, z, n]:
 *
 *   r = sig(Wir x + bir + Whr h + bhr)
 *   z = sig(Wiz x + biz + Whz h + bhz)
 *   n = tanh(Win x + bin + r * (Whn h + bhn))
 *       <- r multiplies the hidden term only
 *
 * h' = (1-z) n + z h
 *
 * scratch needs 6*H floats.
 */
void gt_gru_step(const gt_gru_t *g, const float *x, float *h,
                 float *scratch, int8_t *q8)
{
    const int H = g->H;

    float *gi = scratch;
    float *gh = scratch + 3 * H;

    gt_linear_v(&g->wih, x, gi, q8);
    gt_linear_v(&g->whh, h, gh, q8);

    for (int i = 0; i < H; ++i) {
        const float r = gt_sigmoid(gi[i] + gh[i]);
        const float z = gt_sigmoid(gi[H + i] + gh[H + i]);
        const float nn =
            gt_tanh(
                gi[2 * H + i] +
                r * gh[2 * H + i]
            );

        h[i] = (1.0f - z) * nn + z * h[i];
    }
}

/* Subband feature extraction:
 * output channel c*3+j holds x[c][f+j-1].
 */
void gt_sfe3(const float *x, int C, int F, float *y)
{
    for (int c = 0; c < C; ++c) {
        const float *src = x + (size_t)c * F;

        float *d0 = y + (size_t)(c * 3 + 0) * F;
        float *d1 = y + (size_t)(c * 3 + 1) * F;
        float *d2 = y + (size_t)(c * 3 + 2) * F;

        d0[0] = 0.0f;

        memcpy(
            d0 + 1,
            src,
            (size_t)(F - 1) * sizeof(float)
        );

        memcpy(
            d1,
            src,
            (size_t)F * sizeof(float)
        );

        memcpy(
            d2,
            src + 1,
            (size_t)(F - 1) * sizeof(float)
        );

        d2[F - 1] = 0.0f;
    }
}

/* -------------------------------------------------------------------------
 * radix-2 complex FFT
 *
 * The real transform is done by running a full complex FFT with a zeroed
 * imaginary part. That is 2x the arithmetic of a packed real FFT and still
 * under 2 % of the frame budget, and it removes an entire class of packing
 * bugs. If you ever need the cycles back, esp-dsp's dsps_fft2r_fc32 plus
 * dsps_cplx2real_fc32 drops straight in here.
 * ------------------------------------------------------------------------- */

void gt_fft_init(int n, float *tw_re, float *tw_im, uint16_t *brev)
{
    for (int i = 0; i < n / 2; ++i) {
        double a =
            -2.0 * M_PI * (double)i / (double)n;

        tw_re[i] = (float)cos(a);
        tw_im[i] = (float)sin(a);
    }

    int bits = 0;

    while ((1 << bits) < n)
        ++bits;

    for (int i = 0; i < n; ++i) {
        unsigned r = 0;
        unsigned v = (unsigned)i;

        for (int b = 0; b < bits; ++b) {
            r = (r << 1) | (v & 1u);
            v >>= 1;
        }

        brev[i] = (uint16_t)r;
    }
}

void gt_fft(float *re, float *im, int n,
            const float *tw_re, const float *tw_im,
            const uint16_t *brev, int inverse)
{
    /* Bit reversal */
    for (int i = 0; i < n; ++i) {
        int j = brev[i];

        if (j > i) {
            float t = re[i];
            re[i] = re[j];
            re[j] = t;

            t = im[i];
            im[i] = im[j];
            im[j] = t;
        }
    }

    /* Radix-2 FFT */
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const int step = n / len;

        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; ++k) {
                float wr = tw_re[k * step];
                float wi = tw_im[k * step];

                if (inverse)
                    wi = -wi;

                const int a = i + k;
                const int b = a + half;

                const float xr =
                    re[b] * wr -
                    im[b] * wi;

                const float xi =
                    re[b] * wi +
                    im[b] * wr;

                re[b] = re[a] - xr;
                im[b] = im[a] - xi;

                re[a] += xr;
                im[a] += xi;
            }
        }
    }

    /* 1/N scaling for inverse transform */
    if (inverse) {
        const float s = 1.0f / (float)n;

        for (int i = 0; i < n; ++i) {
            re[i] *= s;
            im[i] *= s;
        }
    }
}

