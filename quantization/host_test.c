/*
 * host_test.c -- run the ESP32 runtime on your laptop.
 *
 * Same sources, same arithmetic, no board. Feed it a blob and a raw float32
 * input file, get a raw float32 output file, and diff that against
 * gtcrn_ref.py. If they agree to ~1e-5 the port is correct and anything you
 * hear on the device afterwards is an I/O problem, not a model problem.
 *
 *   gcc -O2 -std=c99 -I ../firmware/components/gtcrn/include \
 *       -o host_test host_test.c \
 *       ../firmware/components/gtcrn/gtcrn_ops.c \
 *       ../firmware/components/gtcrn/gtcrn_net.c -lm
 *   ./host_test blob.bin input.f32 output.f32
 *
 * It also prints per-frame wall-clock time. Divide by the hop duration
 * (16 ms at 16 kHz) for a rough real-time factor on your desktop; the S3 will
 * land somewhere around 20-40x slower than a modern x86 core.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gtcrn.h"

static void *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    *len = (size_t)n;
    return p;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s blob.bin input.f32 output.f32\n", argv[0]);
        return 2;
    }
    size_t blob_len = 0, in_len = 0;
    void *blob = slurp(argv[1], &blob_len);
    float *in = (float *)slurp(argv[2], &in_len);
    const size_t n_samples = in_len / sizeof(float);

    gtcrn_t *g = gtcrn_create(blob, blob_len);
    if (!g) { fprintf(stderr, "gtcrn_create failed -- bad or truncated blob\n"); return 1; }

    const gtcrn_cfg_t *c = gtcrn_config(g);
    printf("model  : %d Hz, n_fft %d, hop %d, C=%d, dpgrnn=%d, DF=%dx%d\n",
           c->sample_rate, c->n_fft, c->hop_length, c->base_channels,
           c->n_dpgrnn, c->df_order, c->df_bins);
    printf("memory : %.1f kB fast + %.1f kB bulk\n",
           gtcrn_mem_internal(g) / 1024.0, gtcrn_mem_psram(g) / 1024.0);

    const int hop = c->hop_length;
    const size_t n_hops = n_samples / (size_t)hop;
    float *out = (float *)calloc(n_hops * (size_t)hop, sizeof(float));

    clock_t t0 = clock();
    for (size_t h = 0; h < n_hops; ++h)
        gtcrn_process_hop(g, in + h * (size_t)hop, out + h * (size_t)hop);
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;

    FILE *f = fopen(argv[3], "wb");
    fwrite(out, sizeof(float), n_hops * (size_t)hop, f);
    fclose(f);

    const double audio = (double)(n_hops * (size_t)hop) / c->sample_rate;
    printf("ran    : %zu frames, %.3f s audio in %.3f s  (RTF %.3f, %.2f ms/frame)\n",
           n_hops, audio, secs, secs / (audio > 0 ? audio : 1),
           n_hops ? secs * 1000.0 / (double)n_hops : 0.0);
    printf("wrote  : %s\n", argv[3]);

    gtcrn_destroy(g);
    free(blob); free(in); free(out);
    return 0;
}
