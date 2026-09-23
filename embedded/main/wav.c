#include <stdlib.h>
#include <string.h>

#include "wav.h"

#define CHUNK 512

static void put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static uint32_t get_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int wav_write_open(wav_writer_t *w, const char *path, int sample_rate)
{
    memset(w, 0, sizeof(*w));
    w->f = fopen(path, "wb");
    if (!w->f) return -1;
    w->sample_rate = sample_rate;

    uint8_t h[44];
    memcpy(h + 0, "RIFF", 4);
    put_u32(h + 4, 0);                       /* patched on close */
    memcpy(h + 8, "WAVEfmt ", 8);
    put_u32(h + 16, 16);                     /* PCM fmt chunk size */
    h[20] = 1; h[21] = 0;                    /* PCM */
    h[22] = 1; h[23] = 0;                    /* mono */
    put_u32(h + 24, (uint32_t)sample_rate);
    put_u32(h + 28, (uint32_t)sample_rate * 2);
    h[32] = 2; h[33] = 0;                    /* block align */
    h[34] = 16; h[35] = 0;                   /* bits */
    memcpy(h + 36, "data", 4);
    put_u32(h + 40, 0);                      /* patched on close */
    return fwrite(h, 1, 44, w->f) == 44 ? 0 : -1;
}

int wav_write(wav_writer_t *w, const float *s, int n)
{
    int16_t buf[CHUNK];
    int done = 0;
    while (done < n) {
        int m = n - done;
        if (m > CHUNK) m = CHUNK;
        for (int i = 0; i < m; ++i) {
            float v = s[done + i] * 32767.0f;
            if (v > 32767.0f) v = 32767.0f;
            else if (v < -32768.0f) v = -32768.0f;
            buf[i] = (int16_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        }
        if ((int)fwrite(buf, sizeof(int16_t), (size_t)m, w->f) != m) return -1;
        done += m;
    }
    w->n_samples += (uint32_t)n;
    return 0;
}

int wav_write_close(wav_writer_t *w)
{
    if (!w->f) return -1;
    const uint32_t data_bytes = w->n_samples * 2u;
    uint8_t v[4];
    fseek(w->f, 4, SEEK_SET);  put_u32(v, 36u + data_bytes); fwrite(v, 1, 4, w->f);
    fseek(w->f, 40, SEEK_SET); put_u32(v, data_bytes);       fwrite(v, 1, 4, w->f);
    int rc = fclose(w->f);
    w->f = NULL;
    return rc;
}

/* Walks the chunk list rather than assuming a 44-byte header -- plenty of
 * recorders emit a LIST or fact chunk before the data, and a fixed offset would
 * read those bytes as audio. */
int wav_read_open(wav_reader_t *r, const char *path)
{
    memset(r, 0, sizeof(*r));
    r->f = fopen(path, "rb");
    if (!r->f) return -1;

    uint8_t hdr[12];
    if (fread(hdr, 1, 12, r->f) != 12 ||
        memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) goto bad;

    int bits = 0;
    for (;;) {
        uint8_t ch[8];
        if (fread(ch, 1, 8, r->f) != 8) goto bad;
        const uint32_t sz = get_u32(ch + 4);
        if (!memcmp(ch, "fmt ", 4)) {
            uint8_t fmt[16];
            if (sz < 16 || fread(fmt, 1, 16, r->f) != 16) goto bad;
            r->channels = fmt[2] | (fmt[3] << 8);
            r->sample_rate = (int)get_u32(fmt + 4);
            bits = fmt[14] | (fmt[15] << 8);
            if (sz > 16) fseek(r->f, (long)(sz - 16), SEEK_CUR);
        } else if (!memcmp(ch, "data", 4)) {
            if (bits != 16 || r->channels < 1) goto bad;
            r->n_samples = sz / 2u / (uint32_t)r->channels;
            return 0;
        } else {
            fseek(r->f, (long)(sz + (sz & 1u)), SEEK_CUR);
        }
    }
bad:
    fclose(r->f);
    r->f = NULL;
    return -1;
}

int wav_read(wav_reader_t *r, float *out, int n)
{
    int16_t buf[CHUNK];
    int done = 0;
    while (done < n && r->n_samples > 0) {
        int m = n - done;
        if (m > CHUNK / (r->channels > 0 ? r->channels : 1))
            m = CHUNK / (r->channels > 0 ? r->channels : 1);
        if ((uint32_t)m > r->n_samples) m = (int)r->n_samples;
        const int vals = m * r->channels;
        const int got = (int)fread(buf, sizeof(int16_t), (size_t)vals, r->f);
        if (got < r->channels) break;
        const int frames = got / r->channels;
        for (int i = 0; i < frames; ++i) {
            int32_t acc = 0;                       /* downmix to mono */
            for (int c = 0; c < r->channels; ++c) acc += buf[i * r->channels + c];
            out[done + i] = (float)acc / (float)r->channels / 32768.0f;
        }
        done += frames;
        r->n_samples -= (uint32_t)frames;
    }
    return done;
}

void wav_read_close(wav_reader_t *r)
{
    if (r->f) fclose(r->f);
    r->f = NULL;
}
