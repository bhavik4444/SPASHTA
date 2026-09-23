/*
 * wav.h -- minimal 16-bit mono PCM WAV reader and writer.
 *
 * Streaming on both sides: the writer patches the two length fields on close,
 * so a recording never has to exist in RAM in full, and a power cut costs you
 * the header rather than the session.
 */
#ifndef WAV_H
#define WAV_H

#include <stdint.h>
#include <stdio.h>

typedef struct {
    FILE    *f;
    uint32_t n_samples;
    int      sample_rate;
} wav_writer_t;

typedef struct {
    FILE    *f;
    uint32_t n_samples;     /* remaining */
    int      sample_rate;
    int      channels;
} wav_reader_t;

int  wav_write_open(wav_writer_t *w, const char *path, int sample_rate);
int  wav_write(wav_writer_t *w, const float *samples, int n);   /* -1..1 */
int  wav_write_close(wav_writer_t *w);

int  wav_read_open(wav_reader_t *r, const char *path);
int  wav_read(wav_reader_t *r, float *samples, int n);          /* returns n read */
void wav_read_close(wav_reader_t *r);

#endif /* WAV_H */
