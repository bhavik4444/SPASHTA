#ifndef AUDIO_IO_H
#define AUDIO_IO_H

int  sd_mount(void);
void sd_unmount(void);

int  mic_init(int sample_rate);
int  mic_read(float *out, int n);          /* blocking, returns samples read */

int  spk_init(int sample_rate);
int  spk_write(const float *in, int n);

#endif /* AUDIO_IO_H */
