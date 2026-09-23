#pragma once

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define USB_AUDIO_SAMPLE_RATE   48000
#define USB_AUDIO_CHANNELS      1
#define USB_AUDIO_BITS          16

esp_err_t usb_audio_init(void);

/*
 * Read PCM16 mono samples from the USB microphone.
 *
 * Returns:
 *   ESP_OK          - samples were read
 *   ESP_ERR_TIMEOUT - no samples arrived before timeout
 *   other           - USB/UAC error
 */
esp_err_t usb_audio_read(int16_t *samples,
                         uint32_t sample_count,
                         uint32_t timeout_ms);

bool usb_audio_is_ready(void);

#ifdef __cplusplus
}
#endif
