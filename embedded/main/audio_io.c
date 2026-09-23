/*
 * audio_io.c -- I2S capture/playback and the SD card mount.
 *
 * Nothing here is model-specific; it is the standard ESP-IDF v5 plumbing so
 * that gtcrn.c never has to know where samples come from. Swap the mic for a
 * codec, or the SD card for SPIFFS, and app_main is the only other file that
 * needs to change.
 */
#include <string.h>

#include "driver/i2s_std.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"

#include "audio_io.h"
#include "board_config.h"

static const char *TAG = "audio_io";

static i2s_chan_handle_t s_rx, s_tx;
static sdmmc_card_t *s_card;

#define IO_CHUNK 256

/* ------------------------------------------------------------------ SD --- */
int sd_mount(void)
{
    esp_vfs_fat_sdmmc_mount_config_t mcfg = {
        .format_if_mount_failed = false,
        .max_files = 4,
        .allocation_unit_size = 16 * 1024,
    };

    spi_bus_config_t bus = {
        .mosi_io_num = SD_MOSI_GPIO,
        .miso_io_num = SD_MISO_GPIO,
        .sclk_io_num = SD_SCLK_GPIO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };
    esp_err_t err = spi_bus_initialize(SD_SPI_HOST, &bus, SDSPI_DEFAULT_DMA);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "spi_bus_initialize: %s", esp_err_to_name(err));
        return -1;
    }

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = SD_SPI_HOST;
    host.max_freq_khz = SD_FREQ_KHZ;

    sdspi_device_config_t slot = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot.gpio_cs = SD_CS_GPIO;
    slot.host_id = SD_SPI_HOST;

    err = esp_vfs_fat_sdspi_mount("/sdcard", &host, &slot, &mcfg, &s_card);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "mount failed: %s -- check wiring, and that the card is "
                      "FAT32 (not exFAT)", esp_err_to_name(err));
        return -1;
    }
    ESP_LOGI(TAG, "SD mounted: %s, %lluMB", s_card->cid.name,
             ((uint64_t)s_card->csd.capacity) * s_card->csd.sector_size / (1024 * 1024));
    return 0;
}

void sd_unmount(void)
{
    if (s_card) {
        esp_vfs_fat_sdcard_unmount("/sdcard", s_card);
        s_card = NULL;
    }
}

/* ----------------------------------------------------------------- mic --- */
int mic_init(int sample_rate)
{
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(MIC_I2S_PORT, I2S_ROLE_MASTER);
    cc.dma_desc_num = 6;
    cc.dma_frame_num = 256;
    cc.auto_clear = true;
    if (i2s_new_channel(&cc, NULL, &s_rx) != ESP_OK) return -1;

    i2s_std_config_t cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG((uint32_t)sample_rate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = MIC_BCLK_GPIO,
            .ws   = MIC_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = MIC_DIN_GPIO,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };
    cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    if (i2s_channel_init_std_mode(s_rx, &cfg) != ESP_OK) return -1;
    if (i2s_channel_enable(s_rx) != ESP_OK) return -1;
    ESP_LOGI(TAG, "mic ready: %d Hz, 32-bit slots, left channel", sample_rate);
    return 0;
}

int mic_read(float *out, int n)
{
    int32_t raw[IO_CHUNK];
    int done = 0;
    while (done < n) {
        int m = n - done;
        if (m > IO_CHUNK) m = IO_CHUNK;
        size_t got = 0;
        if (i2s_channel_read(s_rx, raw, (size_t)m * sizeof(int32_t), &got,
                             portMAX_DELAY) != ESP_OK)
            break;
        const int k = (int)(got / sizeof(int32_t));
        /* MEMS parts left-justify 24 significant bits in a 32-bit slot.
         * Shifting by MIC_SHIFT keeps the headroom a >>16 would discard. */
        for (int i = 0; i < k; ++i)
            out[done + i] = (float)(raw[i] >> MIC_SHIFT) / 2097152.0f;
        done += k;
        if (k == 0) break;
    }
    return done;
}

/* ----------------------------------------------------------------- spk --- */
int spk_init(int sample_rate)
{
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(SPK_I2S_PORT, I2S_ROLE_MASTER);
    cc.dma_desc_num = 6;
    cc.dma_frame_num = 256;
    cc.auto_clear = true;
    if (i2s_new_channel(&cc, &s_tx, NULL) != ESP_OK) return -1;

    i2s_std_config_t cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG((uint32_t)sample_rate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                        I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = SPK_BCLK_GPIO,
            .ws   = SPK_WS_GPIO,
            .dout = SPK_DOUT_GPIO,
            .din  = I2S_GPIO_UNUSED,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };
    if (i2s_channel_init_std_mode(s_tx, &cfg) != ESP_OK) return -1;
    if (i2s_channel_enable(s_tx) != ESP_OK) return -1;
    return 0;
}

int spk_write(const float *in, int n)
{
    int16_t buf[IO_CHUNK];
    int done = 0;
    while (done < n) {
        int m = n - done;
        if (m > IO_CHUNK) m = IO_CHUNK;
        for (int i = 0; i < m; ++i) {
            float v = in[done + i] * 32767.0f;
            if (v > 32767.0f) v = 32767.0f;
            else if (v < -32768.0f) v = -32768.0f;
            buf[i] = (int16_t)v;
        }
        size_t wrote = 0;
        if (i2s_channel_write(s_tx, buf, (size_t)m * sizeof(int16_t), &wrote,
                              portMAX_DELAY) != ESP_OK)
            break;
        done += (int)(wrote / sizeof(int16_t));
    }
    return done;
}
