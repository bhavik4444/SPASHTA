#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/uart.h"

#include "esp_log.h"
#include "esp_err.h"

#include "gtcrn.h"


#define TAG "GTCRN_LIVE"


/* ============================================================
 * UART
 * ============================================================ */

#define AUDIO_UART      UART_NUM_0
#define UART_BAUD       921600

/*
 * ESP32-S3 typical UART0 pins used by USB-UART COM boards.
 */
#define UART_TX_PIN     43
#define UART_RX_PIN     44


/* ============================================================
 * GTCRN
 * ============================================================ */

#define SAMPLE_RATE     16000
#define HOP_SIZE        256


/* ============================================================
 * PACKET
 * ============================================================ */

#define PACKET_MAGIC    0x47544352u   /* "GTCR" */

typedef struct __attribute__((packed))
{
    uint32_t magic;
    uint16_t samples;
} audio_header_t;


/* ============================================================
 * GLOBAL
 * ============================================================ */

static gtcrn_t *g_gtcrn = NULL;


/* ============================================================
 * UART INIT
 * ============================================================ */

static void uart_audio_init(void)
{
    const uart_config_t cfg =
    {
        .baud_rate = UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
#if SOC_UART_SUPPORT_XTAL_CLK
        .source_clk = UART_SCLK_XTAL,
#endif
    };

    ESP_ERROR_CHECK(
        uart_param_config(
            AUDIO_UART,
            &cfg
        )
    );

    ESP_ERROR_CHECK(
        uart_set_pin(
            AUDIO_UART,
            UART_TX_PIN,
            UART_RX_PIN,
            UART_PIN_NO_CHANGE,
            UART_PIN_NO_CHANGE
        )
    );

    ESP_ERROR_CHECK(
        uart_driver_install(
            AUDIO_UART,
            8192,
            8192,
            0,
            NULL,
            0
        )
    );

    uart_flush_input(AUDIO_UART);
}


/* ============================================================
 * READ EXACT NUMBER OF BYTES
 * ============================================================ */

static bool uart_read_exact(
    uint8_t *buffer,
    size_t length
)
{
    size_t got = 0;

    while (got < length)
    {
        int n = uart_read_bytes(
            AUDIO_UART,
            buffer + got,
            length - got,
            portMAX_DELAY
        );

        if (n <= 0)
        {
            return false;
        }

        got += (size_t)n;
    }

    return true;
}


/* ============================================================
 * WRITE EXACT NUMBER OF BYTES
 * ============================================================ */

static bool uart_write_exact(
    const uint8_t *buffer,
    size_t length
)
{
    size_t sent = 0;

    while (sent < length)
    {
        int n = uart_write_bytes(
            AUDIO_UART,
            (const char *)buffer + sent,
            length - sent
        );

        if (n <= 0)
        {
            return false;
        }

        sent += (size_t)n;
    }

    uart_wait_tx_done(
        AUDIO_UART,
        pdMS_TO_TICKS(1000)
    );

    return true;
}

static float calculate_rms(
    const float *x,
    int n
)
{
    double sum = 0.0;

    for (int i = 0; i < n; ++i) {
        double v = x[i];
        sum += v * v;
    }

    double rms =
        sqrt(sum / (double)n);

    if (rms < 1.0e-9) {
        return 1.0e-9f;
    }

    return (float)rms;
}
/* ============================================================
 * AUDIO TASK
 * ============================================================ */

static void audio_task(void *arg)
{
    (void)arg;

    int16_t input_pcm[HOP_SIZE];
    int16_t output_pcm[HOP_SIZE];

    float input_f[HOP_SIZE];
    float output_f[HOP_SIZE];

    audio_header_t rx_header;
    audio_header_t tx_header;

    uint32_t frame_count = 0;


    tx_header.magic = PACKET_MAGIC;
    tx_header.samples = HOP_SIZE;


    ESP_LOGI(TAG, "====================================");
    ESP_LOGI(TAG, "AUDIO TASK READY");
    ESP_LOGI(TAG, "Waiting for GTCRN packets...");
    ESP_LOGI(TAG, "Packet magic : 0x%08X", PACKET_MAGIC);
    ESP_LOGI(TAG, "Hop          : %d", HOP_SIZE);
    ESP_LOGI(TAG, "Baud         : %d", UART_BAUD);
    ESP_LOGI(TAG, "====================================");


    /*
     * IMPORTANT:
     *
     * From here onward UART0 is raw binary audio.
     * No logging after this point.
     */
    vTaskDelay(pdMS_TO_TICKS(500));

    esp_log_level_set("*", ESP_LOG_NONE);


    while (1)
    {
        /* ----------------------------------------------------
         * HEADER
         * ---------------------------------------------------- */

        if (!uart_read_exact(
                (uint8_t *)&rx_header,
                sizeof(rx_header)))
        {
            continue;
        }


        /*
         * Check magic.
         *
         * If garbage is received before the first packet,
         * resynchronise instead of processing it.
         */
        if (rx_header.magic != PACKET_MAGIC)
        {
            continue;
        }


        if (rx_header.samples != HOP_SIZE)
        {
            continue;
        }


        /* ----------------------------------------------------
         * PCM
         * ---------------------------------------------------- */

        if (!uart_read_exact(
                (uint8_t *)input_pcm,
                sizeof(input_pcm)))
        {
            continue;
        }


        /* ----------------------------------------------------
         * INT16 -> FLOAT
         * ---------------------------------------------------- */

        for (int i = 0; i < HOP_SIZE; ++i)
        {
            input_f[i] =
                (float)input_pcm[i] / 32768.0f;
        }


        /* ----------------------------------------------------
         * GTCRN
         * ---------------------------------------------------- */

        gtcrn_process_hop(
            g_gtcrn,
            input_f,
            output_f
        );


        /* ----------------------------------------------------
         * FLOAT -> INT16
         * ---------------------------------------------------- */

        for (int i = 0; i < HOP_SIZE; ++i)
        {
            float value = output_f[i];

            if (value > 1.0f)
                value = 1.0f;

            if (value < -1.0f)
                value = -1.0f;

            float scaled =
                value * 32767.0f;

            if (scaled > 32767.0f)
                scaled = 32767.0f;

            if (scaled < -32768.0f)
                scaled = -32768.0f;

            output_pcm[i] =
                (int16_t)lrintf(scaled);
        }


        /* ----------------------------------------------------
         * SEND HEADER
         * ---------------------------------------------------- */

        if (!uart_write_exact(
                (const uint8_t *)&tx_header,
                sizeof(tx_header)))
        {
            continue;
        }


        /* ----------------------------------------------------
         * SEND PCM
         * ---------------------------------------------------- */

        if (!uart_write_exact(
                (const uint8_t *)output_pcm,
                sizeof(output_pcm)))
        {
            continue;
        }


        frame_count++;
    }
}


/* ============================================================
 * APP MAIN
 * ============================================================ */

void app_main(void)
{
    ESP_LOGI(TAG, "====================================");
    ESP_LOGI(TAG, "GTCRN LIVE");
    ESP_LOGI(TAG, "====================================");


    /* --------------------------------------------------------
     * UART
     * -------------------------------------------------------- */

    uart_audio_init();


    /* --------------------------------------------------------
     * EMBEDDED MODEL
     * -------------------------------------------------------- */

    extern const uint8_t gtcrn_int8_bin_start[]
        asm("_binary_gtcrn_int8_bin_start");

    extern const uint8_t gtcrn_int8_bin_end[]
        asm("_binary_gtcrn_int8_bin_end");


    const void *blob =
        gtcrn_int8_bin_start;

    size_t blob_len =
        (size_t)(
            gtcrn_int8_bin_end -
            gtcrn_int8_bin_start
        );


    ESP_LOGI(
        TAG,
        "Model blob: %u bytes",
        (unsigned)blob_len
    );


    /* --------------------------------------------------------
     * CREATE MODEL
     * -------------------------------------------------------- */

    g_gtcrn =
        gtcrn_create(
            blob,
            blob_len
        );


    if (g_gtcrn == NULL)
    {
        ESP_LOGE(
            TAG,
            "GTCRN CREATE FAILED"
        );

        return;
    }


    /* --------------------------------------------------------
     * MODEL CONFIG
     * -------------------------------------------------------- */

    const gtcrn_cfg_t *cfg =
        gtcrn_config(g_gtcrn);


    if (cfg == NULL)
    {
        ESP_LOGE(
            TAG,
            "GTCRN CONFIG FAILED"
        );

        gtcrn_destroy(g_gtcrn);
        g_gtcrn = NULL;

        return;
    }


    ESP_LOGI(
        TAG,
        "Sample rate : %d Hz",
        cfg->sample_rate
    );

    ESP_LOGI(
        TAG,
        "FFT         : %d",
        cfg->n_fft
    );

    ESP_LOGI(
        TAG,
        "Hop         : %d samples",
        cfg->hop_length
    );

    ESP_LOGI(
        TAG,
        "Latency     : %d samples",
        gtcrn_latency_samples(g_gtcrn)
    );

    ESP_LOGI(
        TAG,
        "Internal    : %u bytes",
        (unsigned)gtcrn_mem_internal(g_gtcrn)
    );

    ESP_LOGI(
        TAG,
        "PSRAM       : %u bytes",
        (unsigned)gtcrn_mem_psram(g_gtcrn)
    );


    /* --------------------------------------------------------
     * VALIDATION
     * -------------------------------------------------------- */

    if (cfg->sample_rate != SAMPLE_RATE)
    {
        ESP_LOGE(
            TAG,
            "Expected 16 kHz, got %d",
            cfg->sample_rate
        );

        gtcrn_destroy(g_gtcrn);
        g_gtcrn = NULL;

        return;
    }


    if (cfg->hop_length != HOP_SIZE)
    {
        ESP_LOGE(
            TAG,
            "Expected hop 256, got %d",
            cfg->hop_length
        );

        gtcrn_destroy(g_gtcrn);
        g_gtcrn = NULL;

        return;
    }


    /* --------------------------------------------------------
     * MODEL SETTINGS
     * -------------------------------------------------------- */

    gtcrn_set_mask_min(
        g_gtcrn,
        0.02f
    );

    gtcrn_reset(g_gtcrn);


    /* --------------------------------------------------------
     * AUDIO TASK
     * -------------------------------------------------------- */

    BaseType_t ok =
        xTaskCreatePinnedToCore(
            audio_task,
            "gtcrn_audio",
            16384,
            NULL,
            5,
            NULL,
            1
        );


    if (ok != pdPASS)
    {
        ESP_LOGE(
            TAG,
            "AUDIO TASK CREATE FAILED"
        );

        gtcrn_destroy(g_gtcrn);
        g_gtcrn = NULL;

        return;
    }
}
