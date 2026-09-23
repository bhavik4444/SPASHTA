#include "usb_audio.h"

#include <string.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_check.h"

#include "usb/usb_host.h"
#include "usb/uac_host.h"

static const char *TAG = "usb_audio";

static uac_host_device_handle_t s_uac_handle = NULL;
static volatile bool s_ready = false;

static void usb_host_daemon_task(void *arg)
{
    (void)arg;

    while (1) {
        uint32_t event_flags = 0;

        esp_err_t err =
            usb_host_lib_handle_events(portMAX_DELAY, &event_flags);

        if (err != ESP_OK && err != ESP_ERR_TIMEOUT) {
            ESP_LOGW(TAG, "usb_host_lib_handle_events: %s",
                     esp_err_to_name(err));
        }
    }
}

static void uac_device_event_cb(
    uac_host_device_handle_t uac_device_handle,
    const uac_host_device_event_t event,
    void *arg)
{
    (void)arg;

    switch (event) {

    case UAC_HOST_DEVICE_EVENT_RX_DONE:
        /*
         * We intentionally do not read here.
         * The application task performs blocking reads.
         */
        break;

    case UAC_HOST_DEVICE_EVENT_TRANSFER_ERROR:
        ESP_LOGE(TAG, "UAC transfer error");
        break;

    case UAC_HOST_DRIVER_EVENT_DISCONNECTED:
        ESP_LOGW(TAG, "UAC microphone disconnected");

        if (uac_device_handle == s_uac_handle) {
            s_ready = false;
            s_uac_handle = NULL;
        }
        break;

    default:
        break;
    }
}

static void uac_driver_event_cb(
    uint8_t addr,
    uint8_t iface_num,
    const uac_host_driver_event_t event,
    void *arg)
{
    (void)arg;

    if (event != UAC_HOST_DRIVER_EVENT_RX_CONNECTED) {
        return;
    }

    /*
     * This is the microphone stream.
     *
     * Our Linux descriptor showed:
     *   interface 2
     *   mono
     *   S16
     *   48 kHz
     *
     * The event itself gives us the actual interface number and USB address.
     */
    ESP_LOGI(TAG,
             "UAC RX device connected: addr=%u iface=%u",
             addr,
             iface_num);

    uac_host_device_config_t dev_cfg = {
        .addr = addr,
        .iface_num = iface_num,

        /*
         * 8 KB internal UAC ring buffer.
         * Large enough for this first test.
         */
        .buffer_size = 8192,

        /*
         * Start delivering data once some audio is buffered.
         */
        .buffer_threshold = 1024,

        .callback = uac_device_event_cb,
        .callback_arg = NULL,
    };

    esp_err_t err =
        uac_host_device_open(&dev_cfg, &s_uac_handle);

    if (err != ESP_OK) {
        ESP_LOGE(TAG,
                 "uac_host_device_open failed: %s",
                 esp_err_to_name(err));
        s_uac_handle = NULL;
        return;
    }

    ESP_LOGI(TAG, "UAC microphone opened");

    /*
     * Our exact device advertises:
     *   PCM
     *   1 channel
     *   16 bit
     *   48000 Hz
     */
    uac_host_stream_config_t stream_cfg = {
        .channels = USB_AUDIO_CHANNELS,
        .bit_resolution = USB_AUDIO_BITS,
        .sample_freq = USB_AUDIO_SAMPLE_RATE,
        .flags = 0,
    };

    err = uac_host_device_start(s_uac_handle, &stream_cfg);

    if (err != ESP_OK) {
        ESP_LOGE(TAG,
                 "uac_host_device_start failed: %s",
                 esp_err_to_name(err));

        uac_host_device_close(s_uac_handle);
        s_uac_handle = NULL;
        return;
    }

    s_ready = true;

    ESP_LOGI(TAG,
             "USB microphone streaming: %d Hz, %d-bit, %d channel",
             USB_AUDIO_SAMPLE_RATE,
             USB_AUDIO_BITS,
             USB_AUDIO_CHANNELS);
}

esp_err_t usb_audio_init(void)
{
    /*
     * Install the low-level USB Host Library.
     */
    const usb_host_config_t host_cfg = {
        .skip_phy_setup = false,
        .intr_flags = ESP_INTR_FLAG_LOWMED,
    };

    ESP_RETURN_ON_ERROR(
        usb_host_install(&host_cfg),
        TAG,
        "usb_host_install failed"
    );

    /*
     * The Host Library itself does not create its daemon task.
     * We provide it here.
     */
    BaseType_t task_ok = xTaskCreate(
        usb_host_daemon_task,
        "usb_host_daemon",
        4096,
        NULL,
        20,
        NULL
    );

    if (task_ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create USB daemon task");
        return ESP_ERR_NO_MEM;
    }

    /*
     * Install UAC class driver.
     *
     * The UAC driver creates its own background task,
     * which dispatches UAC-level events.
     */
    const uac_host_driver_config_t uac_cfg = {
        .create_background_task = true,
        .task_priority = 20,
        .stack_size = 4096,
        .core_id = tskNO_AFFINITY,
        .callback = uac_driver_event_cb,
        .callback_arg = NULL,
    };

    ESP_RETURN_ON_ERROR(
        uac_host_install(&uac_cfg),
        TAG,
        "uac_host_install failed"
    );

    ESP_LOGI(TAG,
             "USB Host + UAC installed. Waiting for microphone...");

    return ESP_OK;
}

bool usb_audio_is_ready(void)
{
    return s_ready && s_uac_handle != NULL;
}

esp_err_t usb_audio_read(int16_t *samples,
                         uint32_t sample_count,
                         uint32_t timeout_ms)
{
    if (!samples || sample_count == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    if (!usb_audio_is_ready()) {
        return ESP_ERR_INVALID_STATE;
    }

    uint32_t wanted_bytes = sample_count * sizeof(int16_t);
    uint32_t total_read = 0;

    while (total_read < wanted_bytes) {

        uint32_t bytes_read = 0;

        esp_err_t err = uac_host_device_read(
            s_uac_handle,
            (uint8_t *)samples + total_read,
            wanted_bytes - total_read,
            &bytes_read,
            pdMS_TO_TICKS(timeout_ms)
        );

        if (err != ESP_OK) {
            return err;
        }

        if (bytes_read == 0) {
            return ESP_ERR_TIMEOUT;
        }

        total_read += bytes_read;
    }

    return ESP_OK;
}

