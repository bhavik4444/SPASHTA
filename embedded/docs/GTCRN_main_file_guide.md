# GTCRN ESP32 — `firmware/main` File Guide

This document explains what each file in `firmware/main/` is for.

## Runtime source files

| File | Purpose |
|---|---|
| `app_main.c` | Main firmware entry point. Creates/initializes the GTCRN runtime and controls the current audio-processing flow. In the current serial-bridge version, it configures UART at 921600 baud, receives 256-sample 16 kHz PCM hops, runs `gtcrn_process_hop()`, and sends the processed hop back. |
| `audio_io.c` | Hardware audio and SD-card I/O implementation. Provides I2S microphone input, I2S speaker output, and SD-card mounting/unmounting. |
| `audio_io.h` | Header for `audio_io.c`. Declares SD, microphone, and speaker functions such as `sd_mount()`, `mic_init()`, `mic_read()`, `spk_init()`, and `spk_write()`. |
| `usb_audio.c` | USB Audio Class host implementation. Handles USB Host/UAC setup and reads PCM data from a connected USB microphone. |
| `usb_audio.h` | Public interface for the USB microphone module, including the 48 kHz, mono, 16-bit USB audio format used by the implementation. |
| `wav.c` | WAV file reader/writer implementation for 16-bit PCM audio. Handles opening, reading/writing samples, and closing WAV files. |
| `wav.h` | Header for the WAV reader/writer API and its reader/writer structures. |
| `wifi_audio.c` | Reserved/experimental Wi-Fi audio module. The current file is effectively a placeholder and contains no substantial implementation. |
| `wifi_audio.h` | Header for the Wi-Fi audio module. Currently only provides the module interface/placeholder. |

## Model files

| File | Purpose |
|---|---|
| `gtcrn_int8.bin` | **The actual exported GTCRN neural-network model weights.** This is embedded into the ESP32 firmware and loaded by the GTCRN runtime. |
| `gtcrn_int8.bin.json` | Sidecar metadata/configuration for the exported model. Records model parameters and export information for reference. |
| `gtcrn_testvec.bin` | Optional model verification data containing a short input signal and expected output. When present, the build enables the boot self-test so the ESP32 can compare its output against the expected PC result. |

## Build/configuration files

| File | Purpose |
|---|---|
| `CMakeLists.txt` | ESP-IDF component build definition for `main`. Lists source files, dependencies, and embedded files. It always embeds `gtcrn_int8.bin`; it also embeds `gtcrn_testvec.bin` when that file exists. |
| `idf_component.yml` | ESP-IDF Component Manager manifest. Declares the component's IDF dependency and the `espressif/usb_host_uac` dependency used by the USB microphone code. |
| `board_config.h` | Central board/hardware configuration. Defines firmware modes, recording settings, I2S microphone pins, I2S speaker pins, SD-card SPI pins, and output gain. This is the main file to edit when hardware wiring changes. |

## Legacy/backup files

| File | Purpose | Final repo status |
|---|---|---|
| `app_main_gtcrn_backup.c` | Backup copy of an earlier GTCRN `app_main.c` implementation kept during development. | **Remove from final repo** unless history is intentionally being preserved. |
| `app_main_before_profile.c` | Older `app_main.c` snapshot from before profiling changes. | **Remove from final repo**. |

## Recommended final `main/` contents

```text
main/
├── CMakeLists.txt
├── app_main.c
├── audio_io.c
├── audio_io.h
├── board_config.h
├── gtcrn_int8.bin
├── gtcrn_int8.bin.json
├── idf_component.yml
├── usb_audio.c
├── usb_audio.h
├── wav.c
├── wav.h
├── wifi_audio.c
└── wifi_audio.h
```

`gtcrn_testvec.bin` is optional. Keep it when you want the firmware boot self-test; remove it when you intentionally want a smaller/cleaner deployment without the test vector.

## Architecture at a glance

```text
PC / microphone / file
        |
        v
    app_main.c
        |
        v
    GTCRN API
        |
        +--> gtcrn_int8.bin
        |
        v
  GTCRN runtime component

Hardware I/O:
  audio_io.c       -> I2S mic/speaker + SD
  usb_audio.c      -> USB microphone
  wav.c            -> WAV files
  board_config.h   -> pins + mode configuration
```

