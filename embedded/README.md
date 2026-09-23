# Embedded GTCRN Runtime

This is the ESP-IDF project for the ESP32-S3 deployment.

## Build

From the repository root:

```bash
cd embedded
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash
```

For boot logs:

```bash
idf.py -p /dev/ttyUSB0 monitor
```

Exit monitor with `Ctrl+]`.

## Layout

```text
embedded/
├── CMakeLists.txt
├── dependencies.lock
├── partitions.csv
├── sdkconfig.defaults
├── components/
│   └── gtcrn/
│       ├── CMakeLists.txt
│       ├── gt_dot_i8_s3.S
│       ├── gtcrn_internal.h
│       ├── gtcrn_net.c
│       ├── gtcrn_ops.c
│       └── include/gtcrn.h
├── main/
│   ├── app_main.c
│   ├── audio_io.c
│   ├── audio_io.h
│   ├── board_config.h
│   ├── CMakeLists.txt
│   ├── gtcrn_int8.bin
│   ├── gtcrn_int8.bin.json
│   ├── idf_component.yml
│   ├── usb_audio.c
│   ├── usb_audio.h
│   ├── wav.c
│   └── wav.h
├── host/
│   └── esp_audio_bridge.py
└── docs/
    └── GTCRN_main_file_guide.md
```

`gtcrn_int8.bin` is the model currently referenced by the firmware. The original archive also contained a different, larger model export under `model/`; that redundant full-model artifact is intentionally not copied here.

The optional boot test-vector is also not copied automatically because the archive's `model/gtcrn_testvec.bin` must be generated against the exact deployment blob. Generate a fresh vector with `quantization/verify_export.py` when you want the ESP32 boot self-test.
