# SPASHTA GTCRN — Clean Embedded/Quantization Package

## Package structure

```text
embedded/
quantization/
.gitignore
PACKAGE_MANIFEST.md
```

## Included

### `embedded/`

The complete current ESP-IDF firmware source/config needed to rebuild the deployed ESP32-S3 application:

- ESP-IDF project `CMakeLists.txt`
- `dependencies.lock`
- `partitions.csv`
- `sdkconfig.defaults`
- `main/` source, headers, component manifest, and the currently referenced lite model blob
- `components/gtcrn/` runtime source and ESP32-S3 SIMD kernel
- one maintained PC serial demo under `host/`
- file guide under `docs/`

### `quantization/`

The core model export/verification code:

- `export_int8.py`
- `verify_export.py`
- `gtcrn_ref.py`
- `host_test.c`
- `make_synthetic_blob.py`
- `qat_finetune.py`
- file guide

## Intentionally excluded

1. **ESP-IDF build output**
   - `firmware/build/**`
   - generated Ninja/CMake caches
   - `.elf`, `.map`, bootloader/app binaries and other build products

2. **Generated ESP-IDF configuration**
   - `firmware/sdkconfig`
   - generated local configuration/cache files

3. **Managed dependencies**
   - `firmware/managed_components/**`

   The main component manifest declares `espressif/usb_host_uac`, and `dependencies.lock` is retained so ESP-IDF can restore the dependency.

4. **Backup/experimental source snapshots**
   - `gt_dot_i8_s3_old.S`
   - `gtcrn_net.c.orig`
   - `gtcrn_net_backup.c`
   - `gtcrn_net_before_profile.c`
   - `gtcrn_ops_backup_11p7s.c`
   - `gtcrn_ops_backup_before_n8.c`
   - `gtcrn_ops_before_fast_math.c`
   - `gtcrn_ops_before_n8_simd.c`

5. **Unused placeholder source**
   - `wifi_audio.c`
   - `wifi_audio.h`

   The current `main/CMakeLists.txt` does not build these files.

6. **Redundant model artifacts**
   - original `model/gtcrn_int8.bin`
   - original `model/gtcrn_int8.bin.json`

   The deployed firmware uses the smaller model in `embedded/main/gtcrn_int8.bin`.

7. **Test vector**
   - original `model/gtcrn_testvec.bin`

   It is intentionally not copied because a test vector is only valid when generated against the exact blob being deployed. Generate a fresh one with `quantization/verify_export.py` for `embedded/main/gtcrn_int8.bin`.

8. **Python/generated artifacts**
   - `tools/__pycache__/`
   - `*.pyc`
   - old/duplicate serial-bridge variants
   - generated WAV/audio files

## Parent SPASHTA repository

The current public repository is a training/model-development repository. Its root contains training source such as `model.py`, `train.py`, `infer.py`, `dataset.py`, `sound_mixer.py`, and multiple checkpoint/sample-data directories. The quantization scripts in this package intentionally rely on the root training implementation rather than duplicating it.

