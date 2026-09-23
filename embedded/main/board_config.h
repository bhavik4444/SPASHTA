/*
 * board_config.h -- every pin and every mode switch, in one place.
 *
 * These are the assumptions I made for a generic ESP32-S3 N16R8 board with an
 * I2S MEMS microphone and an SD card on SPI. Nothing below is sacred; change
 * the numbers to match your wiring and nothing else in the project needs to
 * move.
 */
#ifndef BOARD_CONFIG_H
#define BOARD_CONFIG_H

/* ---------------------------------------------------------------------------
 * What the firmware does on boot.
 *
 * GTCRN_MODE_FILE     read /sdcard/noisy.wav, enhance, write /sdcard/clean.wav.
 *                     Start here. It is fully deterministic, needs no
 *                     microphone, and lets you diff the board's output against
 *                     the PC's sample-for-sample.
 *
 * GTCRN_MODE_RECORD   capture RECORD_SECONDS from the mic into PSRAM, then
 *                     enhance the buffer and write /sdcard/clean.wav.
 *                     This is the default, and it is the honest one: at
 *                     base_channels=32 the model runs slower than real time on
 *                     one core, so capture and processing are separated rather
 *                     than pretending to keep up.
 *
 * GTCRN_MODE_LIVE     continuous mic -> enhance -> I2S out. Only usable with a
 *                     model small enough to hit RTF < 1 (see the RT-lite
 *                     profile in INSTRUCTIONS.md). With the full model it will
 *                     report overruns and drop audio.
 * ------------------------------------------------------------------------- */
#define GTCRN_MODE_FILE     0
#define GTCRN_MODE_RECORD   1
#define GTCRN_MODE_LIVE     2

#ifndef GTCRN_MODE
#define GTCRN_MODE          GTCRN_MODE_RECORD
#endif

#define RECORD_SECONDS      20
#define OUTPUT_WAV_PATH     "/sdcard/clean.wav"
#define INPUT_WAV_PATH      "/sdcard/noisy.wav"

/* Run the embedded self-test before anything else. Costs about a second and
 * tells you immediately whether the board computes what your laptop computed. */
#define GTCRN_SELFTEST      1

/* ---------------------------------------------------------------------------
 * I2S input -- MEMS microphone (INMP441, ICS-43434, SPH0645 and friends).
 * 32-bit slots, mono, left channel. These parts put 24 significant bits at the
 * top of a 32-bit frame, which is why the reader shifts right by 11 rather than
 * by 16: shifting by 16 throws away 8 bits of headroom you actually want on
 * quiet defence recordings.
 * ------------------------------------------------------------------------- */
#define MIC_I2S_PORT        I2S_NUM_0
#define MIC_BCLK_GPIO       4
#define MIC_WS_GPIO         5
#define MIC_DIN_GPIO        6
#define MIC_SHIFT           11

/* ---------------------------------------------------------------------------
 * I2S output -- only used in LIVE mode (MAX98357A, PCM5102, ES8388 ...).
 * ------------------------------------------------------------------------- */
#define SPK_I2S_PORT        I2S_NUM_1
#define SPK_BCLK_GPIO       15
#define SPK_WS_GPIO         16
#define SPK_DOUT_GPIO       7

/* ---------------------------------------------------------------------------
 * SD card over SPI. Use a card you do not mind reformatting.
 * ------------------------------------------------------------------------- */
#define SD_MOSI_GPIO        11
#define SD_MISO_GPIO        13
#define SD_SCLK_GPIO        12
#define SD_CS_GPIO          10
#define SD_SPI_HOST         SPI2_HOST
#define SD_FREQ_KHZ         20000

/* Output gain applied after enhancement, before writing 16-bit PCM.
 * The model is trained at a fixed input loudness; audio_io normalises to that
 * and this puts it back. 1.0 keeps the original level. */
#define OUTPUT_GAIN         1.0f

#endif /* BOARD_CONFIG_H */
