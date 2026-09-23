import serial
import sounddevice as sd
import numpy as np
import struct
import time
import wave


# ============================================================
# CONFIG
# ============================================================

PORT = "/dev/ttyUSB0"
BAUD = 921600

SAMPLE_RATE = 16000
HOP = 256

# ------------------------------------------------------------
# Demo recording length.
#
# 20 seconds = 1250 GTCRN hops.
# GTCRN still processes 256 samples at a time and keeps its
# recurrent state across all hops.
# ------------------------------------------------------------
DURATION_SEC = 20.0

NUM_SAMPLES = int(
    DURATION_SEC * SAMPLE_RATE
)

# Make sure the recording is an exact number of GTCRN hops.
NUM_HOPS = NUM_SAMPLES // HOP
NUM_SAMPLES = NUM_HOPS * HOP

ACTUAL_DURATION_SEC = (
    NUM_SAMPLES / SAMPLE_RATE
)

MAGIC = 0x47544352
MAGIC_BYTES = struct.pack(
    "<I",
    MAGIC
)

HEADER = struct.Struct("<IH")

INPUT_WAV = "input.wav"
OUTPUT_WAV = "gtcrn_output.wav"

# GTCRN model training/input loudness.
TARGET_RMS = 0.10

# Output master volume after restoring the input level.
# This gives some headroom before the final safety clip.
OUTPUT_MASTER = 0.85


# ============================================================
# HELPERS
# ============================================================

def calculate_rms(x):
    x = np.asarray(
        x,
        dtype=np.float32
    )

    return float(
        np.sqrt(
            np.mean(
                x * x
            )
        )
    )


# ============================================================
# SERIAL
# ============================================================

print("Opening ESP32...")

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=2.0,
    write_timeout=5.0,
    dsrdtr=False,
    rtscts=False,
)

print("Waiting for ESP32 boot...")
time.sleep(4.0)

ser.reset_input_buffer()
ser.reset_output_buffer()

print("ESP32 serial ready.")


# ============================================================
# RECORD
# ============================================================

print()
print("========================================")
print("GTCRN DEMO RECORDING")
print("========================================")
print(
    f"Speak normally for "
    f"{ACTUAL_DURATION_SEC:.1f} seconds."
)
print()
print(
    "The complete recording will be normalized once "
    "to the GTCRN target RMS."
)
print()

audio = sd.rec(
    NUM_SAMPLES,
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)

sd.wait()

audio = audio[:, 0].astype(
    np.float32
)

print("Recording finished.")


# ============================================================
# INPUT MEASUREMENT + WHOLE-RECORDING NORMALIZATION
# ============================================================

input_rms = calculate_rms(
    audio
)

gain = (
    TARGET_RMS
    / max(input_rms, 1e-9)
)

# Reasonable safety limits. These are not per-hop.
gain = max(
    0.25,
    min(
        4.0,
        gain
    )
)

model_input = audio * gain

model_input = np.clip(
    model_input,
    -1.0,
    1.0
)

normalized_rms = calculate_rms(
    model_input
)

print()
print("========================================")
print("INPUT NORMALIZATION")
print("========================================")
print(
    f"Input RMS      : {input_rms:.6f}"
)
print(
    f"Target RMS     : {TARGET_RMS:.6f}"
)
print(
    f"Applied gain   : {gain:.6f}"
)
print(
    f"Model RMS      : {normalized_rms:.6f}"
)
print()


# ============================================================
# SAVE INPUT WAV
#
# Save the ORIGINAL recording, not the normalized model input.
# ============================================================

input_pcm = np.clip(
    audio,
    -1.0,
    1.0
)

input_pcm = np.rint(
    input_pcm * 32767.0
).astype(np.int16)

with wave.open(
    INPUT_WAV,
    "wb"
) as f:

    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(SAMPLE_RATE)

    f.writeframes(
        input_pcm.tobytes()
    )

print(
    f"Saved input: {INPUT_WAV}"
)


# ============================================================
# PACKET RX
# ============================================================

rx_buffer = bytearray()


def read_processed_hop():

    global rx_buffer

    while True:

        # ----------------------------------------------------
        # Find packet magic.
        # ----------------------------------------------------

        pos = rx_buffer.find(
            MAGIC_BYTES
        )

        if pos >= 0:

            if pos > 0:
                del rx_buffer[:pos]

            # ------------------------------------------------
            # Wait for full header.
            # ------------------------------------------------

            while len(rx_buffer) < HEADER.size:

                data = ser.read(512)

                if data:
                    rx_buffer.extend(data)

            magic, samples = HEADER.unpack(
                rx_buffer[:HEADER.size]
            )

            if magic != MAGIC:

                del rx_buffer[0]

                continue

            if samples != HOP:

                del rx_buffer[0]

                continue

            packet_size = (
                HEADER.size
                + HOP * 2
            )

            # ------------------------------------------------
            # Wait for full payload.
            # ------------------------------------------------

            while len(rx_buffer) < packet_size:

                data = ser.read(
                    packet_size
                    - len(rx_buffer)
                )

                if data:
                    rx_buffer.extend(data)

            pcm_start = HEADER.size
            pcm_end = (
                HEADER.size
                + HOP * 2
            )

            pcm = bytes(
                rx_buffer[
                    pcm_start:pcm_end
                ]
            )

            del rx_buffer[
                :packet_size
            ]

            return np.frombuffer(
                pcm,
                dtype=np.int16
            ).copy()

        # ----------------------------------------------------
        # Preserve partial magic.
        # ----------------------------------------------------

        if len(rx_buffer) > 3:

            rx_buffer = (
                rx_buffer[-3:]
            )

        data = ser.read(512)

        if data:
            rx_buffer.extend(data)


# ============================================================
# PROCESS THROUGH ESP32
# ============================================================

output_pcm = np.zeros(
    NUM_SAMPLES,
    dtype=np.int16
)

print()
print("========================================")
print("GTCRN PROCESSING")
print("========================================")
print(
    f"Sending {NUM_HOPS} hops "
    f"({ACTUAL_DURATION_SEC:.3f} seconds)"
)
print()

start_time = time.time()

for hop_number in range(NUM_HOPS):

    start = hop_number * HOP
    end = start + HOP

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We are slicing the ALREADY WHOLE-RECORDING-NORMALIZED
    # signal. We do NOT recalculate RMS here.
    # --------------------------------------------------------

    hop_float = model_input[
        start:end
    ]

    hop_pcm = np.rint(
        hop_float * 32767.0
    ).astype(np.int16)

    # --------------------------------------------------------
    # Send packet.
    # --------------------------------------------------------

    ser.write(
        HEADER.pack(
            MAGIC,
            HOP
        )
    )

    ser.write(
        hop_pcm.tobytes()
    )

    ser.flush()

    # --------------------------------------------------------
    # Receive processed hop.
    # --------------------------------------------------------

    processed = read_processed_hop()

    output_pcm[
        start:end
    ] = processed

    # --------------------------------------------------------
    # Progress every 50 hops so the terminal remains readable.
    # --------------------------------------------------------

    if (
        (hop_number + 1) % 50 == 0
        or
        (hop_number + 1) == NUM_HOPS
    ):

        elapsed = (
            time.time()
            - start_time
        )

        percent = (
            (hop_number + 1)
            / NUM_HOPS
            * 100.0
        )

        print(
            f"{percent:5.1f}%  "
            f"{hop_number + 1}/{NUM_HOPS} hops  "
            f"elapsed {elapsed:.1f}s"
        )


# ============================================================
# CONVERT ESP32 OUTPUT
# ============================================================

output_normalized = (
    output_pcm.astype(
        np.float32
    )
    / 32768.0
)

# ============================================================
# OUTPUT POST-PROCESSING
#
# We fed:
#
#     model_input = original * gain
#
# Therefore the GTCRN output is at the normalized level.
# Restore the original recording scale:
#
#     output_restored = output / gain
# ============================================================

output_restored = (
    output_normalized
    / max(gain, 1e-9)
)

# Overall demo volume/headroom.
output_restored *= OUTPUT_MASTER

restored_peak = float(
    np.max(
        np.abs(
            output_restored
        )
    )
)

output_restored = np.clip(
    output_restored,
    -1.0,
    1.0
)

output_rms = calculate_rms(
    output_restored
)

output_pcm_final = np.rint(
    output_restored * 32767.0
).astype(np.int16)

elapsed = (
    time.time()
    - start_time
)


# ============================================================
# RESULTS
# ============================================================

print()
print("========================================")
print("PROCESSING COMPLETE")
print("========================================")
print(
    f"Time           : {elapsed:.1f} s"
)
print(
    f"Audio duration : "
    f"{ACTUAL_DURATION_SEC:.3f} s"
)
print(
    f"Real-time ratio: "
    f"{elapsed / ACTUAL_DURATION_SEC:.2f}x"
)
print(
    f"Restored peak  : "
    f"{restored_peak:.6f}"
)
print(
    f"Output RMS     : "
    f"{output_rms:.6f}"
)


# ============================================================
# SAVE OUTPUT WAV
# ============================================================

with wave.open(
    OUTPUT_WAV,
    "wb"
) as f:

    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(SAMPLE_RATE)

    f.writeframes(
        output_pcm_final.tobytes()
    )

print(
    f"Saved output: {OUTPUT_WAV}"
)


# ============================================================
# PLAY ENHANCED AUDIO
# ============================================================

print()
print("========================================")
print("PLAYING GTCRN OUTPUT")
print("========================================")

sd.play(
    output_restored,
    SAMPLE_RATE
)

sd.wait()

print(
    "Playback finished."
)


# ============================================================
# CLOSE
# ============================================================

ser.close()

print()
print("Done.")
