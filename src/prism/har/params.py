"""
Audio
"""
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.6
STFT_HOP_LENGTH_SECONDS = 0.03
EXAMPLE_WINDOW_SECONDS = 96 * 0.03   # Each example contains 96 frames -- 2.88 sec
# EXAMPLE_HOP_SECONDS = 96*0.03*0.1    # with zero overlap.
EXAMPLE_HOP_SECONDS = 7 * 0.03  # 7 frames overlap

MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.001  # Offset used for stabilized log of input mel-spectrogram.
NUM_MEL_BINS = 64  # Frequency bands in input mel-spectrogram patch.

"""
Motion
"""
SAMPLE_RATE_IMU = 50
WINDOW_LENGTH_IMU = 100
HOP_LENGTH_IMU = 10
GRAVITY = 9.81  # m/s^2