from keras.models import Model, load_model
import numpy as np

import prism.config as config
from prism.har import params

class BasicAudioWindower:
    """
    A class to window audio data.
    Mel-Spectrograms are computed from the audio data.
    """
    def __check_input_format__(self, raw_data):
        """
        This method checks if the input raw data is of the expected format.

        Args:
        * raw_data (np.ndarray): raw data to be checked, shape = (n_samples,)
            values should be in [-1.0, +1.0] range.

        Raises:
        * ValueError: if raw_data is not expected type.
        """
        if not isinstance(raw_data, np.ndarray):
            raise ValueError(f'Expected np.ndarray, but got {type(raw_data)}')
        if len(raw_data.shape) != 1:
            raise ValueError(f'Expected raw_data to be 1D, got {len(raw_data.shape)}D')
        if np.max(np.abs(raw_data)) > 1.0:
            raise ValueError(f'Expected raw_data values to be in [-1.0, +1.0], but got max value {np.max(np.abs(raw_data))}')

    def __call__(self, raw_data):
        raw_data = raw_data / (2**15)        # Convert signed 16-bit to [-1.0, +1.0]
        if len(raw_data.shape) > 1:          # Convert to mono.
            raw_data = np.mean(raw_data, axis=1)

        self.__check_input_format__(raw_data)

        log_mel = self._log_mel_spectrogram(
            raw_data,
            audio_sample_rate=params.SAMPLE_RATE,
            log_offset=params.LOG_OFFSET,
            window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
            hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
            num_mel_bins=params.NUM_MEL_BINS,
            lower_edge_hertz=10,
            upper_edge_hertz=params.SAMPLE_RATE // 2
        )

        # Frame features into examples.
        example_window_length = int(round(params.EXAMPLE_WINDOW_SECONDS / params.STFT_HOP_LENGTH_SECONDS))  # 96
        example_hop_length = int(round(params.EXAMPLE_HOP_SECONDS / params.STFT_HOP_LENGTH_SECONDS))  # 7
        return self._frame(log_mel, window_length=example_window_length, hop_length=example_hop_length)

    # MFCC Spectrogram conversion code from VGGish, Google Inc.
    # https://github.com/tensorflow/models/tree/master/research/audioset

    def _frame(self, data, window_length, hop_length):
        if data.shape[0] < window_length:
            # pad zeros
            len_pad = int(np.ceil(window_length)) - data.shape[0]
            to_pad = np.zeros((len_pad, ) + data.shape[1:])
            data = np.concatenate([data, to_pad], axis=0)
        num_samples = data.shape[0]
        num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
        shape = (num_frames, int(window_length)) + data.shape[1:]
        strides = (data.strides[0] * int(hop_length),) + data.strides
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    def _periodic_hann(self, window_length):
        return 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length)))


    def _stft_magnitude(self, signal, fft_length, hop_length=None, window_length=None):
        frames = self._frame(signal, window_length, hop_length)
        window = self._periodic_hann(int(window_length))
        windowed_frames = frames * window
        return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


    def _hertz_to_mel(self, frequencies_hertz):
        # Mel spectrum constants and functions.
        _MEL_BREAK_FREQUENCY_HERTZ = 700.0
        _MEL_HIGH_FREQUENCY_Q = 1127.0
        return _MEL_HIGH_FREQUENCY_Q * np.log(1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


    def _get_mel_matrix(self,
                        audio_sample_rate,
                        num_mel_bins=20,
                        num_spectrogram_bins=129,
                        lower_edge_hertz=125.0,
                        upper_edge_hertz=3800.0):

        nyquist_hertz = audio_sample_rate / 2.
        if lower_edge_hertz >= upper_edge_hertz:
            raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" % (lower_edge_hertz, upper_edge_hertz))
        spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
        spectrogram_bins_mel = self._hertz_to_mel(spectrogram_bins_hertz)
        band_edges_mel = np.linspace(self._hertz_to_mel(lower_edge_hertz), self._hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
        # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
        # of spectrogram values.
        mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
        for i in range(num_mel_bins):
            lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
            lower_slope = ((spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel))
            upper_slope = ((upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel))
            # .. then intersect them with each other and zero.
            mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
        mel_weights_matrix[0, :] = 0.0
        return mel_weights_matrix


    def _log_mel_spectrogram(self, data, audio_sample_rate, log_offset, window_length_secs, hop_length_secs, **kwargs):
        window_length_samples = audio_sample_rate * window_length_secs
        hop_length_samples = audio_sample_rate * hop_length_secs
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

        spectrogram = self._stft_magnitude(data, fft_length=fft_length, hop_length=hop_length_samples, window_length=window_length_samples)
        mel_matrix = self._get_mel_matrix(audio_sample_rate, num_spectrogram_bins=spectrogram.shape[1], **kwargs)
        mel_spectrogram = np.dot(spectrogram, mel_matrix)
        return np.log(mel_spectrogram + log_offset)
    

class SAMoSAAudioFeaturizer:
    """ A class to featurize audio data using SAMoSA model.

    output feature dimensions = 128
    """

    def __init__(self):
        self.audio_model = self._build_audio_model()
        self.max_batch_size = 256  # For memory efficiency

    def _build_audio_model(self):
        path_to_model = config.datadrive / 'pretrained_models' /'audio_model.h5'
        audio_model = load_model(path_to_model)
        fc2_op = audio_model.get_layer('fc2').output
        final_model = Model(
            inputs=audio_model.inputs,
            outputs=fc2_op,
            name='somohar_sound_model'
        )
        return final_model

    def __check_input_format__(self, examples):
        """
        This method checks if the input examples are of the expected format.

        Args:
        * examples (np.ndarray): examples to be checked, shape = (n_frame, window_length, n_input_dim)

        Raises:
        * ValueError: if examples is not expected type.
        """
        if not isinstance(examples, np.ndarray):
            raise ValueError(f'Expected np.ndarray, got {type(examples)}')
        if len(examples.shape) != 3:
            raise ValueError(f'Expected examples to be 3D, got {len(examples.shape)}D')
        
    def __call__(self, examples):
        self.__check_input_format__(examples)

        features = []
        for start in range(0, len(examples), self.max_batch_size):
            end = start + self.max_batch_size
            batch = examples[start:end]
            features.append(self.audio_model(batch))
        return np.concatenate(features, axis=0)
