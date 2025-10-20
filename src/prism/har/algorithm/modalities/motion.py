from keras.models import Model, load_model
import numpy as np
import pickle
from scipy import stats

from .base import SingleModalityWindower, SingleModalityFeaturizer

import prism.config as config
from prism.har import params

class BasicMotionWindower(SingleModalityWindower):
    """
    A class to window motion data.
    """
    
    def __check_input_format__(self, raw_data):
        """
        This method checks if the input raw data is of the expected format.

        Args:
        * raw_data (np.ndarray): raw data to be checked, (n_samples, n_channels)
            n_channels should be 19 (08/15/2025)

        Raises:
        * ValueError: if raw_data is not expected type.
        """
        if not isinstance(raw_data, np.ndarray):
            raise ValueError(f'Expected np.ndarray, got {type(raw_data)}')
        if len(raw_data.shape) != 2:
            raise ValueError(f'Expected raw_data to be 2D, got {len(raw_data.shape)}D')
        if raw_data.shape[1] != 19:
            raise ValueError(f'Expected raw_data to have 19 channels, got {raw_data.shape[1]} channels')

    def __call__(self, raw_data):
        self.__check_input_format__(raw_data)

        window_length = params.WINDOW_LENGTH_IMU
        hop_length = params.HOP_LENGTH_IMU

        if raw_data.shape[0] < window_length:  # zero padding
            len_pad = int(np.ceil(window_length)) - raw_data.shape[0]
            to_pad = np.zeros((len_pad, ) + raw_data.shape[1:])
            raw_data = np.concatenate([raw_data, to_pad], axis=0)

        num_samples = raw_data.shape[0]
        num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
        shape = (num_frames, int(window_length)) + raw_data.shape[1:]
        strides = (raw_data.strides[0] * int(hop_length),) + raw_data.strides
        return np.lib.stride_tricks.as_strided(raw_data, shape=shape, strides=strides)


class SAMoSAMotionFeaturizer(SingleModalityFeaturizer):
    """ A class to featurize motion data using SAMoSA model.

    output feature dimensions = 256
    """

    def __init__(self):
        self.model = self._build_motion_model()
        self.norm_params = self._get_motion_norm_params()
        self.max_batch_size = 256  # For memory efficiency

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

    def _build_motion_model(self):
        path_to_model = config.datadrive / 'pretrained_models' / 'motion_model.h5'
        motion_model = load_model(path_to_model)
        dense2_op = motion_model.get_layer('dense_2').output
        return Model(
            inputs=motion_model.inputs,
            outputs=dense2_op,
            name='somohar_motion_model'
        )

    def _get_motion_norm_params(self):
        path_to_params = config.datadrive / 'pretrained_models' / 'motion_norm_params.pkl'
        with open(path_to_params, 'rb') as f:
            norm_params = pickle.load(f)
        return norm_params

    def _normalize_motion(self, motion):
        pseudo_max = self.norm_params['max']
        pseudo_min = self.norm_params['min']
        mean = self.norm_params['mean']
        std = self.norm_params['std']

        motion_normalized = 1 + (motion - pseudo_max) * 2 / (pseudo_max - pseudo_min)
        motion_normalized = (motion_normalized - mean) / std
        return motion_normalized

    def __call__(self, examples):
        """
        Args:
        * examples (np.ndarray): examples to be featurized, shape = (n_frame, window_length, n_input_dim)
            last dim: (timestamp, 'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
            'gravity.x', 'gravity.y', 'gravity.z', ...)
        """
        self.__check_input_format__(examples)

        acc = examples[:, :, 1:4]
        acc += examples[:, :, 4:7]
        acc *= params.GRAVITY  # Convert to m/s^2
        acc *= -1  # To match with SAMoSA's processing (empirically confirmed)

        features = []
        for start in range(0, len(acc), self.max_batch_size):
            end = start + self.max_batch_size
            batch = acc[start:end]
            batch = self._normalize_motion(batch)
            features.append(self.model(batch))
        return np.concatenate(features, axis=0)


class BasicMotionFeaturizer(SingleModalityFeaturizer):
    """
    A class to featurize motion data with MECE (Mutually Exclusive, Collectively Exhaustive) features.
    
    Input channels (19D -> filtered to 12D):
    - userAcceleration: x, y, z (indices 0-2)
    - gravity: x, y, z (indices 3-5) 
    - rotationRate: x, y, z (indices 6-8)
    - magneticField: x, y, z (indices 9-11) [EXCLUDED]
    - attitude: roll, pitch, yaw (indices 12-14) [USED]
    - attitude.quaternion: x, y, z, w (indices 15-18) [EXCLUDED - redundant with roll/pitch/yaw]
    
    MECE Feature Groups:
    1. UserAcceleration (3 channels): statistical(33) + temporal(18) + magnitude(6) + frequency(9) = 66 features
    2. Gravity (3 channels): statistical(33) + magnitude(6) + tilt/orientation(6) = 45 features  
    3. RotationRate (3 channels): statistical(33) + magnitude(6) = 39 features
    4. Attitude (3 channels): statistical(33) + circular_variance(9) = 42 features
    
    Total features: ~192 features per window
    """

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

    def _filter_channels(self, examples):
        """
        Filter input channels to remove redundant/unwanted data.
        
        Input: 19 channels [userAcc(3), gravity(3), rotation(3), magnetic(3), attitude(3), quaternion(4)]
        Output: 15 channels [userAcc(3), gravity(3), rotation(3), attitude(3)]
        """
        # Keep: userAcc(0-2), gravity(3-5), rotation(6-8), attitude_rpy(12-14)
        # Remove: magnetic(9-11), quaternion(15-18)
        selected_indices = list(range(0, 9)) + list(range(12, 15))  # 0-8, 12-14
        return examples[:, :, selected_indices]

    def _extract_statistical_features(self, data):
        """Extract statistical features from windowed data."""        
        features = []
        # Basic statistics
        features.append(np.mean(data, axis=1))      # mean
        features.append(np.std(data, axis=1))       # std
        features.append(np.min(data, axis=1))       # min
        features.append(np.max(data, axis=1))       # max
        features.append(np.ptp(data, axis=1))       # range (peak-to-peak)
        features.append(np.var(data, axis=1))       # variance
        
        # Higher order moments
        features.append(stats.skew(data, axis=1))         # skewness
        features.append(stats.kurtosis(data, axis=1))     # kurtosis
        
        # Percentiles
        features.append(np.percentile(data, 25, axis=1))  # 25th percentile
        features.append(np.percentile(data, 75, axis=1))  # 75th percentile
        features.append(np.median(data, axis=1))          # median (50th percentile)
        
        return np.concatenate(features, axis=1)

    def _extract_temporal_features(self, data):
        """Extract temporal features like velocity and acceleration."""
        features = []
        
        # First derivative (velocity)
        velocity = np.diff(data, axis=1)
        features.append(np.mean(velocity, axis=1))
        features.append(np.std(velocity, axis=1))
        features.append(np.max(np.abs(velocity), axis=1))  # max absolute velocity
        
        # Second derivative (acceleration)
        if velocity.shape[1] > 1:
            acceleration = np.diff(velocity, axis=1)
            features.append(np.mean(acceleration, axis=1))
            features.append(np.std(acceleration, axis=1))
            features.append(np.max(np.abs(acceleration), axis=1))  # max absolute acceleration
        
        return np.concatenate(features, axis=1)

    def _extract_magnitude_features(self, vector_data, name="magnitude"):
        """Extract features from 3D vector magnitude."""
        # Calculate magnitude for 3D vectors
        magnitude = np.sqrt(np.sum(vector_data ** 2, axis=2))
        
        features = []
        features.append(np.mean(magnitude, axis=1))
        features.append(np.std(magnitude, axis=1))
        features.append(np.min(magnitude, axis=1))
        features.append(np.max(magnitude, axis=1))
        features.append(np.var(magnitude, axis=1))
        
        # Zero crossing rate for magnitude
        zero_crossings = np.sum(np.diff(np.sign(magnitude - np.mean(magnitude, axis=1, keepdims=True))) != 0, axis=1)
        features.append(zero_crossings[:, np.newaxis])
        
        return np.concatenate([f[:, np.newaxis] if f.ndim == 1 else f for f in features], axis=1)

    def _extract_frequency_features(self, data, max_channels=6):
        """Extract frequency domain features using FFT."""
        features = []
        
        # Limit to first max_channels to avoid excessive features
        n_channels = min(data.shape[2], max_channels)
        
        for i in range(n_channels):
            # Compute FFT
            fft_vals = np.fft.fft(data[:, :, i], axis=1)
            power_spectrum = np.abs(fft_vals) ** 2
            
            # Only use positive frequencies
            pos_freqs = power_spectrum[:, :power_spectrum.shape[1]//2]
            
            # Spectral energy (sum of power spectrum)
            spectral_energy = np.sum(pos_freqs, axis=1)
            features.append(spectral_energy[:, np.newaxis])
            
            # Spectral centroid (weighted mean of frequencies)
            freqs = np.fft.fftfreq(data.shape[1])[:pos_freqs.shape[1]]
            spectral_centroid = np.sum(freqs[np.newaxis, :] * pos_freqs, axis=1) / (np.sum(pos_freqs, axis=1) + 1e-8)
            features.append(spectral_centroid[:, np.newaxis])
            
            # Dominant frequency (frequency with max power)
            dominant_freq_idx = np.argmax(pos_freqs, axis=1)
            dominant_freq = np.abs(freqs[dominant_freq_idx])
            features.append(dominant_freq[:, np.newaxis])
        
        return np.concatenate(features, axis=1) if features else np.zeros((data.shape[0], 0))

    def _extract_gravity_specific_features(self, gravity):
        """Extract gravity-specific features like tilt angles."""
        features = []
        
        # Gravity magnitude 
        gravity_mag = np.sqrt(np.sum(gravity ** 2, axis=2))
        
        # Tilt angle (angle from vertical, assuming z-axis is up when device is upright)
        tilt_angle = np.arccos(np.abs(gravity[:, :, 2]) / (gravity_mag + 1e-8))
        features.append(np.mean(tilt_angle, axis=1))
        features.append(np.std(tilt_angle, axis=1))
        features.append(np.max(tilt_angle, axis=1))
        
        # Gravity direction consistency (how stable is the gravity direction)
        gravity_norm = gravity / (gravity_mag[:, :, np.newaxis] + 1e-8)
        direction_consistency = np.std(gravity_norm, axis=1)
        features.extend([direction_consistency[:, i] for i in range(3)])
        
        return np.concatenate([f[:, np.newaxis] if f.ndim == 1 else f for f in features], axis=1)

    def _extract_attitude_specific_features(self, attitude):
        """Extract attitude-specific features like circular variance for angles."""
        features = []
        
        # Circular variance for each angle (roll, pitch, yaw)
        for i in range(3):  # roll, pitch, yaw
            # Circular variance: 1 - |mean(exp(i*angle))|
            angle_var = 1 - np.abs(np.mean(np.exp(1j * attitude[:, :, i]), axis=1))
            features.append(angle_var.real)
            
            # Angular velocity (rate of change)
            angle_diff = np.diff(attitude[:, :, i], axis=1)
            # Handle angle wrapping (e.g., from π to -π)
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            features.append(np.mean(np.abs(angle_diff), axis=1))
            features.append(np.std(angle_diff, axis=1))
        
        return np.concatenate([f[:, np.newaxis] if f.ndim == 1 else f for f in features], axis=1)

    def __call__(self, examples):
        """
        Args:
        * examples (np.ndarray): examples to be featurized, shape = (n_frame, window_length, n_input_dim)
            Expected 19 channels: [userAcceleration(3), gravity(3), rotationRate(3), 
                                  magneticField(3), attitude.rpy(3), attitude.quaternion(4)]

        Returns:
        * np.ndarray: featurized examples, shape = (n_frame, n_features)
            MECE feature groups:
            1. Linear motion (userAcceleration): statistical + temporal + magnitude + frequency
            2. Gravity-based: statistical + magnitude + tilt features  
            3. Rotation: statistical + magnitude features
            4. Attitude: statistical + circular variance features
        """
        self.__check_input_format__(examples)
        
        # Filter channels: remove magnetic field and quaternion (keep only useful 12 channels)
        filtered_data = self._filter_channels(examples)
        
        feature_groups = []
        
        # 1. USER ACCELERATION features (channels 0-2)
        user_acc = filtered_data[:, :, 0:3]
        
        # Statistical features for user acceleration
        user_acc_stat = self._extract_statistical_features(user_acc)
        feature_groups.append(user_acc_stat)
        
        # Temporal features for user acceleration
        user_acc_temporal = self._extract_temporal_features(user_acc)
        feature_groups.append(user_acc_temporal)
        
        # Magnitude features for user acceleration
        user_acc_mag = self._extract_magnitude_features(user_acc)
        feature_groups.append(user_acc_mag)
        
        # Frequency features for user acceleration
        user_acc_freq = self._extract_frequency_features(user_acc)
        feature_groups.append(user_acc_freq)
        
        # 2. GRAVITY features (channels 3-5)
        gravity = filtered_data[:, :, 3:6]
        
        # Statistical features for gravity
        gravity_stat = self._extract_statistical_features(gravity)
        feature_groups.append(gravity_stat)
        
        # Magnitude features for gravity
        gravity_mag = self._extract_magnitude_features(gravity)
        feature_groups.append(gravity_mag)
        
        # Gravity-specific features (tilt, orientation)
        gravity_specific = self._extract_gravity_specific_features(gravity)
        feature_groups.append(gravity_specific)
        
        # 3. ROTATION RATE features (channels 6-8)
        rotation = filtered_data[:, :, 6:9]
        
        # Statistical features for rotation
        rotation_stat = self._extract_statistical_features(rotation)
        feature_groups.append(rotation_stat)
        
        # Magnitude features for rotation
        rotation_mag = self._extract_magnitude_features(rotation)
        feature_groups.append(rotation_mag)
        
        # 4. ATTITUDE features (channels 9-11)
        attitude = filtered_data[:, :, 9:12]
        
        # Statistical features for attitude
        attitude_stat = self._extract_statistical_features(attitude)
        feature_groups.append(attitude_stat)
        
        # Attitude-specific features (circular variance)
        attitude_specific = self._extract_attitude_specific_features(attitude)
        feature_groups.append(attitude_specific)
        
        # Concatenate all features
        all_features = np.concatenate(feature_groups, axis=1)
        
        # Handle any NaN or inf values
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return all_features
        
        