class SingleModalityWindower:
    """
    A class to window single modality data.
    """
    
    def __check_input_format__(self, raw_data):
        """
        This method checks if the input raw data is of the expected format.

        Args:
        * raw_data (np.ndarray): raw data to be checked, shape = (n_samples, n_channels)

        Raises:
        * ValueError: if raw_data is not expected type.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def __call__(self, raw_data):
        """
        Args:
        * raw_data (np.ndarray): raw data to be windowed, shape = (n_samples, n_channels)

        Returns:
        * np.ndarray: windowed examples, shape = (n_frames, window_length, n_input_dim)
        """

        raise NotImplementedError("This method should be implemented in subclasses.")


class SingleModalityFeaturizer:
    """ A class to featurize single modality data.
    """

    def __check_input_format__(self, examples):
        """
        This method checks if the input examples are of the expected format.

        Args:
        * examples (np.ndarray): examples to be checked, shape = (n_frame, window_length, n_input_dim)

        Raises:
        * ValueError: if examples is not expected type.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


    def __call__(self, examples):
        """
        Args:
        * examples (np.ndarray): examples to be featurized, shape = (n_frame, window_length, n_input_dim)

        Returns:
        * np.ndarray: featurized examples, shape = (n_frames, n_features)
        """
        raise NotImplementedError("This method should be implemented in subclasses.")