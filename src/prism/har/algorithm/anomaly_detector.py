import numpy as np
from sklearn.mixture import GaussianMixture

class GMMAnomalyDetection:

    def __init__(self,
                 n_components=2,
                 window_sec=15,
                 skip_sec=30,
                 decay=0.9,
                 epsilon=0.01,
                 ):
        """Initialize model.

        Args:
            n_components: Number of GMM components.
                default = 2
            window_sec: Window length in seconds.
                default = 15
            skip_sec: Length of the begginig irrelevant part to be skipped in seconds.
                default = 30
            decay: Parameter for GMM, indicating weight decay.
                default = 0.9
            epsilon: Parameter for GMM, indicating a small number for calculating probability.
                default = 0.01
        """
        self.n_components = n_components
        self.window_sec = window_sec
        self.skip_sec = skip_sec
        self.decay = decay
        self.epsilon = epsilon

    @staticmethod
    def calc_importance(gmm, dimensions, frame_data):
        """Calculate importance of each feature to the output of the given frame.

        Args:
            gmm (GaussianMixture): GMM model.
            dimensions (List[dict]): `dict` has keys `name`, `first`, and `last`.
            frame_data (numpy.ndarray): feature data for a single frame.

        Returns:
            importance (dict): Keys are names of features and values are corresponding importance from 0 to 1.
        """

        importance = {}
        for feature in dimensions:
            data = gmm.means_.copy()
            data[:, feature['first']: feature['last']] = frame_data[feature['first']: feature['last']]
            likelihood = gmm.score(data) / (feature['last'] - feature['first'])
            # In almost all cases, the likelihood values are negative.
            # Next code affirms that all values should be lower than or equal to 0.
            importance[feature['name']] = min(likelihood, 0)

        sum_likelihood = sum([-v for v in importance.values()])
        if sum_likelihood > 0:
            for k, v in importance.items():
                importance[k] = -v / sum_likelihood
        return importance

    def __call__(self, features, dimensions, fps):
        """Predict.

        Args:
            features (numpy.ndarray): Features.
            dimensions (List[dict]): `dict` has keys `name`, `first`, and `last`.
            fps (float): frame per second.

        Returns:
            events (List[dict]):
                `dict` has keys `start`, `end`, `score`, `min`, and `max`.
                The first three values are corresponding time, time, and anomaly score.
                The remaining two are not included if the window time is within `self.skip_sec`.
                The two values are most unlikely and likely frame index, respectively.
                `score` is likelihood, so the lower it is, the more anomalous the window is.
        """
        window, skip = map(lambda x: int(x * fps), (self.window_sec, self.skip_sec))
        events = []

        self.gmm = GaussianMixture(n_components=self.n_components)
        self.gmm.fit(features[:window])

        self.single_gmm = GaussianMixture(n_components=1)
        self.single_gmm.fit(features[:window])

        weights = self.gmm.weights_
        means = self.gmm.means_
        covars = self.gmm.covariances_

        for frame_index in range(0, len(features) - window, window):
            data = features[frame_index: frame_index + window]

            proba = self.gmm.predict_proba(data)
            proba = (1 - self.epsilon) * proba + self.epsilon / self.gmm.n_components

            new_covars = np.zeros((self.gmm.n_components, data.shape[1], data.shape[1]))
            for j in range(data.shape[0]):
                new_covars += np.multiply.outer(proba[j], np.outer(data[j], data[j]))

            weights = self.decay * weights + (1 - self.decay) * proba.sum(axis=0) / window
            means = self.decay * means + (1 - self.decay) * np.dot(proba.T, data) / window
            covars = self.decay * covars + (1 - self.decay) * new_covars / window

            self.gmm.weights_ = weights
            self.gmm.means_ = means / weights[:, np.newaxis]
            self.gmm.covariances_ = covars / weights[:, np.newaxis, np.newaxis]
            for component in range(self.n_components):
                self.gmm.covariances_[component, :,
                                      :] -= np.outer(self.gmm.means_[component], self.gmm.means_[component])

            event = {'start': frame_index / fps, 'end': (frame_index + len(data)) / fps, 'score': self.gmm.score(data)}

            if frame_index >= skip:
                event = dict(event, min=[], max=[])

                for component in range(self.gmm.n_components):
                    self.single_gmm.means_ = self.gmm.means_[component][np.newaxis]
                    self.single_gmm.covariances_ = self.gmm.covariances_[component][np.newaxis]

                    scores = self.single_gmm.score_samples(data)
                    # most common data
                    max_index = int(scores.argmax()) + frame_index
                    max_event = {
                                'frame': max_index,
                                'time': max_index / fps,
                                'importance': self.calc_importance(self.single_gmm, dimensions,
                                                                   data[int(scores.argmax())]),
                                }
                    event['max'].append(max_event)
                    # most anomaly data
                    min_index = int(scores.argmin()) + frame_index
                    min_event = {
                                'frame': min_index,
                                'time': min_index / fps,
                                'importance': self.calc_importance(self.single_gmm, dimensions,
                                                                   data[int(scores.argmin())]),
                                }
                    event['min'].append(min_event)

            events.append(event)

        return events
    

def fetch_topk_windows(windows, k=5, min_time_distance=None):
    """Extract windows with the lowest score from the output of __call__.

    Args:
        windows (List[dict]): A target output. The format is the same as the output value of __call__.
        k (int) : A maximum number of selected windows.
            default = 5
        min_time_distance (float) : Selected windows should be further away than this second.
            default = None

    Returns:
        important_windows (List[dict]): An extracted output. The format is the same as the output value of __call__.
        Note.
            An element of the output list corresponds to one frame.
            Frames with short time intervals are skipped.
            A window may be divided into two elements corresponding to frames.
    """
    windows = filter(lambda x: 'max' in x and 'min' in x, windows)
    # score is likelihood, so choose lower ones as anomaly.
    sorted_windows = sorted(windows, key=lambda x: x['score'], reverse=False)
    # fetch topk windows in enough time away
    important_windows = []
    if min_time_distance is None:
        min_time_distance = -1
    for window in sorted_windows:
        if len(important_windows):
            times = np.array([w['end'] for w in important_windows])
            now_time = window['end']
            if np.all(abs(times - now_time) > min_time_distance):
                important_windows.append(window)
        else:
            important_windows.append(window)

        if len(important_windows) >= k:
            break

        if len(important_windows) >= k:
            break

    important_windows = sorted(important_windows, key=lambda x: x['start'])

    return important_windows