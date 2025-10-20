import importlib
import numpy as np

from .algorithm import Classifier
from .. import config

class HumanActivityRecognitionAPI():
    """
    API for Human Activity Recognition, mainly for the streaming purpose.
    """

    def __init__(self, task_name, model_hash):
        
        feature_extractor_names ={
            'audio': {
                'windower': 'BasicAudioWindower',
                'featurizer': 'SAMoSAAudioFeaturizer'
            },
            'motion': {
                'windower': 'BasicMotionWindower',
                'featurizer': 'BasicMotionFeaturizer'
            }
        }
        self.feature_extractor_models = {}
        prefix = 'prism.har.algorithm.modalities.'
        for modality, model_dict in feature_extractor_names.items():
            self.feature_extractor_models[modality] = {}
            for key, class_name in model_dict.items():
                module = importlib.import_module(prefix + modality)
                self.feature_extractor_models[modality][key] = getattr(module, class_name)()

        clf_path = config.datadrive / 'tasks' / task_name / 'har' / model_hash / 'all' / 'model.pkl'
        self.classifier = Classifier()
        self.classifier.load(clf_path)

    def __call__(self, data) -> list:
        """
        Predicts the activity label for the given data.
        If the audio and motion data have different lengths, only the last "windowed example" is used.

        Args:
        * data (dict): a dictionary containing the audio and motion data.
            audio should NOT be normalized.

        Returns:
        * list: a list of probability for each activity label.
        """
        audio_examples = self.feature_extractor_models['audio']['windower'](data['audio'])
        audio_feature = self.feature_extractor_models['audio']['featurizer'](audio_examples)
        motion_examples = self.feature_extractor_models['motion']['windower'](data['motion'])
        motion_feature = self.feature_extractor_models['motion']['featurizer'](motion_examples)
        if audio_feature.shape[0] != motion_feature.shape[0]:
            audio_feature = audio_feature[-1:]
            motion_feature = motion_feature[-1:]
        feature = np.hstack((motion_feature, audio_feature))  # order is important
        return self.classifier.predict_proba(feature)[0]