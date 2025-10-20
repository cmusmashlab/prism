"""
This script featurizes the audio and motion data for a given task.
before
- datadrive / tasks / {task_name} / dataset / original

after
- datadrive / tasks / {task_name} / dataset / featurized
"""
import argparse
import importlib
import numpy as np
import pandas as pd
import pickle
from scipy.io import wavfile

from prism import config
from prism.har import params


def get_label(t, times, steps):
    """
    Get the label for a given timestamp `t` based on the provided `times` and `steps`.

    Args:
    * t (float): The timestamp in milliseconds.
    * times (list): A list of timestamps in milliseconds.
    * steps (list): A list of step labels corresponding to the timestamps.

    Returns:
    * str: The label corresponding to the timestamp `t`.
    """
    for i, time in enumerate(times):
        if t < time:
            return steps[i - 1]
    raise ValueError(f'No label found for {t=}') 


def remove_unnecessary_steps_in_the_beginning_and_end(audio_examples, motion_examples, labels, times):
    """
    Delete beginning and end examples classified as 'OTHER'.
    """
    other_categories = ['BEGIN', 'OTHER', 'END']  # TODO: this is hard-coded
    i = 0
    last_index = len(labels) - 1
    j = last_index
    while (i < len(labels)):
        if str(labels[i]) in other_categories:
            i += 1
        else:
            break
    while (j >= 0):
        if str(labels[j]) in other_categories:
            j -= 1
        else:
            break

    audio = audio_examples[i:j + 1,]
    imu = motion_examples[i:j + 1,]
    labels = labels[i:j + 1]
    times = times[i:j + 1]
    print(f'removed other at the beginning and ending: 0 -- {i}, {j} -- {last_index}')
    assert len(labels) == audio.shape[0] == imu.shape[0] == len(times)
    return audio, imu, labels, times


def create_feature_pkl(sid):
    sid_dir = task_dir / 'dataset' / 'original' / sid
    annotation_path = sid_dir / 'annotation.txt'
    if not annotation_path.exists():
        print(f'No annotation file found for {sid}. Skipping.')
        return None
    else:
        annotation = pd.read_csv(sid_dir / 'annotation.txt')
        annotation = annotation.sort_values(by='Timestamp')
        annotation_times, annotation_steps = annotation['Timestamp'].tolist(), annotation['Step'].tolist()
    # load data
    audio_file_path = sid_dir / 'audio.wav'
    motion_file_path = sid_dir / 'motion.txt'

    # get examples for batch processing based on the window parameters
    ## audio
    sr, audio_data = wavfile.read(audio_file_path)  # audio data (n_samples, n_channels))
    assert audio_data.dtype == 'int16' and sr == params.SAMPLE_RATE, 'Invalid audio file format.'
    audio_examples = feature_extractor_models['audio']['windower'](audio_data)

    ## motion
    motion_df = pd.read_csv(motion_file_path, sep='\s+', header=0, engine='python')
    motion_data = motion_df.to_numpy()
    motion_examples = feature_extractor_models['motion']['windower'](motion_data[:, 1:])  # Exclude timestamp column
    motion_timestamps = motion_df['timestamp'].tolist()
    print(f'Loaded data for {sid}: {audio_examples.shape=}, {motion_examples.shape=}')

    # align motion and audio examples
    aligned_audio_examples = []
    aligned_motion_examples = []
    labels = []
    relative_times = []
    for i in range(audio_examples.shape[0]):
        end_audio_sec = params.EXAMPLE_WINDOW_SECONDS + params.EXAMPLE_HOP_SECONDS * i
        motion_example_index = int((params.SAMPLE_RATE_IMU * end_audio_sec - params.WINDOW_LENGTH_IMU) / params.HOP_LENGTH_IMU)
        if motion_example_index >= motion_examples.shape[0]:
            print(f'out of bounds {motion_example_index=} {motion_examples.shape[0]=} {i=} {audio_examples.shape[0]=}')
            break
        motion_example = motion_examples[motion_example_index, :, :]

        # get the timestamp from the motion data
        last_frame_index_motion_example = int(motion_example_index * params.HOP_LENGTH_IMU + params.WINDOW_LENGTH_IMU)
        ms = motion_timestamps[last_frame_index_motion_example]

        if ms > annotation_times[-1]:
            print(f'Break at {ms=} > {annotation_times[-1]=}')
            break

        try:
            label = get_label(ms, annotation_times, annotation_steps)
            labels.append(label)
            relative_times.append(end_audio_sec * 1000)
            aligned_motion_examples.append(motion_example)
            aligned_audio_examples.append(audio_examples[i, :, :])
        except Exception as e:
            print(f'Error in getting label for {sid}: ', e)
            continue

    aligned_audio_examples = np.array(aligned_audio_examples)
    aligned_motion_examples = np.array(aligned_motion_examples)
    assert len(labels) == len(relative_times) == aligned_motion_examples.shape[0] == aligned_audio_examples.shape[0]

    print(f'Featurizing for {sid}: {aligned_audio_examples.shape=}, {aligned_motion_examples.shape=}, {len(labels)=}, {len(relative_times)=}')

    audio_examples, motion_examples, labels, new_times = remove_unnecessary_steps_in_the_beginning_and_end(
                                                            aligned_audio_examples,
                                                            aligned_motion_examples,
                                                            labels,
                                                            relative_times
                                                        )

    motion_feature = feature_extractor_models['motion']['featurizer'](motion_examples)
    audio_feature = feature_extractor_models['audio']['featurizer'](audio_examples)

    dataset = {
        'motion': motion_feature,
        'audio': audio_feature,
        'label': labels,
        'timestamp': new_times
    }

    print(f'Featurized done for {sid}: {dataset["motion"].shape=}, {dataset["audio"].shape=}, {len(dataset["label"])=}, {len(dataset["timestamp"])=}')
    return dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Task name', required=True)
    parser.add_argument('--sids', type=str, help='To specify session IDs to be processed (e.g., --sids 10,11,14)', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    print('===== HAR Featurization Script Started =====')
    args = get_args()
    task_dir = config.datadrive / 'tasks' / args.task
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
    feature_extractor_models = {}
    prefix = 'prism.har.algorithm.modalities.'
    for modality, model_dict in feature_extractor_names.items():
        feature_extractor_models[modality] = {}
        for key, class_name in model_dict.items():
            module = importlib.import_module(prefix + modality)
            feature_extractor_models[modality][key] = getattr(module, class_name)()

    to_be_featurized_sids = [f.stem for f in task_dir.glob('dataset/original/*') if f.is_dir()]
    if args.sids is None:
        already_featurized_sids = [f.stem for f in task_dir.glob('dataset/featurized/*.pkl') if f.is_file()]
        print('Already featurized session IDs:', already_featurized_sids)
        to_be_featurized_sids = [sid for sid in to_be_featurized_sids if sid not in already_featurized_sids]
    else:
        to_be_featurized_sids = [sid for sid in to_be_featurized_sids if sid in args.sids.split(',')]
    print('Session IDs to be featurized:', to_be_featurized_sids)

    for sid in to_be_featurized_sids:
        if '.DS_Store' in sid:
            continue

        print(f'-----Create feature pkl for {sid}-----')
        dataset = create_feature_pkl(sid)
        if dataset is None:
            continue
        save_fp = task_dir / 'dataset' / 'featurized' / f'{sid}.pkl'
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(save_fp, 'wb') as f:
            pickle.dump(dataset, f)
