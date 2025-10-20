"""
Check the format of the "original" dataset.
- datadrive / tasks / {task_name} / dataset / original
"""

import argparse
import pandas as pd
from scipy.io import wavfile

from prism import config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Task name', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataset_dir = config.datadrive / 'tasks' / args.task / 'dataset'

    with open(dataset_dir / 'steps.txt', 'r') as f:
        steps = [s.strip() for s in f.readlines()]
        for step in steps:
            assert step.lower() == step, f'Step {step} is not in lowercase.'
        steps = ['BEGIN'] + steps + ['END']
    print(steps)

    error_cnt = 0
    for sid_dir in dataset_dir.glob('original/*'):
        try:
            if not sid_dir.is_dir():
                continue
            audio_path = sid_dir / 'audio.wav'
            motion_path = sid_dir / 'motion.txt'
            annotation_path = sid_dir / 'annotation.txt'

            sr, audio = wavfile.read(audio_path)
            assert sr == 16000, f'Sampling rate is {sr} instead of 16000.'
            audio_length = len(audio) / sr
            motion = pd.read_csv(motion_path, sep='\s', engine='python')
            motion_length = motion['timestamp'].iloc[-1] / 1000
            assert abs(audio_length - motion_length) < 1, f'Length mismatch between audio and motion: {audio_length}, {motion_length}'

            annotation = pd.read_csv(annotation_path)
            annotation_length = annotation['Timestamp'].iloc[-1] / 1000
            assert audio_length >= annotation_length, f'Annotation length is longer than audio: {annotation_length}, {audio_length}'

            assert annotation['Step'].iloc[0] == 'BEGIN', f'First step is not BEGIN.'
            assert annotation['Step'].iloc[-1] == 'END', f'Last step is not END.'
            for step in annotation['Step']:
                assert step in steps, f'Unknown step: {step}'
        except Exception as e:
            error_cnt += 1
            print(f'Error in {sid_dir.stem} --  {e}')
    if error_cnt == 0:
        print('All dataset is in the correct format.')
    else:
        print(f'{error_cnt} errors found in the dataset.')
