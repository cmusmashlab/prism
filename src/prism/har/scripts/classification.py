"""
This script trains a frame-level classifier for the given task.
requires
- datadrive / tasks / {task_name} / dataset / featurized / *.pkl

after
- datadrive / tasks / {task_name} / har / {model_hash} / loso
- datadrive / tasks / {task_name} / har / {model_hash} / all
"""

import argparse
import datetime
import pathlib
import pickle
from typing import Union, List, Tuple

import numpy as np
import numpy.typing as npt

from prism import config
from prism.har.algorithm import Classifier, Evaluator


def load_imu_and_audio_data(
        pickle_files: List[Union[str, pathlib.Path]],
        steps: List[str],
        use_idle_filter: bool = False
    ) -> Tuple[npt.NDArray, List[int]]:
    """
    This function loads IMU and audio data from a set of pickle files.

    Args:
    * pickle_files (List[Union[str, pathlib.Path]]): a list of paths to the pickle files containing IMU and audio data.
    * steps (List[str]): a list of strings representing the different steps in the procedure.
    * use_idle_filter (bool): a flag to indicate whether to remove idle frames from the data.

    Returns:
    * X (npt.NDArray): a 2D numpy array containing the frame-based time-series IMU and audio data.
    * y (List[int]): a list of integers representing the index of the step for each time frame.
    """
    X, y = None, []
    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as fp:
            data = pickle.load(fp)
            
        motion_data, audio_data, labels = [], [], []
        for i, label in enumerate(data['label']):
            if label != 'OTHER':
                if use_idle_filter and data['is_idle'][i]:
                    # Skip the data if idle filter is used and the data is not marked as an idle frame
                    continue
                motion_data.append(data['motion'][i])
                audio_data.append(data['audio'][i])
                labels.append(label)
            else:
                print('Warning: OTHER label found in the data. Skipping...')

        x = np.hstack((np.array(motion_data), np.array(audio_data)))
        if X is None:
            X = x
        else:
            X = np.vstack((X, x))
        
        label_indices = list(map(lambda l: steps.index(l), labels))
        y += label_indices
    return X, y


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Task name', required=True)
    parser.add_argument('--test_sids', type=str, help='To specify test session IDs (e.g., --test_sids 10,11,14)', default=None)
    parser.add_argument('--n_train', type=int, help='Number of training sessions', default=None)
    parser.add_argument('--model_hash', type=str, help='Model hash for saving the model', default=None)
    parser.add_argument('--seed', type=int, help='Random seed', default=123)
    parser.add_argument('--use_idle_filter', action='store_true', help='Use idle filter to remove idle frames', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    print('===== HAR Classification Script Started =====')
    args = get_args()
    task_dir = config.datadrive / 'tasks' / args.task
    model_hash = args.model_hash if args.model_hash is not None else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    working_dir = task_dir / 'har' / model_hash
    working_dir.mkdir(parents=True, exist_ok=True)
    with open(working_dir / 'config.txt', 'w') as f:
        f.write(f'Use Idle Filter: {args.use_idle_filter}\n')
        f.write(f'Seed: {args.seed}\n')
        if args.n_train is not None:
            f.write(f'Number of Training Sessions: {args.n_train}\n')
        else:
            f.write('Number of Training Sessions: All\n')

    with open(task_dir / 'dataset' / 'steps.txt', 'r') as f:
        steps = [s.strip() for s in f.readlines()]

    feature_files = list(task_dir.glob('dataset/featurized/*.pkl'))
    if args.test_sids is not None:
        test_files = [f for f in feature_files if f.stem in args.test_sids.split(',')]
    else:
        test_files = feature_files
    print('Test Session IDs:', [f.stem for f in test_files])
    y_test_concatenated, y_pred_concatenated = [], []
    evaluator = Evaluator()

    for test_file in test_files:
        print(f'Leave-one-session-out: {test_file.stem}')
        sid_dir = working_dir / 'loso' / test_file.stem
        sid_dir.mkdir(parents=True, exist_ok=True)
                                                      
        train_files = [f for f in feature_files if f != test_file]
        if args.n_train is not None:
            assert args.n_train < len(train_files)
            train_files = np.random.RandomState(args.seed).choice(train_files, args.n_train, replace=False)

        train_info_text_file = sid_dir / 'train_info.txt'
        with open(train_info_text_file, 'w') as f:
            for train_file in train_files:
                f.write(f'{train_file.stem}\n')

        X_train, y_train = load_imu_and_audio_data(train_files, steps, use_idle_filter=args.use_idle_filter)
        X_test, y_test = load_imu_and_audio_data([test_file], steps)
        for class_id in range(len(steps)):  # add dummy data for classes not appeared
            if class_id not in y_train:
                X_train = np.vstack((X_train, np.zeros((1, X_train.shape[1]))))
                y_train = y_train + [class_id]

        clf = Classifier()
        clf.train(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)

        if args.use_idle_filter:
            # update y_pred_proba based on idle detection results
            with open(test_file, 'rb') as f:
                data = pickle.load(f)
            new_y_pred_proba = []
            for i, proba in enumerate(y_pred_proba):
                if data['is_idle'][i]:
                    # uniform distribution for idle frames; this will be ignored in the tracking step
                    new_y_pred_proba.append(np.ones_like(proba) / len(steps))
                else:
                    new_y_pred_proba.append(proba)
            y_pred_proba = np.array(new_y_pred_proba)           

        with open(sid_dir / 'true.pkl', 'wb') as f:
            pickle.dump(y_test, f)
        with open(sid_dir / 'pred_raw.pkl', 'wb') as f:
            pickle.dump(y_pred_proba, f)
        
        if args.use_idle_filter:
            y_test = [y for i, y in enumerate(y_test) if not data['is_idle'][i]]
            y_pred_proba = [y for i, y in enumerate(y_pred_proba) if not data['is_idle'][i]]

        evaluator.frame_level_metrics(y_test, np.argmax(y_pred_proba, axis=1), save_fp=sid_dir / 'result_raw.txt')
        evaluator.visualize_confusion_matrix(y_test, np.argmax(y_pred_proba, axis=1), steps, save_dir=sid_dir, suffix='raw')
            
        y_test_concatenated = np.concatenate((y_test_concatenated, y_test))
        y_pred_concatenated = np.concatenate((y_pred_concatenated, np.argmax(y_pred_proba, axis=1)))

    # aggregate results
    if len(y_test_concatenated) > 0:
        evaluator.frame_level_metrics(y_test_concatenated, y_pred_concatenated, save_fp=working_dir / 'loso' / 'result_raw.txt')
        evaluator.visualize_confusion_matrix(y_test_concatenated, y_pred_concatenated, steps, save_dir=working_dir / 'loso', suffix='raw')
        
    # train on all data
    print('Train on all data')
    (working_dir / 'all').mkdir(parents=True, exist_ok=True)
    X_all, y_all = load_imu_and_audio_data(feature_files, steps)
    clf = Classifier()
    clf.train(X_all, y_all)
    clf.save(working_dir / 'all' / 'model.pkl')