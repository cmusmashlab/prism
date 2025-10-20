"""
This script runs the remaining time estimation algorithm.
requires
- datadrive / tasks / {task_name} / tracker / {model_hash}

after
- datadrive / tasks / {task_name} / observer / {model_hash} / loso / {sid} / remaining_time_distribution.pkl
"""

import argparse
import pickle
from tqdm import tqdm

from prism import config
from prism.tracker.algorithm import ViterbiTracker
from prism.tracker.algorithm.utils import get_graph, get_raw_cm
from prism.observer.algorithm import RemainingTimeEstimator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Name of the task', required=True)
    parser.add_argument('--model_hash', type=str, help='Model hash for saving the model', required=True)
    parser.add_argument('--test_sids', type=str, help='To specify test session IDs (e.g., --test_sids 10,11,14)', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    print('===== Observer Remaining Time Script Started =====')
    args = get_args()
    task_dir = config.datadrive / 'tasks' / args.task
    graph = get_graph(args.task)
    cm = get_raw_cm(cm_path=task_dir / 'har' / args.model_hash / 'loso' / 'cm_raw.pkl')
    if args.test_sids is None:
        test_sids = [fp.stem for fp in task_dir.glob('dataset/featurized/*.pkl') if 'inference-only' not in str(fp)]
    else:
        test_sids = args.test_sids.split(',')
    print('Test Session IDs:', test_sids)

    remaining_time_estimator = RemainingTimeEstimator(graph, mc_samples=1000)
    for test_sid in test_sids:
        print(f'-----Calculating remaining time for {test_sid}-----')
        with open(task_dir / 'har' / args.model_hash / 'loso' / test_sid / 'pred_raw.pkl', 'rb') as f:
            raw_pred_probas = pickle.load(f)
        with open(task_dir / 'har' / args.model_hash / 'loso' / test_sid / 'true.pkl', 'rb') as f:
            y_test = pickle.load(f)  # List[int]
        tracker = ViterbiTracker(graph, confusion_matrix=cm)
        remaining_time_estimator.reset()

        ground_truth = {}  # Dict[Step, int]
        time = 0
        for raw_pred_prob in tqdm(raw_pred_probas):
            _ = tracker.forward(raw_pred_prob)
            _ = remaining_time_estimator.forward(tracker.__get_best_entries_for_each_step__(tracker.curr_entries))  # Filter out -> otherwise, it will be too slow.
            time += 1
            if time == len(y_test):
                break
            if y_test[time - 1] != y_test[time] and graph.steps[y_test[time] + 1] not in ground_truth:
                ground_truth[graph.steps[y_test[time] + 1]] = time

        save_dir = task_dir / 'observer' / args.model_hash / 'loso' / test_sid
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'remaining_time_distribution.pkl', 'wb') as f:
            pickle.dump(
                {'expectations': remaining_time_estimator.expectations,
                'entropys': remaining_time_estimator.entropys,
                'ground_truth': ground_truth},
                f
            )