"""
This script detects idle frames in a given task's dataset using anomaly detection.

The following data will be updated:
- datadrive / tasks / {task_name} / dataset / featurized / *.pkl
"""
import argparse
import numpy as np
import pickle

from prism import config
from prism.har.algorithm import GMMAnomalyDetection, fetch_topk_windows
from prism.har.params import EXAMPLE_HOP_SECONDS

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Task name', required=True)
    parser.add_argument('--sids', type=str, help='To specify session IDs to be processed (e.g., --sids 10,11,14)', default=None)
    parser.add_argument('--idle_ratio', type=float, help='Anomaly ratio', default=0.1)
    return parser.parse_args()

skip_sec = 5
window_sec = EXAMPLE_HOP_SECONDS * 10

if __name__ == '__main__':
    print('===== HAR Idle Detection Script Started =====')
    args = get_args()
    task_dir = config.datadrive / 'tasks' / args.task
    to_be_processed_sids = [f.stem for f in task_dir.glob('dataset/featurized/*.pkl')]
    if args.sids is not None:
        to_be_processed_sids = [sid for sid in to_be_processed_sids if sid in args.sids.split(',')]
    print('Session IDs to be processed:', to_be_processed_sids)

    for sid in to_be_processed_sids:
        with open(task_dir / 'dataset' / 'featurized' / f'{sid}.pkl', 'rb') as f:
            data = pickle.load(f)
        X = np.hstack((data['motion'], data['audio']))
        model = GMMAnomalyDetection(window_sec=window_sec, skip_sec=skip_sec)
        events = model(
            features=X - np.mean(X, axis=0),
            dimensions=[{'name': 'concatenated', 'first': 0, 'last': X.shape[1]}],
            fps=1/EXAMPLE_HOP_SECONDS,
        )
        k = int(len(events) * (1 - args.idle_ratio))
        topk_results = fetch_topk_windows(events, k)

        idle_flags = []
        for t in data['timestamp']:
            time_sec = t / 1000.0
            is_idle = True
            if time_sec < skip_sec:
                is_idle = False
            else:
                for result in topk_results:
                    if result['start'] <= time_sec <= result['end']:
                        is_idle = False
                        break
            idle_flags.append(is_idle)
        data['is_idle'] = idle_flags
        assert len(idle_flags) == len(data['timestamp'])
        print('Idle frame ratio: ', sum(idle_flags) / len(idle_flags))
        with open(task_dir / 'dataset' / 'featurized' / f'{sid}.pkl', 'wb') as f:
            pickle.dump(data, f)
        print(f'Idle detection done for {sid}')