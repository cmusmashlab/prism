"""
This script performs tracking using the Viterbi algorithm on a specified task dataset.
requires
- datadrive / tasks / {task_name} / dataset / featurized / *.pkl
- datadrive / tasks / {task_name} / har / {model_hash} 


after
- datadrive / tasks / {task_name} / tracker / {model_hash}
"""

import argparse
import pickle
import time

import numpy as np

from prism import config
from prism.har.algorithm import Evaluator
from prism.tracker.algorithm import ViterbiTracker
from prism.tracker.algorithm.utils import get_graph, get_raw_cm, get_proxy_graph


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Name of the task', required=True)
    parser.add_argument('--test_sids', type=str, help='To specify test session IDs (e.g., --test_sids 10,11,14)', default=None)
    parser.add_argument('--model_hash', type=str, help='Model hash for saving the model', required=True)
    parser.add_argument('--graph_type', type=str, help='Graph type. all,generated,train', default='all')
    return parser.parse_args()


def get_graph_by_type(task_name, graph_type, train_sids=None):
    """
    Get the graph based on the specified type.
    
    Args:
    * task_name (str): Name of the task.
    * graph_type (str): Type of the graph to retrieve. Options are 'all', 'generated', 'train'.
    * train_sids (list, optional): List of training session IDs. Required if graph_type is 'train'.
    
    Returns:
    * Graph object: The graph corresponding to the specified type.
    """
    if graph_type == 'all':
        return get_graph(task_name)
    elif graph_type == 'generated':
        edges_fp = config.datadrive / 'tasks' / task_name / 'tracker' / 'generated_edges.pkl'
        return get_proxy_graph(task_name, edges_fp=edges_fp, train_sids=train_sids)
    elif graph_type == 'train':
        return get_graph(task_name, train_sids=train_sids)
    else:
        raise ValueError('Invalid graph type. Choose from all, generated, train.')


if __name__ == '__main__':
    print('===== Tracker Tracking Script Started =====')
    args = get_args()
    np.random.seed(2024)
    task_dir = config.datadrive / 'tasks' / args.task

    with open(task_dir / 'dataset' / 'steps.txt', 'r') as f:
        steps = [s.strip() for s in f.readlines()]
    raw_cm = get_raw_cm(cm_path=task_dir / 'har' / args.model_hash / 'loso' / 'cm_raw.pkl')
    if args.test_sids is None:
        test_sids = [fp.stem for fp in task_dir.glob('dataset/featurized/*.pkl') if 'inference-only' not in str(fp)]
    else:
        test_sids = args.test_sids.split(',')
    print('Test Session IDs:', test_sids)

    y_test_concatenated, y_pred_concatenated = [], []
    evaluator = Evaluator()
    suffix = f'graph_type={args.graph_type}'

    for test_sid in test_sids:
        print(f'-----Tracking for {test_sid}-----')
        sid_har_dir = task_dir / 'har' / args.model_hash / 'loso' / test_sid
        with open(sid_har_dir / 'pred_raw.pkl', 'rb') as f:
            raw_pred_probas = pickle.load(f)
        with open(sid_har_dir / 'true.pkl', 'rb') as f:
            trues = pickle.load(f) 

        # graph
        train_info_fp = sid_har_dir / 'train_info.txt'
        if train_info_fp.exists():
            with open(sid_har_dir / 'train_info.txt', 'r') as f:
                train_sids = [line.strip() for line in f.readlines()]
        else:
            train_sids = None
            assert args.graph_type == 'all', 'Graph type should be all if train_sids is not provided'
        graph = get_graph_by_type(args.task, args.graph_type, train_sids)

        tracker = ViterbiTracker(graph, confusion_matrix=raw_cm)
        start_time = time.time()
        viterbi_pred_probas = []
        for pred_probas in tracker.predict_batch(raw_pred_probas):
            viterbi_pred_probas.append(pred_probas)
        viterbi_pred_probas = np.array(viterbi_pred_probas)
        elapsed_time = time.time() - start_time
        print(f'Elapsed time: {elapsed_time:.2f} sec -> {elapsed_time / len(raw_pred_probas):.4f} sec per frame')
        assert raw_pred_probas.shape == viterbi_pred_probas.shape

        # save results
        sid_tracker_dir = task_dir / 'tracker' / args.model_hash / 'loso' / test_sid
        sid_tracker_dir.mkdir(parents=True, exist_ok=True)
        with open(sid_tracker_dir / f'pred_viterbi@{suffix}.pkl', 'wb') as f:
            pickle.dump(viterbi_pred_probas, f)
        evaluator.frame_level_metrics(trues, np.argmax(viterbi_pred_probas, axis=1), save_fp=sid_tracker_dir / f'result_viterbi@{suffix}.txt')
        evaluator.visualize_confusion_matrix(trues, np.argmax(viterbi_pred_probas, axis=1), steps, save_dir=sid_tracker_dir, suffix=f'viterbi@{suffix}')

        y_test_concatenated = np.concatenate((y_test_concatenated, trues))
        y_pred_concatenated = np.concatenate((y_pred_concatenated, np.argmax(viterbi_pred_probas, axis=1)))

    # aggregate results
    tracker_dir = task_dir / 'tracker' / args.model_hash
    if len(y_test_concatenated) > 0:        
        evaluator.frame_level_metrics(y_test_concatenated, y_pred_concatenated, save_fp=tracker_dir / 'loso' / f'result_viterbi@{suffix}.txt')
        evaluator.visualize_confusion_matrix(y_test_concatenated, y_pred_concatenated, steps, save_dir=tracker_dir / 'loso', suffix=f'viterbi@{suffix}')