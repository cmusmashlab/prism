"""
This script evaluates the observer policy.
requires
- datadrive / tasks / {task_name} / observer / {model_hash} / loso / {sid} / remaining_time_distribution.pkl

after
- datadrive / tasks / {task_name} / observer / {model_hash} / loso / {sid} / step_time_dict.pkl
- datadrive / tasks / {task_name} / observer / {model_hash} / loso / {sid} / step_threshold_dict.pkl
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
from tqdm import tqdm

from prism import config
from prism.har import params as har_params
from prism.tracker.algorithm.utils import get_graph
from prism.observer.algorithm import InterventionPolicy, BaselinePolicy


def get_gt_time(ground_truth, target_step):
    for step, gt in ground_truth.items():
        if step.index == target_step.index:
            return gt
    return 0


def find_best_h_threshold(training_data, target_step, threshold_range=np.arange(0.5, 4.0, 0.01)):
    """
    This function finds the best threshold for the h value in the policy.
    
    Args:
    * training_data (List[Dict[str, Dict[Step, List[float]]]): a list of dictionaries containing the expectations and entropys for each step in the procedure.
    * target_step (Step): the target step for the policy.
    * threshold_range (np.ndarray): a range of threshold values to search for the best threshold.

    Returns:
    * best_h_threshold (float): the best threshold for the h value in the policy.
    """
    best_h_threshold = 0
    best_error = 1e9

    for i, h_threshold in enumerate(threshold_range):
        errors = []
        policy = InterventionPolicy(target_step, h_threshold)
        for dt_distribution in training_data:
            expectations = dt_distribution['expectations']
            entropys = dt_distribution['entropys']
            gt_time = get_gt_time(dt_distribution['ground_truth'], target_step)
            triggered_time = policy.predict_batch(expectations, entropys)
            errors.append(abs(triggered_time - gt_time))
        
        if np.mean(np.abs(errors)) < best_error:
            best_error = np.mean(np.abs(errors))
            best_h_threshold = h_threshold
    return best_h_threshold

def visualize_results(step_loo_result, task_name, save_path):
    width = 10
    fig, ax = plt.subplots(figsize=(8, 3))
    frame_to_sec = har_params.EXAMPLE_HOP_SECONDS
    proposed_list, baseline_list = [], []
    for i, step_id in enumerate(step_loo_result.keys()):
        x = i * width * 3
        proposed = np.abs(step_loo_result[step_id]['proposed']) * frame_to_sec
        baseline = np.abs(step_loo_result[step_id]['baseline']) * frame_to_sec
        proposed_list.append(proposed)
        baseline_list.append(baseline)
        rects1 = ax.bar(x + width/2, np.mean(proposed),
                        width, yerr=scipy.stats.sem(proposed), label='Proposed', color='SkyBlue', capsize=5)
        rects2 = ax.bar(x - width/2, np.mean(baseline),
                        width, yerr=scipy.stats.sem(baseline), label='Baseline', color='IndianRed', capsize=5)

        # Add significance stars if p < 0.05
        p_value = scipy.stats.ttest_rel(proposed, baseline)[1]
        h = max(np.mean(proposed), np.mean(baseline)) + max(scipy.stats.sem(proposed), scipy.stats.sem(baseline))
        if p_value < 0.05:
            ax.text(x-3, h, '*', ha='center', va='bottom', fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=12)
    ax.set_title(task_name, fontsize=14)
    ax.set_xticks([i * width * 3 for i in range(len(step_loo_result))],[f's{s}' for s in sorted(step_loo_result.keys())])
    ax.set_ylabel('Error [sec]', fontsize=12)
    ax.set_xlabel('Target step', fontsize=12)
    ax.set_ylim([0, 350])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print('Proposed: ', round(np.mean(proposed_list), 1))
    print('Baseline: ', round(np.mean(baseline_list), 1))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Name of the task', required=True)
    parser.add_argument('--model_hash', type=str, help='Model hash for loading and saving the result', required=True)
    parser.add_argument('--test_sids', type=str, help='To specify test session sids (e.g., --test_sids 10,11,14)', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    print('===== Observer Evaluating Policy Script Started =====')
    args = get_args()
    task_dir = config.datadrive / 'tasks' / args.task
    graph = get_graph(args.task)
    if args.test_sids is None:
        test_sids = [fp.stem for fp in task_dir.glob('dataset/featurized/*.pkl') if 'inference-only' not in str(fp)]
    else:
        test_sids = args.test_sids.split(',')
    print('Test Session IDs:', test_sids)
    working_dir = task_dir / 'observer' / args.model_hash

    # grab all sids
    all_sid_distribution = {}
    for sid_model_dir in working_dir.glob('loso/*'):
        if not sid_model_dir.is_dir():
            continue
        sid = sid_model_dir.stem
        with open(sid_model_dir / 'remaining_time_distribution.pkl', 'rb') as f:
            all_sid_distribution[sid] = pickle.load(f)
    print('All Session IDs:', all_sid_distribution.keys())

    for test_sid in test_sids:
        print(f'-----Evaluating observer policy for {test_sid}-----')
        training_distributions = [dist for sid, dist in all_sid_distribution.items() if sid != test_sid]
        test_distribution = all_sid_distribution[test_sid]
        step_time_dict, step_threshold_dict = {}, {}
        for target_step in tqdm(graph.steps[1:-1]):
            # baseline
            baseline_policy = BaselinePolicy(target_step)
            baseline_time = baseline_policy.predict_batch(test_distribution['expectations'])
            # proposed
            best_h_threshold = find_best_h_threshold(training_distributions, target_step)
            policy = InterventionPolicy(target_step, best_h_threshold)
            policy_time = policy.predict_batch(test_distribution['expectations'], test_distribution['entropys'])
            # gt
            gt_time = get_gt_time(test_distribution['ground_truth'], target_step)
            step_time_dict[target_step] = {'ground_truth': gt_time, 'proposed': policy_time, 'baseline': baseline_time}
            step_threshold_dict[target_step] = best_h_threshold
        with open(working_dir / 'loso' / test_sid / 'step_time_dict.pkl', 'wb') as f:
            pickle.dump(step_time_dict, f)
        with open(working_dir / 'loso' / test_sid / 'step_threshold_dict.pkl', 'wb') as f:
            pickle.dump(step_threshold_dict, f)

    # Visualize the results
    step_loo_result = {}
    for sid in test_sids:
        with open(working_dir / 'loso' / sid / 'step_time_dict.pkl', 'rb') as f:
            result = pickle.load(f)
        for step, dic in result.items():
            if step.index not in step_loo_result.keys():
                step_loo_result[step.index] = {'proposed': [], 'baseline': []}
            step_loo_result[step.index]['proposed'].append(dic['ground_truth'] - dic['proposed'])
            step_loo_result[step.index]['baseline'].append(dic['ground_truth'] - dic['baseline'])
    visualize_results(step_loo_result, args.task, working_dir / 'step_loo_result.png')