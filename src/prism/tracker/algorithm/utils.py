import itertools
from openai import OpenAI
from pathlib import Path
import pickle
from typing import List, Union, Dict

import numpy as np

from .collections import Graph, Step
from ... import config


def get_graph(task_name, train_sids=None):
    """
    Get Graph object for a task.
    Args:
    * task_name (Str): task name (e.g., latte_making)
    * train_sids (List[Str]): a list of session ids to be used for training. If None, all sessions are used.

    Returns:
    * graph (Graph): a Graph object.
    """
    task_dir = config.datadrive / 'tasks' / task_name
    preprocessed_files = [fp for fp in task_dir.glob('dataset/featurized/*.pkl') if 'inference-only' not in str(fp)]
    if train_sids is not None:
        preprocessed_files = [fp for fp in preprocessed_files if fp.stem in train_sids]
    if len(preprocessed_files) == 0:
        raise ValueError(f'No preprocessed files found in {task_dir / "dataset/featurized"}.')
    with open(task_dir / 'dataset/steps.txt', 'r') as f:
        steps = [s.strip() for s in f.readlines()]
    graph = _build_graph(preprocessed_files, steps)
    return graph


def get_raw_cm(cm_path: Union[str, Path]) -> List[List[float]]:
    """
    Get raw confusion matrix for a task.
    Args:
    * cm_path (Union[str, pathlib.Path]): path to the confusion matrix file (pickle format).

    Returns:
    * cm (List[List[float]]): a list of lists representing the confusion matrix of the task, obtaine through loso evaluation.
    """
    with open(cm_path, 'rb') as f:
        cm = pickle.load(f)
    print('Confusion matrix is loaded from ', cm_path)
    return cm


def _build_graph(pickle_files: List[Union[str, Path]], steps: List[str]) -> Graph:
    """
    This function builds a graph object from a set of pickle files and a list of steps.
    The graph represents transitions between the different steps in a procedure, with the time taken for each step.

    Args:
    * pickle_files (List[Union[str, pathlib.Path]]): a list of the paths to the pickle files containing the input data.
    * steps (List[str]): a list of strings representing the steps in the process.

    Returns:
    * graph (Graph): a graph object with a list of step objects and a dictionary containing transition probabilities.
    """
    steps = ['BEGIN'] + steps + ['END']
    transition_graph = np.zeros((len(steps), len(steps)))
    time_dict: Dict[str, List[int]] = {k: [] for k in steps}
    
    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as pickle_fp:
            pickle_data = pickle.load(pickle_fp)

        pickle_data['label'] = ['BEGIN'] + pickle_data['label'] + ['END']
        prev_step = None

        for curr_step, group in itertools.groupby(pickle_data['label']):
            if prev_step is not None:
                transition_graph[steps.index(prev_step)][steps.index(curr_step)] += 1

            time_dict[curr_step].append(len(list(group)))
            prev_step = curr_step

    step_list = []
    for i, k in enumerate(steps):
        step_list.append(Step(i, k, mean_time=np.nan_to_num(np.mean(time_dict[k])), std_time=np.nan_to_num(np.std(time_dict[k]))))

    edge_dict: Dict[Step, Dict[Step, float]] = {}
    for s in step_list:
        edge_dict[s] = {}
        total = np.sum(transition_graph[s.index])
        for next_step_index in np.nonzero(transition_graph[s.index])[0]:
            edge_dict[s][step_list[next_step_index]] = transition_graph[s.index][next_step_index] / total

    return Graph(steps=step_list, edges=edge_dict)


def get_proxy_graph(task_name, edges_fp, train_sids=None, llm_repetition=20, std_rate=0.1) -> Graph:
    """
    Get a proxy graph for a task by estimating the transition probabilities.
    
    Args:
    * task_name (str): The name of the task for which the proxy graph is to be generated.
    * edges_fp (pathlib.Path): The file path to the edges pickle file.
    * train_sids (List[str]): A list of session IDs to be used for training. If None, all sessions are used.
    * llm_repetition (int): The number of times to repeat the LLM call for generating the transition.
    * std_rate (float): The standard deviation to be used for the time estimation of each step.

    Returns:
    * Graph object: A Graph object representing the proxy graph.
    """
    graph = get_graph(task_name, train_sids=train_sids)  # Set mean_time and Step instances
    if edges_fp.exists():
        print(f'Loading edges from {edges_fp}...')
        with open(edges_fp, 'rb') as f:
            edges = pickle.load(f)  # Dict[Step, Dict[Step, float]]
    else:
        print(f'Generating edges with LLM...')
        transitions = []
        openai_client = OpenAI(max_retries=1, timeout=120)
        for i in range(llm_repetition):
            try:
                transition = _generate_transition(openai_client, task_name, graph.steps, llm_name='gpt-4o')
                print(transition)
                if len(transition) != len(graph.steps) - 1:  # Exclude BEGIN
                    print(f'[warning] Transition length {len(transition)} does not match expected length {len(graph.steps) - 2}.')
                transitions.append(transition)
            except Exception as e:
                print(e)
                continue
        edges = {step: {} for step in transitions[0]}  # Assuming all steps are covered in the gerenerated transitions
        for step in edges.keys():
            for transition in transitions:
                if step not in transition:
                    continue
                step_index = transition.index(step)
                if step_index == len(transition) - 1:
                    continue
                next_step = transition[step_index + 1]
                if next_step not in edges[step]:
                    edges[step][next_step] = 0
                edges[step][next_step] += 1 / llm_repetition
        edges_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(edges_fp, 'wb') as f:
            pickle.dump(edges, f)
        print(f'Edges file {edges_fp} has been generated.')
    
    for step, next_steps in edges.items():
        graph.edges[step] = next_steps
    for i in range(len(graph.steps)):
        graph.steps[i].std_time = graph.steps[i].mean_time * std_rate
    
    return graph

def _generate_transition(openai_client, task_name, step_list, llm_name = 'gpt-4o') -> List[Step]:
    """
    Generate a transition using LLM.
    This is a placeholder function and should be replaced with actual LLM call.

    Args:
    * openai_client (OpenAI): An instance of OpenAI client to interact with the LLM.
    * task_name (str): The name of the task for which the transition is to be generated.
    * step_list (List[Step]): A list of Step objects representing the steps in the task.
    * llm_name (str): The name of the LLM model to be used for generating the transition.

    Returns:
    * List[Step]: A list of steps representing the transition.
    """

    prompt = """
You are an expert in task planning and execution.
Given the following task description and its associated steps, generate a valid permutation of step indices to complete the task.

Constraints:
- You must use every step exactly once, without omission or repetition.
- You are encouraged to change the order reasonably (imagine a various user behavior), but must not skip or duplicate any step.
- Your answer must be a single line of comma-separated integers, with no explanation, no extra text, and no brackets or quotes.
- The last step must be the END step. Use the index of the END step as the last number in your output.

Format: 1,2,3,4,5 (← just like this, as a single line with numbers only)
Output Example: [OUTPUT_EXAMPLE]

Input:
Task Description: [TASK_DESCRIPTION]

Output:
    """
    task_dir = config.datadrive / 'tasks' / task_name
    with open(task_dir / 'dataset' / 'task_description.txt', 'r') as f:
        task_description = f.read()
        task_description += f'\n\nStep {len(step_list) - 1}: END'
    prompt = prompt.replace('[TASK_DESCRIPTION]', task_description)
    output_example = ','.join([str(step.index) for step in step_list if step.index != 0])  # Exclude BEGIN
    prompt = prompt.replace('[OUTPUT_EXAMPLE]', output_example)

    response = openai_client.chat.completions.create(
        model=llm_name,
        messages=[
            {'role': 'user', 'content': prompt},
        ],
    )
    try:
        step_indices = response.choices[0].message.content.strip().split(',')
        steps = [step_list[int(i)] for i in step_indices]
        return steps
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {response.choices[0].message.content}. Error: {e}")