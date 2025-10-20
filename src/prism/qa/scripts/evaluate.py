"""
This script generates answers to the question dataset and evaluate them.
requires
- datadrive / tasks / {task_name} / tracker / {model_hash} / lopo / {sid} / pred_viterbi.pkl
- datadrive / tasks / {task_name} / qa / questions.csv

after
- datadrive / tasks / {task_name} / qa / outputs_{llm_name}.csv
- datadrive / tasks / {task_name} / qa / eval_{pipeline_type}_{llm_name}.csv
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import ragas
from ragas.metrics import answer_correctness, answer_similarity
from datasets import Dataset

from prism import config
from prism.har import params
from prism.tracker.algorithm.utils import get_graph
from prism.qa.algorithm import QuestionAnsweringModel


def get_input_data(row, sid_preds_dict, graph):
    """
    Prepare input data for the question answering model based on the row from the DataFrame.
    Args:
    * row (pd.Series): A row from the DataFrame containing session_id, timestamp, question, etc.
    * sid_preds_dict (dict): A dictionary mapping session_id to predictions of the tracker.
    * graph (Graph): The graph object representing the task.
    Returns:
    * user_query (str): The question to be answered.
    * dialogue_history List[Dict]: The dialogue history up to the current step. In this case, it is empty.
    * tracker_context (Dict): a context dictionary from tracker.
            Keys are 'current_step_index', 'current_step_probs', 'history', and 'potential_next_step_indices'
    """
    user_query = row['question']
    dialogue_history = []  # In this case, we do not use dialogue history

    # get the tracker context
    # This should be the same as the `get_current_context` method in the TrackerAPI
    sid = str(row['session_id'])
    preds = sid_preds_dict[sid]
    timestamp = row['timestamp']
    time_index = int(timestamp / params.EXAMPLE_HOP_SECONDS)
    if time_index >= len(preds):
        print(f'Warning: {time_index} is equal to or greater than {len(preds)=} at {sid=}')
        time_index = len(preds) - 1
    pred = preds[time_index]
    step_index_history = np.argmax(preds, axis=1)[:time_index + 1].tolist()
    history = []
    prev_step_index = None
    for step_index in step_index_history:
        if prev_step_index is None:
            history.append(step_index)
        elif step_index != prev_step_index:
            history.append(step_index)
        prev_step_index = step_index

    tracker_context = {
        'current_step_index': np.argmax(pred) + 1, # +1 because step indices start from 1
        'current_step_probs': list(pred),
        'history': history,
        'potential_next_step_indices': [step.index for step in graph.get_potential_next_steps(graph.steps[int(np.argmax(pred) + 1)])]
    }
    return user_query, dialogue_history, tracker_context


def evaluate_with_ragas(questions, references, outputs, contexts=None):
    metrics = [answer_correctness, answer_similarity]
    if contexts is None:
        contexts = [['']] * len(questions)
    assert len(questions) == len(references) == len(outputs) == len(contexts), \
        f'{len(questions)=}, {len(references)=}, {len(outputs)=}, {len(contexts)=}'
    dataset = Dataset.from_dict(
        {'question': questions, 'answer': outputs, 'contexts': contexts, 'ground_truth': references}
    )
    return ragas.evaluate(dataset, metrics)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Task name', required=True)
    parser.add_argument('--model_hash', type=str, help='Model hash for loading and saving the result', required=True)
    parser.add_argument('--api_client_name', type=str, help='API client name for the LLM', default='openai')
    parser.add_argument('--llm', type=str, help='LLm name (`all` for all llms)', default='gpt-4o-mini')
    parser.add_argument('--eval_only', action='store_true', default=False)  # no output generation
    return parser.parse_args()


if __name__ == '__main__':
    print('===== Q&A Evaluating Script Started =====')
    args = get_args()
    task_dir = config.datadrive / 'tasks' / args.task
    qa_df = pd.read_csv(task_dir / 'qa' / 'questions.csv')
    graph = get_graph(args.task)

    # load viterbi predictions for each session
    sid_preds_dict = {}
    for pkl_fp in (task_dir / 'tracker' / args.model_hash / 'loso').glob('*/pred_viterbi@graph_type=train.pkl'):
        sid = str(pkl_fp).split('/')[-2]
        with open(pkl_fp, 'rb') as f:
            sid_preds_dict[sid] = pickle.load(f)

    # list combinations
    if get_args().llm == 'all':
        llm_names = ['gpt-3.5', 'gpt-4']  # If you want to use other LLMs like Llama-3, use Ollama.
    else:
        llm_names = [args.llm]
    print(f'Testing LLMs: ', llm_names)
    prompt_types = ['exp_proposed', 'exp_baseline']

    save_dir = task_dir / 'qa' / args.model_hash
    save_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        for llm_name in llm_names:
            outputs = {k: [] for k in prompt_types}
            tokens = {k: [] for k in prompt_types}
            pipelines = {k: QuestionAnsweringModel(args.task, k, args.api_client_name, llm_name, use_context= k != 'baseline') for k in prompt_types}
            pbar = tqdm(total=len(qa_df))
            for i, row in qa_df.iterrows():
                pbar.update(1)
                if str(row['session_id']) not in sid_preds_dict.keys():
                    # the session id in the Q&A dataset not in the tracker (e.g., dry run)
                    continue
                user_query, dialogue_history, tracker_context = get_input_data(row, sid_preds_dict, graph)
                for k, pipeline in pipelines.items():
                    output = pipeline(
                        user_query=user_query, 
                        dialogue_history=dialogue_history, 
                        tracker_context=tracker_context,
                        verbose=False
                    )
                    outputs[k].append(output['text'])
                    tokens[k].append(output['num_token_total'])

            for prompt_type in prompt_types:
                print(f'Tokens for {prompt_type}: {np.mean(tokens[prompt_type])} +- {np.std(tokens[prompt_type])}')
            output_df = pd.DataFrame.from_dict(outputs)
            output_df.to_csv(save_dir / f'outputs_{llm_name}.csv')

    # evaluation with ragas
    metrics = [answer_correctness, answer_similarity]
    for llm_name in llm_names:
        output_df = pd.read_csv(save_dir / f'outputs_{llm_name}.csv')
        for prompt_type in prompt_types:
            print(f'Evaluating {llm_name=} {prompt_type=}')
            eval = evaluate_with_ragas(
                questions=qa_df['question'],
                outputs=output_df[prompt_type],
                references=qa_df['reference']
            )  # TODO: add contexts
            print('answer correctness: ', np.mean(eval['answer_correctness']))
            print('answer similarity: ', np.mean(eval['answer_similarity']))
            eval.to_pandas().to_csv(save_dir / f'eval_{prompt_type}_{llm_name}.csv')