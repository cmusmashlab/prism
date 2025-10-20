import json
import time
from typing import Optional

from .algorithm import ContextModel
from ..qa.algorithm import QuestionAnsweringModel


class DialogueAPI():
    """
    API for PrISM-Dialogue, mainly for the streaming purpose.
    """

    def __init__(self, task_name, api_client_name='openai', llm_name='gpt-4o-mini'):
        """
        Args:
        * task_name (Str): a task name.
        * api_client_name (Str): API client name, currently only 'openai' and 'groq' are supported.
            default is 'openai'.
        * llm_name (Str): a language model name.
            default is 'gpt-4o-mini'.
        """
        self.qa_model = QuestionAnsweringModel(task_name=task_name, prompt_type='dialogue', api_client_name=api_client_name, llm_name=llm_name)
        self.context_model = ContextModel(task_name=task_name, api_client_name=api_client_name, llm_name=llm_name)
        self.dialogue_history = [] # List[Dict]: [{'role': 'user', 'text': 'text', 'time': 1234567890.123}, ...]
        self.recent_exchange_range_sec = 30  # TODO: find the optimal logic for this.
        self.new_exchange_start_time = None  

    def _get_recent_dialogue_history(self):
        recent_history = []
        curr_time = time.time()
        for item in self.dialogue_history:
            if self.new_exchange_start_time is None:
                if item['time'] > curr_time - self.recent_exchange_range_sec:
                    recent_history.append(item)
            else:
                if item['time'] > self.new_exchange_start_time:
                    recent_history.append(item)
        return recent_history

    def __call__(self, user_query, tracker_context=None) -> Optional[str]:
        """
        Args:
        * user_query (Str): a query string.
        * context (Dict): a context dictionary from tracker.
            Keys are 'current_step_index', 'current_step_probs', 'history', and 'potential_next_step_indices'

        Returns:
        * response (Optional[str]): a response string from the assistant.
        """
        recent_dialogue_history = self._get_recent_dialogue_history()
        curr_time = time.time()
        self.dialogue_history.append({'role': 'user', 'text': user_query, 'time': curr_time})
        output = self.qa_model(user_query, recent_dialogue_history, tracker_context)
        response = output['text']
        if "no response" in response.lower():
            response = None
        else:
            self.dialogue_history.append({'role': 'assistant', 'text': response, 'time': time.time()})
        return response

    def predict_context_from_dialogue(self) -> dict:
        """
        Returns:
        * context (Dict): a context dictionary.
            `context_type`: 'undetermined', 'current', 'has_done', 'not_yet'
            `step_index`: an integer representing the step index.
        """
        recent_dialogue_history = self._get_recent_dialogue_history()
        if len(recent_dialogue_history) == 0:
            print('[warning] No recent dialogue history found. Returning undetermined context.')
            return {
                'context_type': 'undetermined',
                'step_index': -1
            }

        output = self.context_model(recent_dialogue_history, verbose=False)
        try:
            context = json.loads(output['text'])['context']
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Response string: {output['text']}")
            context = {
                'context_type': 'undetermined',
                'step_index': -1
            }
        return context