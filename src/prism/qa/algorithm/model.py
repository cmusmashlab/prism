from openai import OpenAI
from groq import Groq
from pathlib import Path

from ... import config


class QuestionAnsweringModel:

    def __init__(self, task_name, prompt_type, api_client_name='openai', llm_name='gpt-4o-mini', rag_context_file=None, use_context=True):
        """
        Args:
        * task_name (Str): task name (e.g., latte_making)
        * prompt_type (Str): prompt type, corresponding to the system prompt file name. (e.g., proposed)
        * api_client_name (Str): API client name, currently only 'openai' and 'groq' are supported.
            default is 'openai'.
        * llm_name (Str): language model name
            default is 'gpt-4o-mini'.     
        * rag_context_file (Str): a path to the RAG context file.
            Note: RAG has been removed from the current implementation to avoid complexity in installing `langchain`.
            If you want to use RAG (e.g., for a very long task manual), please contact the author.
        * use_context (bool): a flag to use context or not. If False, the model will not use the context.
            Set this to False if you want to use the model as a pure Q&A model without context as a baseline.
        """
        self.task_name = task_name
        self.llm_name = llm_name
        self.prompt_type = prompt_type
        self.rag_context_file = rag_context_file
        self.use_rag = rag_context_file is not None
        self.use_context = use_context
        if api_client_name == 'openai':
            self.client = OpenAI(max_retries=1, timeout=120)
        elif api_client_name == 'groq':
            self.client = Groq(max_retries=1, timeout=120)
        else:
            raise ValueError(f'Unknown API client name: {api_client_name}')

        fp = Path(__file__).parent / 'system_prompts' / f'{prompt_type}.txt'
        self.system_prompt = self._get_system_prompt(fp)
        print(f'Q&A Model initialized with {prompt_type=} {llm_name=} {use_context=}')

    def _get_system_prompt(self, prompt_template_fp):
        """
        Args:
        * prompt_template_fp (Str): a path to the prompt template file.

        Returns:
        * system_prompt (Str): a system prompt.
        """
        with open(prompt_template_fp, 'r') as f:
            prompt_template = f.read()

        task_dir = config.datadrive / 'tasks' / self.task_name
        with open(task_dir / 'dataset' / 'task_description.txt', 'r') as f:
            task_description = f.read()

        if '[TASK_EXAMPLE]' in prompt_template:
            fp = task_dir / 'qa' / 'qa_example.txt'
            if not fp.exists():
                print(f'[Warning] Context example file {fp} does not exist. Skipping example replacement.')
            else:
                with open(fp, 'r') as f:
                    task_example = f.read()
                prompt_template = prompt_template.replace('[TASK_EXAMPLE]', task_example)
        
        return prompt_template.replace('[TASK_NAME]', self.task_name).replace('[TASK_DESCRIPTION]', task_description)
    
    def __call__(self, user_query, dialogue_history=[], tracker_context=None, verbose=False):
        """
        Args:
        * user_query (Str): a query string.
        * dialogue_history (List[Dict]): a list of dialogue history.
        * tracker_context (Dict): a context dictionary from tracker.
            Keys are 'current_step_index', 'current_step_probs', 'history', and 'potential_next_step_indices'
        * verbose (bool): a flag to print the user prompt and output for debugging.

        Returns:
        * output (Dict): a dictionary containing the answer and other information.
            - 'text': the generated answer text
            - 'num_token_inp': number of input tokens
            - 'num_token_out': number of output tokens
            - 'num_token_total': total number of tokens
        """
        if self.use_rag:
            raise NotImplementedError

        user_prompt = ''
        for item in dialogue_history:
            if item['role'] == 'user':
                user_prompt += f"User: {item['text']}\n"
            else:
                user_prompt += f"Assistant: {item['text']}\n"
        user_prompt += f"User: {user_query}\n"
        user_prompt += f"Assistant: "

        if self.use_context and tracker_context is not None:
            user_prompt += "\n"
            user_prompt += f"Current Step: Step {tracker_context['current_step_index']}\n"
            history_string = ','.join([f'Step {s}' for s in tracker_context['history']])
            user_prompt += f"Step History So Far: {history_string}\n"
            next_steps_string = ','.join([f'Step {s}' for s in tracker_context['potential_next_step_indices']])
            user_prompt += f"Next Step Candidates: {next_steps_string}\n"

        if self.llm_name == 'gpt-5-mini':
            temperature = 1.0  # this is the only option for gpt-5-mini
        else:
            temperature = 1e-20
        response = self.client.chat.completions.create(
            model=self.llm_name,
            temperature=temperature,
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        output = {
            'text': response.choices[0].message.content,
            'num_token_inp': response.usage.prompt_tokens,
            'num_token_out': response.usage.completion_tokens,
            'num_token_total': response.usage.total_tokens,
        }

        if verbose:
            print(user_prompt)
            print(f"Output: {output['text']}")
        return output