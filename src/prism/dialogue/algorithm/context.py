from pathlib import Path
from openai import OpenAI
from groq import Groq

from ... import config

class ContextModel:

    def __init__(self, task_name, api_client_name='openai', llm_name='gpt-4o-mini'):
        """
        Args:
        * task_name (Str): a task name.
        * api_client_name (Str): API client name, currently only 'openai' and 'groq' are supported.
            default is 'openai'.
        * llm_name (Str): a language model name.
            default is 'gpt-4o-mini'.
        """
        self.task_name = task_name
        self.llm_name = llm_name
        if api_client_name == 'openai':
            self.client = OpenAI(max_retries=1, timeout=120)
        elif api_client_name == 'groq':
            self.client = Groq(max_retries=1, timeout=120)
        else:
            raise ValueError(f'Unknown API client name: {api_client_name}')

        fp = Path(__file__).parent / 'system_prompts' / 'default.txt'
        self.system_prompt = self._get_system_prompt(fp)
        print(f'Context Extraction Model initialized with {llm_name=}')
    
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
            fp = task_dir / 'dialogue' / 'context_example.txt'
            if not fp.exists():
                print(f'[Warning] Context example file {fp} does not exist. Skipping example replacement.')
            else:
                with open(fp, 'r') as f:
                    task_example = f.read()
                prompt_template = prompt_template.replace('[TASK_EXAMPLE]', task_example)
        
        return prompt_template.replace('[TASK_NAME]', self.task_name).replace('[TASK_DESCRIPTION]', task_description)

    def __call__(self, dialogue_history, verbose=False):
        """
        Args:
        * dialogue_history (List[Dict]): a list of dialogue history.
        * verbose (Bool): whether to print the user prompt and output.

        Returns:
        * context (Dict): a context dictionary.
            `context_type`: 'undetermined', 'current', 'has_done', 'not_yet'
            `step_index`: an integer representing the step index.
        """
        user_prompt = ''
        for item in dialogue_history:
            if item['role'] == 'user':
                user_prompt += f"User: {item['text']}\n"
            else:
                user_prompt += f"Assistant: {item['text']}\n"

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
            print(f'Output: {output["text"]}')
        return output