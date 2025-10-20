import numpy as np

from .algorithm import ViterbiTracker
from .algorithm.utils import get_graph, get_raw_cm

from prism import config


class TrackerAPI():
    """
    API for PrISM-Tracker, mainly for the streaming purpose.
    """

    def __init__(self, task_name, model_hash, max_confirmation_per_session=3, confirmation_threshold=0.4):
        """
        Args:
        * task_name (Str): task name (e.g., latte_making)
        * model_hash (Str): a model hash for loading the model.
        * max_confirmation_per_session (Int): maximum number of confirmations to send per session.
            default = 3
        * confirmation_threshold (Float): threshold for sending confirmation.
            default = 0.4
        """
        graph = get_graph(task_name)  # Using all training data to build the graph
        cm = get_raw_cm(cm_path=config.datadrive / 'tasks' / task_name / 'har' / model_hash / 'loso' / 'cm_raw.pkl')

        self.tracker = ViterbiTracker(graph, confusion_matrix=cm)
        self.max_confirmation_per_session = max_confirmation_per_session
        self.confirmation_threshold = confirmation_threshold
        self.confirmation_count = 0

    def __call__(self, probs):
        """
        This method calculates the Viterbi forward algorithm for a single frame given the current prediction entries.

        Args:
        * probs (List[float]): a list of the observation probabilities of each step at the current frame.

        Returns:
        * probs (List[float]): a list of floats representing the current probabilities of each step.
        """
        return self.tracker.forward(probs)
    
    def set_current_context(self, context):
        """
        This method sets the current context of the tracker, which is obtained by DialogueAPI.

        Args:
        * context (Dict): a context dictionary.
            `context_type`: ('undetermined'), 'current', 'has_done', 'not_yet'
            `step_index`: an integer representing the step index.
        """
        if 'step_index' not in context:
            raise ValueError("Context must contain 'step_index' key.")
        step_index = context['step_index']
        print(sorted(self.tracker.curr_entries, key=lambda x: x.log_prob, reverse=True)[:5])  # Debugging: print first 5 entries
        if context['context_type'] == 'current':
            self.tracker.set_current_step(step_index)
        elif context['context_type'] == 'has_done':
            self.tracker.set_if_a_step_is_done(step_index, True)
        elif context['context_type'] == 'not_yet':
            self.tracker.set_if_a_step_is_done(step_index, False)
        else:
            raise NotImplementedError(f'Unknown context type: {context["context_type"]}')
        print(sorted(self.tracker.curr_entries, key=lambda x: x.log_prob, reverse=True)[:5])  # Debugging: print first 5 entries

    def get_current_context(self):
        """
        This method returns the current context of the tracker to provide it to Q&A module.

        Returns:
        * context (Dict): a dictionary containing the current step index, the probabilities of each step, and the history of steps.
            Keys are 'current_step_index', 'current_step_probs', 'history', and 'potential_next_step_indices'.
        """
        probs = self.tracker.__get_probs__(self.tracker.curr_entries)
        probs = [round(x, 4) for x in list(np.nan_to_num(probs))]

        entries = sorted(self.tracker.curr_entries, key=lambda entry: entry.log_prob, reverse=True)
        step_index_order = entries[0].step_index_order

        ret = {
            'current_step_index': int(np.argmax(probs) + 1),
            'current_step_probs': probs,
            'history': step_index_order,
            'potential_next_step_indices': [step.index for step in self.tracker.graph.get_potential_next_steps(self.tracker.steps[int(np.argmax(probs) + 1)])]
        }
        return ret

    def should_send_confirmation(self, probs):
        """
        This method checks if the tracker should send a confirmation message to the user.
        Args:
        * probs (List[float]): a list of the observation probabilities of each step at the current frame.

        Returns:
        * should_send (Bool): whether to send a confirmation message.
        """
        # TODO: consider logic using hypotheses
        if self.confirmation_count >= self.max_confirmation_per_session:
            return False
        else:
            max_prob = max(probs)
            if max_prob < self.confirmation_threshold:
                self.confirmation_count += 1
                return True
            else:
                return False
