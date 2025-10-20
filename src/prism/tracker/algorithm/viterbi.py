import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .collections import Graph, HiddenState, HiddenTransition, ViterbiEntry


class ViterbiTracker:


    def __init__(self,
                 graph: Graph,
                 confusion_matrix: List[List[float]],
                 max_time_to_consider: int = 500):
        """
        Args:
        * graph (Graph): a graph object, which represents transitions between the different steps in a procedure.
        * confusion_matrix (List[List[float]]): a list of lists representing the confusion matrix of the observation probabilities.
        * max_time_to_consider (int): maximum time to consider for the transition probabilities.
        """
        self.graph = graph
        self.steps = sorted(graph.steps, key=lambda step: step.index)
        self.original_cm = confusion_matrix
        self.max_time_to_consider = max_time_to_consider

        self.max_size_of_current_entries = 50  # latency exponential to this value 
        self.curr_entries: List[ViterbiEntry] = []

        # initialize escape probability matrix and current entries
        self.escape_prob_matrix = self.__initialize_escape_prob_matrix__()
        self.curr_time = 0
        self.curr_entries.append(ViterbiEntry(np.log(1.0), history=[HiddenState(0, self.curr_time)]))  # start with BEGIN step

        print('Tracker initialized.')

    def __initialize_escape_prob_matrix__(self) -> Dict[HiddenState, float]:
        """ This method initializes the escape probability matrix, which represents the probability of staying on the step at the time.   
        """
        escape_prob_matrix: Dict[HiddenState, float] = {}
        for step in self.steps:
            # 3 sigma can cover almost 100% of the transition
            max_time = max(self.max_time_to_consider, int(step.mean_time + step.std_time * 3))
            # cdf represents the (reversed) probability of staying on the step at the time
            probs = 1 - stats.norm.cdf(range(max_time), loc=step.mean_time, scale=step.std_time)
            for time in range(max_time):
                if step.index == 0:
                    prob = 1.0 - sys.float_info.epsilon
                elif step.index == len(self.steps) - 1:
                    prob = sys.float_info.epsilon
                else:
                    if time == max_time - 1:
                        prob = 1.0 - sys.float_info.epsilon
                    else:
                        prob = 1.0 - probs[time + 1] / probs[time]
                escape_prob_matrix[HiddenState(step.index, time)] = prob

        return escape_prob_matrix

    def __get_possible_transitions__(self, curr_entry: ViterbiEntry) -> List[HiddenTransition]:
        """
        This method returns the possible transitions from the current step based on the history context.
        That is, it excludes the transitions to the steps that have already been done in the history.
        Args:
        * curr_entry (ViterbiEntry): a ViterbiEntry object.

        Returns:
        * transitions (List[HiddenTransition]): a list of HiddenTransition objects.
        """
        if curr_entry.last_state is None:
            return []
        else:
            curr_state = curr_entry.last_state
            if curr_state in self.escape_prob_matrix:
                escape_prob = self.escape_prob_matrix[curr_state]
            else:
                print(f'[warning] No escape probability found for {curr_state}. Using default value of 1.0 - sys.float_info.epsilon.')
                escape_prob = 1.0 - sys.float_info.epsilon  # default value
            
            curr_step = self.steps[curr_state.step_index]
            possible_edges_from_curr_step = {}
            for step in self.steps:
                prob = self.graph.edges[curr_step].get(step, 0.0)
                if step not in curr_entry.step_index_order and prob > 0:
                    possible_edges_from_curr_step[step] = prob
            
            transitions: List[HiddenTransition] = []
            transitions.append(HiddenTransition(curr_step.index, np.log(1 - escape_prob)))  # stay on the step
            for dest_step, dest_prob in possible_edges_from_curr_step.items():  # transition to other steps
                transitions.append(HiddenTransition(dest_step.index, np.log(escape_prob * dest_prob)))

            return transitions


    def __get_best_entries_for_each_step__(self, entries: Iterable[ViterbiEntry]) -> List[ViterbiEntry]:
        """
        This method selects the best entry for each step from a list of ViterbiEntry.

        Args:
        * entries (Iterable[ViterbiEntry]): a list of ViterbiEntry to choose from.

        Returns:
        * best_entries (List[ViterbiEntry]): a list of ViterbiEntry with the highest log probability for each step.
        """
        entries = sorted(entries, key=lambda entry: entry.log_prob, reverse=True)
        best_entries = []
        seen_steps = set()
        for entry in entries:
            if entry.last_state is None or entry.last_state.step_index == 0 or entry.last_state.step_index == len(self.steps) - 1:
                continue
            if entry.last_state.step_index not in seen_steps:
                seen_steps.add(entry.last_state.step_index)
                best_entries.append(entry)
        assert len(best_entries) <= len(self.steps), "More best entries than steps. This should not happen."
        return best_entries
    

    def __get_probs__(self, entries: Iterable[ViterbiEntry]) -> List[float]:
        """
        This method returns the probabilities of each step based on a list of ViterbiEntry.
        
        Args:
        * entries (Iterable[ViterbiEntry]): a list of ViterbiEntry.

        Returns:
        * probs (List[float]): a list of floats representing the current probabilities of each step.
        """
        probs = [0.0] * (len(self.steps) - 2)  # exclude BEGIN and END
        # best_entries_for_each_step
        best_entries_for_each_step = self.__get_best_entries_for_each_step__(entries)

        max_log_prob = max([entry.log_prob for entry in best_entries_for_each_step if not np.isnan(entry.log_prob)], default=-np.inf)

        for entry in best_entries_for_each_step:
            if entry.last_state is not None:
                if entry.last_state.step_index == 0 or entry.last_state.step_index == len(self.steps) - 1:
                    continue
                probs[entry.last_state.step_index - 1] += np.nan_to_num(np.exp(entry.log_prob - max_log_prob))
        probs /= np.sum(probs)
        return probs


    def set_if_a_step_is_done(self, step_index: int, is_done: bool) -> None:
        """
        This method applies a penalty to the current entries if a step is done or not done.
        
        Args:
        * step_index (int): an integer representing the step index.
        * is_done (bool): a boolean value indicating whether the step is done.
        """
        penalty = sorted(self.curr_entries, key=lambda entry: entry.log_prob, reverse=True)[0].log_prob
        penalized_entries = []
        for entry in self.curr_entries:
            previous_order = entry.step_index_order
            if (step_index in previous_order) != is_done:
                entry.log_prob += penalty
            penalized_entries.append(entry)
        self.curr_entries = penalized_entries


    def set_current_step(self, step_index: int) -> None:
        """
        This method sets the current step of the tracker.

        Args:
        * step_index (int): an integer representing the step index.
        """
        penalty = sorted(self.curr_entries, key=lambda entry: entry.log_prob, reverse=True)[0].log_prob
        penalized_entries = []
        for entry in self.curr_entries:
            if entry.last_state is not None and entry.last_state.step_index != step_index:
                entry.log_prob += penalty
            penalized_entries.append(entry)
        self.curr_entries = penalized_entries


    def forward(self, observation: List[float]) -> List[float]:
        """
        This method calculates the Viterbi forward algorithm for a single frame given the current prediction entries.

        Args:
        * observation (List[float]): a list of the observation probabilities of each step at the current frame.

        Returns:
        * probs (List[float]): a list of floats representing the current probabilities of each step.
        """
        assert len(self.curr_entries) > 0, 'No current entries to forward. Maybe the tracker is not initilized.'
        assert len(observation) == len(self.steps) - 2, f'Observation length {len(observation)} does not match the number of steps {len(self.steps) - 2}.'

        # calibrate frame-based observation probabilities with a known confusion matrix
        observed_log_probs: Dict[int, float] = {}  # dict[step_index, log_prob]
        for actual_step in self.steps: 
            if actual_step.index == 0 or actual_step.index == len(self.steps) - 1:  # BEGIN or END
                observed_log_probs[actual_step.index] = np.log(sys.float_info.epsilon)
                continue
            total_prob = sys.float_info.epsilon  # avoid -np.inf
            for observed_step, cm_prob in zip(self.steps[1: -1], self.original_cm[actual_step.index - 1]):  # exclude BEGIN and END
                total_prob += cm_prob * observation[observed_step.index - 1]  # exclude BEGIN
            observed_log_probs[actual_step.index] = np.log(total_prob)

        # forward with the observation probabilities
        next_entries: List[ViterbiEntry] = []
        for curr_entry in self.curr_entries:
            curr_step_index = curr_entry.last_state.step_index
            if curr_entry.log_prob == -np.inf or np.isnan(curr_entry.log_prob):  # discard invalid entries
                continue
            possible_transitions = self.__get_possible_transitions__(curr_entry)  # List[HiddenTransition], length = max_time

            for transition in possible_transitions:  # stay or transit hypotheses
                log_prob = curr_entry.log_prob + transition.log_prob + observed_log_probs[transition.next_step_index]

                if curr_step_index == transition.next_step_index:  # stay on the step
                    next_state = HiddenState(transition.next_step_index, curr_entry.last_state.time + 1)
                else:
                    next_state = HiddenState(transition.next_step_index, 0)

                next_entries.append(ViterbiEntry(log_prob, curr_entry.history + [next_state]))

        # select top self.max_size_of_current_entries entries
        next_entries = sorted(next_entries, key=lambda entry: entry.log_prob, reverse=True)
        # keep the entry with the best probability among those with the same step transition order
        next_entries_without_duplicates, seen_order = [], []
        for entry in next_entries:
            order = entry.step_index_order
            if order not in seen_order:
                seen_order.append(order)
                next_entries_without_duplicates.append(entry)

        if len(next_entries_without_duplicates) > self.max_size_of_current_entries:
            next_entries_without_duplicates = next_entries_without_duplicates[:self.max_size_of_current_entries]

        self.curr_entries = next_entries_without_duplicates
        self.curr_time += 1
        return self.__get_probs__(self.curr_entries)


    def predict_batch(self, observations: List[List[float]], oracle: Optional[Dict[int, int]] = None) -> Iterator[List[float]]:
        """
        This function is used for predicting the steps of a procedure using the complete observation data.

        Args:
        * observations (List[List[float]]): a numpy array containing the observation probabilities (frames x steps).
        * oracle (Optional[Dict[int, int]]): an optional dictionary where the keys are time and the values are the given step index.

        For each time frame, returns:
        * probs (List[float]): a list of floats representing the current probabilities of each step.
        """
        oracle = {} if oracle is None else oracle

        # dp: basic viterbi algorithm
        for i, observation in enumerate(observations):
            yield self.forward(observation)
            if self.curr_time in oracle:
                self.set_current_step(oracle[self.curr_time])