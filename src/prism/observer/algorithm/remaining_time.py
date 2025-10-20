import numpy as np
import random
import scipy

from .utils import monte_carlo_estimate


class RemainingTimeEstimator():

    def __init__(self, graph, n_bins_for_entropy=50, mc_samples=10000):
        self.graph = graph
        self.n_bins_for_entropy = n_bins_for_entropy
        self.mc_samples = mc_samples
        self.reset()

    def reset(self):
        """
        Reset the internal state.
        """
        self.expectations = {step: [0.0] for step in self.graph.steps}
        self.entropys = {step: [0.0] for step in self.graph.steps}
        self.time = 0

    def __get_latest__(self):
        """
        Return the latest expectation and entropy for each step.

        Returns:
        * expectations (Dict[Step, float]): a dictionary of the expected remaining time for each step.
        * entropys (Dict[Step, float]): a dictionary of the entropy of the expected remaining time for each step.
        """
        return {step: self.expectations[step][-1] for step in self.graph.steps}, {step: self.entropys[step][-1] for step in self.graph.steps}

    def forward(self, curr_entries):
        """
        Args:
        * curr_entries (List[ViterbiEntry]): a list of ViterbiEntry objects for each transition.

        Returns:
        * expectations (Dict[Step, float]): a dictionary of the expected remaining time for each step.
        * entropys (Dict[Step, float]): a dictionary of the entropy of the expected remaining time for each step.
        """
        self.time += 1
        for step in self.graph.steps:  # self.expectations[step][self.time] = 0.0
            self.expectations[step].append(0.0)
            self.entropys[step].append(0.0)
        
        total_prob = 0.0
        max_log_prob = max([entry.log_prob for entry in curr_entries if not np.isnan(entry.log_prob)], default=-np.inf)
        for entry in curr_entries:
            total_prob += np.nan_to_num(np.exp(entry.log_prob - max_log_prob))
        total_log_prob = np.log(total_prob)
        
        dt_samples = {step: [] for step in self.graph.steps}

        for entry in curr_entries:
            mc_expectations, mc_results_raw = monte_carlo_estimate(self.graph, self.graph.steps[entry.last_state.step_index], entry.last_state.time, n_samples=self.mc_samples)
            prob = np.nan_to_num(np.exp(entry.log_prob - max_log_prob - total_log_prob))
            for step, expected_time in mc_expectations.items():
                if step.index not in entry.step_index_order:
                    self.expectations[step][self.time] += prob * expected_time
            
            for step, samples in mc_results_raw.items():
                dt_samples[step] += random.sample(samples, max(0, int(prob * len(samples))))
        
        for step, samples in dt_samples.items():
            self.entropys[step][self.time] = scipy.stats.entropy(np.histogram(samples, bins=self.n_bins_for_entropy)[0])
        
        return self.__get_latest__()
