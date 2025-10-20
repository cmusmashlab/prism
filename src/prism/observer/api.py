from .algorithm import RemainingTimeEstimator, InterventionPolicy
from ..tracker.algorithm.utils import get_graph

class ObserverAPI():
    """
    API for PrISM-Observer, mainly for the streaming purpose.
    """

    def __init__(self, task_name, policy_config, n_bins_for_entropy=50, mc_samples=1000):
        """
        Args:
        * task_name (Str): task name (e.g., latte_making)
        * policy_config (Dict[int, Dict[str: Any]]): a dictionary of target steps and their configurations.
            index 0 -> BEGIN
            Note: It is recommended to specify not many (maximum 3) steps for computation time and accuracy purposes.
        * n_bins_for_entropy (int): the number of bins for the entropy calculation.
        * mc_samples (int): the number of Monte Carlo samples for the expectation calculation.
        """
        graph = get_graph(task_name)
        steps = graph.steps

        # initialize
        self.remaining_time_estimator = RemainingTimeEstimator(graph, n_bins_for_entropy=n_bins_for_entropy, mc_samples=mc_samples)
        self.policies = {}
        for target_step_index, config in policy_config.items():
            target_step = steps[target_step_index]
            self.policies[target_step] = InterventionPolicy(target_step, config['h_threshold'], offset=config['offset'])
        self.time = 0
        self.timers = {step: None for step in self.policies.keys()}

    def __call__(self, curr_entries) -> list:
        """
        Args:
        * curr_entries (List[ViterbiEntry]): a list of ViterbiEntry objects for each transition.

        Returns:
        * triggered_step (Optional[Step]): a triggered step. None if no step is triggered.
        """
        self.time += 1
        if len(self.policies) == 0 or len(self.timers) == 0:
            # already finished
            return None

        expectations, entropys = self.remaining_time_estimator.forward(curr_entries)
        for target_step in self.policies.keys():
            e, h = expectations[target_step], entropys[target_step]
            status, remaining_time = self.policies[target_step].forward(e, h)
            if status == 'timer_start':
                self.timers[target_step] = self.time + remaining_time
            elif status == 'timer_stop':
                self.timers[target_step] = None
            else:
                pass
        
        for step, timer in self.timers.items():
            if timer is not None and self.time >= timer:  # trigger
                del self.timers[step], self.policies[step]
                return step
        return None
