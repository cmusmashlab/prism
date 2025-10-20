from typing import Dict, List, Optional


class Step:
    def __init__(self, index: int, description: str, mean_time: float, std_time: float):
        self.index = index
        self.description = description
        self.mean_time = mean_time
        self.std_time = std_time

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Step):
            # don't attempt to compare against unrelated types
            return False
        return self.index == __value.index
    
    def __hash__(self) -> int:
        return id(self.index)
    
    def __repr__(self):
        return f's{self.index}--{self.description}'


class Graph:
    def __init__(self, steps: List[Step], edges: Dict[Step, Dict[Step, float]]):
        self.steps = steps  # 0 is BEGIN, -1 is END
        self.edges = edges
        self.start = self.steps[0]
        self.end = self.steps[-1]

    def __repr__(self) -> str:
        text = f'Graph: {len(self.steps)} steps, {len(self.edges)} edges, start={self.start}, end={self.end}\n'
        for step in self.steps:
            text += f'{step} ({round(step.mean_time, 1)} +- {round(step.std_time, 1)}) -> {dict(map(lambda x: (x[0], round(x[1], 1)), self.edges[step].items()))}\n'
        return text

    def get_potential_next_steps(self, step: Step) -> List[Step]:
        return list(self.edges[step].keys())


class HiddenState:
    def __init__(self, step_index: int, time: int):
        self.step_index = step_index
        self.time = time

    def __repr__(self):
        return f's{self.step_index}_{self.time}'
    
    def __hash__(self) -> int:
        return hash((self.step_index, self.time))
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, HiddenState):
            return False
        return self.step_index == __value.step_index and self.time == __value.time


class HiddenTransition:
    def __init__(self, next_step_index: int, log_prob: float):
        self.next_step_index = next_step_index
        self.log_prob = log_prob

    def __repr__(self):
        return f'{self.next_step_index=}@{self.log_prob}'


class ViterbiEntry:
    def __init__(self, log_prob: float, history: List[HiddenState]):
        self.log_prob = log_prob
        self.history = history

    def __repr__(self):
        text = ''
        prev_state = None
        for state in self.history:
            if prev_state is None:
                prev_state = state
            if state.step_index != prev_state.step_index:
                text += f'{prev_state}->'
            prev_state = state
        text += f'{state}'
        return f'{text}@{self.log_prob}'

    @property
    def last_state(self) -> Optional[HiddenState]:
        if len(self.history) == 0:
            return None
        return self.history[-1]
    
    @property
    def step_index_order(self) -> List[int]:
        indices = [state.step_index for state in self.history]
        # remove duplicates while preserving order
        seen = set()
        return [x for x in indices if not (x in seen or seen.add(x))]