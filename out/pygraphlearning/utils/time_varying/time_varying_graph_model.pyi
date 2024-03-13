import numpy as np
from _typeshed import Incomplete

class TimevaryingGraph:
    N: Incomplete
    T: Incomplete
    W: Incomplete
    def __init__(self, N, T) -> None: ...

class TimevaryingErdosRenyiGraph(TimevaryingGraph):
    p_init: Incomplete
    q_change: Incomplete
    seed: Incomplete
    random_state: Incomplete
    W: Incomplete
    def __init__(self, N, T, p_init: float = 0.05, q_change: float = 0.05, seed: int = 42) -> None: ...
    def generate_graph_signals_eachtime(self, K: int, sigma: float = 0.5) -> np.ndarray: ...
