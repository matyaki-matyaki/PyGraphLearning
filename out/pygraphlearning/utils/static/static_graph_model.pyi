import numpy as np
from _typeshed import Incomplete

class StaticGraph:
    N: Incomplete
    W: Incomplete
    def __init__(self, N: int) -> None: ...

class StaticErdosRenyiGraph(StaticGraph):
    p: Incomplete
    seed: Incomplete
    random_state: Incomplete
    W: Incomplete
    def __init__(self, N: int, p: float = 0.05, seed: int = 42) -> None: ...
    def generate_graph_signals(self, K: int, sigma: float = 0.5) -> np.ndarray: ...
