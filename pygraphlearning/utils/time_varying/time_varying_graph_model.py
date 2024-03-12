import numpy as np
import networkx as nx


class TimevaryingGraph:
    """
    Class for a time-varying graph

    Attributes
    ----------
    N: int
        Number of vertices
    T: int
        Number of observation times
    """
    def __init__(self, N, T):
        self.N = N
        self.T = T
        self.W = np.empty((T, N, N))


class TimevaryingErdosRenyiGraph(TimevaryingGraph):
    """
    Class for a time-varying Erdos-Renyi graph

    Attributes
    ----------
    N: int
        Number of vertices
    T: int
        Number of observation times
    p_init: float
        Probability of an edge being placed between each pair of vertices
    q_change: float
        Probability of an edge being newly created
    seed: int
        Random seed value
    random_state: np.random.RandomState
        RandomState
    W: np.ndarray
        A series of T adjacency matrices, each of size (N, N)
    """
    def __init__(self, N, T, p_init=0.05, q_change=0.05, seed=42):
        super().__init__(N, T)
        self.p_init = p_init
        self.q_change = q_change
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.W = self._simulate()

    def _simulate(self) -> np.ndarray:
        """
        Generates a series of weighted adjacency matrices (T, N, N)
        following the time-varying Erdos-Renyi graph model

        Returns
        -------
        np.ndarray
            A series of T adjacency matrices of N vertices
        """
        G = nx.fast_gnp_random_graph(self.N, self.p_init, seed=self.seed)
        for u, v in G.edges():
            G.edges[u, v]['weight'] = self.random_state.rand()
        W = np.empty((self.T, self.N, self.N))
        for t in range(self.T):
            edges = np.array(G.edges)
            nonedges = np.array(list(nx.non_edges(G)))
            n_changes = np.ceil(edges.shape[0] * self.q_change).astype(int)
            chosen_existing_indices = self.random_state.choice(
                edges.shape[0],
                n_changes,
                replace=False
                )
            chosen_new_indices = self.random_state.choice(
                nonedges.shape[0],
                n_changes,
                replace=False
                )
            chosen_existing_edges = edges[chosen_existing_indices]
            chosen_new_edges = nonedges[chosen_new_indices]
            for u, v in chosen_existing_edges:
                G.remove_edge(u, v)
            for u, v in chosen_new_edges:
                G.add_edge(u, v, weight=self.random_state.rand())
            W[t] = nx.to_numpy_array(G)
        return W

    def generate_graph_signals_eachtime(
        self,
        K: int,
        sigma: float = 0.5
    ) -> np.ndarray:
        """
        Generates K smooth graph signals on the graph
        with adjacency matrix self.W[t] at each time t

        Parameters
        ----------
        K : int
            Number of graph signals generated at each time
        sigma : float, optional
            Standard deviation of Gaussian noise

        Returns
        -------
        np.ndarray
            A data matrix of shape (T, N, K)
        """
        X = np.empty((self.T, self.N, K))
        for t in range(self.T):
            d_vec = np.sum(self.W[t], axis=0)
            D = np.diag(d_vec)
            L = D - self.W[t]
            cov = np.linalg.inv(L + sigma ** 2 * np.identity(self.N))
            X_t = self.random_state.multivariate_normal(
                np.zeros(self.N),
                cov,
                size=K).transpose()
            X[t] = X_t
        return X
