import numpy as np
import networkx as nx


class StaticGraph:
    """
    Class for static graphs

    Attributes
    ----------
    N: int
        Number of vertices
    """
    def __init__(self, N: int) -> None:
        self.N = N
        self.W = np.empty((N, N), dtype=float)


class StaticErdosRenyiGraph(StaticGraph):
    """
    Class for static Erdos-Renyi graphs

    Attributes
    ----------
    N: int
        Number of vertices
    p: float
        Probability of an edge being present between two vertices
    seed: int
        Random seed value
    random_state: np.random.RandomState
        RandomState instance
    W : np.ndarray
        Adjacency matrix in mathbb{R}^{N x N}
    """
    def __init__(
        self,
        N: int,
        p: float = 0.05,
        seed: int = 42
    ) -> None:
        super().__init__(N)
        self.p = p
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        self.W = self._simulate()

    def _simulate(self) -> np.ndarray:
        """
        Generate a weighted adjacency matrix
        following the Erdos-Renyi graph model

        Returns
        -------
        np.ndarray
            Adjacency matrix
        """
        G = nx.fast_gnp_random_graph(self.N, self.p, seed=self.seed)
        for u, v in G.edges():
            G.edges[u, v]['weight'] = self.random_state.rand()
        W = nx.to_numpy_array(G)
        return W

    def generate_graph_signals(self, K: int, sigma: float = 0.5) -> np.ndarray:
        """
        Generate K smooth graph signals on a graph
        with self.W as its adjacency matrix

        Parameters
        ----------
        K : int
            Number of graph signals
        sigma : float
            Standard deviation of Gaussian noise

        Returns
        -------
        np.ndarray
            Data matrix consisting of K graph signals; matrix shape is (N, K)
            (N: number of vertices)
        """
        d_vec = np.sum(self.W, axis=0)
        D = np.diag(d_vec)
        L = D - self.W
        cov = np.linalg.inv(L + sigma**2 * np.identity(self.N))
        X = self.random_state.multivariate_normal(
            np.zeros(self.N),
            cov,
            size=K
            ).transpose()
        return X


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    static_graph = StaticErdosRenyiGraph(N=36, p=0.1)

    # weighted adjacancy matrix
    plt.imshow(static_graph.W)
    plt.colorbar()
    plt.title('W')
    plt.savefig('tmp_0.png')
    plt.close()

    # graph signals
    X = static_graph.generate_graph_signals(K=100, sigma=0.1)
    plt.imshow(X)
    plt.colorbar()
    plt.title('graph signals')
    plt.savefig('tmp_1.png')
    plt.close()
