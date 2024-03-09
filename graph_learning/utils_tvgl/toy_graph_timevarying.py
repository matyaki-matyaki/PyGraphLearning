import numpy as np
import networkx as nx

class TimevaryingGraph:
    """
    時変グラフのクラス
    
    Attributes
    ----------
    N: int
        頂点数
    T: int
        観測時刻数
    """
    def __init__(self, N, T):
        self.N = N
        self.T = T
        self.W = np.empty((T, N, N))

class TimevaryingErdosRenyiGraph(TimevaryingGraph):
    """
    時変ErdosRenyiグラフのクラス

    Attributes
    ----------
    N: int
        頂点数
    T: int
        観測時刻数
    p_init: float
        各頂点間に辺が貼られる確率
    q_change: float
        辺が新たに生まれる確率
    seed: int
        乱数シード値
    random_state: np.random.RandomState
        RandomState
    W: np.ndarray
        (T, N, N)の型である，T個の隣接行列の列    
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
        時変ErdosRenyiグラフモデルに従って重み付き隣接行列の列(T, N, N)を生成

        Returns
        -------
        np.ndarray
            T個の頂点数Nの隣接行列の列
        """
        G = nx.fast_gnp_random_graph(self.N, self.p_init, seed=self.seed)
        for u, v in G.edges():
            G.edges[u, v]['weight'] = self.random_state.rand()
        W = np.empty((self.T, self.N, self.N))
        for t in range(self.T):
            edges = np.array(G.edges)
            nonedges = np.array(list(nx.non_edges(G)))
            n_changes = np.ceil(edges.shape[0] * self.q_change).astype(int)
            chosen_existing_indices = self.random_state.choice(edges.shape[0], n_changes, replace=False)
            chosen_new_indices = self.random_state.choice(nonedges.shape[0], n_changes, replace=False)
            chosen_existing_edges = edges[chosen_existing_indices]
            chosen_new_edges = nonedges[chosen_new_indices]
            for u, v in chosen_existing_edges:
                G.remove_edge(u, v)
            for u, v in chosen_new_edges:
                G.add_edge(u, v, weight=self.random_state.rand())
            W[t] = nx.to_numpy_array(G)
        return W
    
    def generate_graph_signals_eachtime(self, K: int, sigma: float=0.5) -> np.ndarray:
        """
        各時刻tにself.W[t]を隣接行列に持つグラフ上で滑らかなグラフ信号をK個生成

        Parameters
        ----------
        K : int
            各時刻に生成されるグラフ信号の数
        sigma : float, optional
            ガウシアンノイズの標準偏差

        Returns
        -------
        np.ndarray
            (T, N, K)の形のデータ行列
        """
        X = np.empty((self.T, self.N, K))
        for t in range(self.T):
            d_vec = np.sum(self.W[t], axis=0)
            D = np.diag(d_vec)
            L = D - self.W[t]
            cov = np.linalg.inv(L + sigma ** 2 * np.identity(self.N))
            X_t = self.random_state.multivariate_normal(np.zeros(self.N), cov, size=K).transpose()
            X[t] = X_t
        return X
            
            
            
    