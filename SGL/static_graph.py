import numpy as np
import networkx as nx


class StaticGraph:
    """
    静的グラフのクラス
    
    Attributes
    ----------
    N: int
        頂点数
    """
    def __init__(self, N):
        self.N = N
        self.W = np.empty((N, N))


class StaticErdosRenyiGraph(StaticGraph):
    """
    静的ErdosRenyiグラフのクラス

    Attributes
    ----------
    N: int
        頂点数
    p: float
        頂点間に辺が貼られる確率
    seed: int
        乱数シード値
    random_state: np.random.RandomState
        RandomState
    W : np.ndarray
        隣接行列\in\mathbb{R}^{N\times N}
    """
    def __init__(self, N, p=0.05, seed=42):
        super().__init__(N)
        self.p = p
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        self.W = self._simulate()
    
    def _simulate(self) -> np.ndarray:
        """
        ErdosRenyiグラフモデルに従って重み付き隣接行列を生成

        Returns
        -------
        np.ndarray
            隣接行列
        """
        G = nx.fast_gnp_random_graph(self.N, self.p, seed=self.seed)
        for u, v in G.edges():
            G.edges[u, v]['weight'] = self.random_state.rand()
        W = nx.to_numpy_array(G)
        return W
    
    def generate_graph_signal(self, K: int, sigma: float=0.5) -> np.ndarray:
        """
        self.Wを隣接行列に持つグラフ上で滑らかなグラフ信号をK個生成

        Parameters
        ----------
        K : int
            グラフ信号の数
        sigma : ガウシアンノイズの標準偏差

        Returns
        -------
        np.ndarray
            K個のグラフ信号からなるデータ行列；行列の形は(N, K) (N: 頂点数)
        """
        d_vec = np.sum(self.W, axis=0)
        D = np.diag(d_vec)
        L = D - self.W
        cov = np.linalg.inv(L + sigma**2 * np.identity(self.N))
        X = self.random_state.multivariate_normal(np.zeros(self.N), cov, size=K).transpose()
        return X
            
