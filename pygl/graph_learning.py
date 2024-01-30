import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .utils_sgl import sgl
from .utils_tvgl import tvgl

class GraphLearning(BaseEstimator, TransformerMixin):
    """
    Perform static/time-varying graph learning.

    Parameters
    ----------
    graph_type : {'static', 'time-varying}, optional
        Method for graph learning, by default 'static'
    is_laplacian : bool, optional
        Whether to convert the adjacency matrix to the graph Laplacian and return it, by default False
    positivity_offset : float | None, optional
        How much offset to add to the graph Laplacian to make it a positive definite matrix., by default None
    alpha : float, optional
        Hyperparameter (adjusting the scale of edge weights of the resulting graph, so no adjustment needed), by default 1.
    beta : float, optional
        Hyperparameter (adjusting the sparcity of the graph), by default 1e-2
    eta : float | None, optional
        Hyperparameter (adjusting the variability of the time-varying graph), by default None
    step : float, optional
        Step size (convergence is guaranteed, so no adjustment needed), by default 0.5
    max_iter : int, optional
        Maximum number of iterations, by default 10000
    epsilon : float, optional
        Threshold for the stopping condition, by default 1e-5
    """
    def __init__(
        self,
        graph_type: str = 'static',
        is_laplacian: bool = False,
        positivity_offset: float | None = None,
        alpha: float = 1.,
        beta: float = 1e-2,
        eta: float | None = None,
        step: float = 0.5,
        max_iter : int = 10000,
        epsilon : float = 1e-5
    ):
        """
        Init.
        
        Parameters
        ----------
        graph_type : {'static', 'time-varying}, optional
            Method for graph learning, by default 'static'
        is_laplacian : bool, optional
            Whether to convert the adjacency matrix to the graph Laplacian and return it, by default False
        positivity_offset : float | None, optional
            How much offset to add to the graph Laplacian to make it a positive definite matrix., by default None
        alpha : float, optional
            Hyperparameter (adjusting the scale of edge weights of the resulting graph, so no adjustment needed), by default 1.
        beta : float, optional
            Hyperparameter (adjusting the sparcity of the graph), by default 1e-2
        eta : float | None, optional
            Hyperparameter (adjusting the variability of the time-varying graph), by default None
        step : float, optional
            Step size (convergence is guaranteed, so no adjustment needed), by default 0.5
        max_iter : int, optional
            Maximum number of iterations, by default 10000
        epsilon : float, optional
            Threshold for the stopping condition, by default 1e-5
        """
        if graph_type not in {'static', 'time-varying'}:
            raise ValueError("graph_type must be 'static' or 'time-varying'.")
        else:
            self.graph_type = graph_type
        self.is_laplacian = is_laplacian
        if self.is_laplacian:
            self.positivity_offset = positivity_offset
        else:
            self.positivity_offset = None
        self.alpha = alpha
        self.beta = beta
        if self.graph_type == 'static':
            self.eta = None
        else:
            self.eta = eta
        self.step = step
        self.max_iter = max_iter
        self.epsilon = epsilon
    
    def fit(self, X, y=None):
        """
        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : np.ndarray
            - For static graph learning: Data matrix consisting of K graph signals; the shape of the matrix is (N, K) (N: number of vertices)
            - For time-varying graph learning: Data matrix with K graph signals at each observation time; matrix shape is (T, N, K) (T: number of observation times)
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : GraphLearning instance
            The GraphLearning instance.
        """
        return self
    
    def transform(self, X):
        """
        - For static graph learning: Estimate a weighted adjacency matrix or a graph laplacian matrix
        - For time-varying graph learing: Estimate weighted adjacency matrices or graph laplacians matrices

        Parameters
        ----------
        X : np.ndarray
            - For static graph learning: Data matrix consisting of K graph signals; the shape of the matrix is (N, K) (N: number of vertices)
            - For time-varying graph learning: Data matrix with K graph signals at each observation time; matrix shape is (T, N, K) (T: number of observation times)

        Returns
        -------
        np.ndarray
            - For static graph learning: Estimated matrix (N, N)
            - For time-varying graph learning: Series of estimated matrices (T, N, N)
        """
        if self.graph_type == 'static':
            W = sgl(
                X,
                alpha=self.alpha,
                beta=self.beta,
                step=self.step,
                max_iter=self.max_iter,
                epsilon=self.epsilon
            )
            if self.is_laplacian:
                d = np.sum(W, axis=-1)
                D = np.diag(d)
                L = D - W
                if self.positivity_offset is not None:
                    L += self.positivity_offset * np.eye(L.shape[-1])
                return L
            else:
                return W
        else:
            W_array = tvgl(
                X,
                alpha=self.alpha,
                beta=self.beta,
                eta=self.eta,
                step=self.step,
                max_iter=self.max_iter,
                epsilon=self.epsilon
            )
            if self.is_laplacian:
                d = np.sum(W_array, axis=-1)
                D_array = d[:, np.newaxis, :] * np.eye(W_array.shape[-1])
                L_array = D_array - W_array
                if self.positivity_offset is not None:
                    L_array += self.positivity_offset * np.eye(L_array.shape[-1])
                return L_array
            else:
                return W_array
            
