import numpy as np
from scipy import sparse, spatial

def _calc_z(X: np.ndarray) -> np.ndarray:
    """
    Calculate the vector of the upper-right elements of the pair-wise distances matrix Z of the data matrix X

    Parameters
    ----------
    X : np.ndarray
        Data matrix consisting of K graph signals; the shape of the matrix is (N, K) (N: number of vertices)

    Returns
    -------
    np.ndarray
        A vector of dimension N(N-1)/2
    """
    z = spatial.distance.pdist(X, 'sqeuclidean')
    z /= np.max(z)
    return z

def _create_S(N: int) -> sparse.csr_array:
    """
    Create the matrix \bm{S} that satisfies \bm{S}\bm{w} = \bm{d}
    Here,
    - \bm{w}: A N(N-1)/2-dimensional vector composed of the upper-right elements of the adjacency matrix \bm{W}
    - \bm{d}: N-dimensional vector of degrees of the graph with adjacency matrix \bm{W}
    
    Parameters
    ----------
    N : int
        Number of vertices

    Returns
    -------
    sparse.csr_array
        Matrix S in sparse.csr_array format
    """
    row_indices = np.repeat(np.arange(N), (N - 1))
    col_indices = np.arange(N * (N - 1))
    tmp_array = np.arange(N - 1)
    for i in range(N - 1):
        tmp_array[i + 1 :] += N - 2 - i
        tmp_array[:i] += 1
        col_indices[(N - 1) * (i + 1) : (N - 1) * (i + 2)] = tmp_array
    data = np.ones(N * (N - 1), dtype = np.float64)
    S = sparse.csr_array((data, (row_indices, col_indices)), shape = (N, N*(N - 1) // 2))
    
    return S


def _PDS_sgl(z: np.ndarray, A: sparse.csr_array, alpha: float, beta: float, gamma: float, maxit: int, epsilon: float) -> np.ndarray:
    """
    Primal Dual Splitting (PDS) method for solving the optimization problem of SGL
    PDS can solve: min. f(\bm{x}) + g(\bm{A}\bm{x}) + h(\bm{x}),
    where f, g, and h are proper, convex lower-semicontinuous functions,
    and h is a differentiable function having a Lipschitzian gradient with a Lipschitz constant L
    Here, we solve the following problem:
    min. 2\bm{x}^\top\bm{z} + I_{x\geq 0}(x) - \alpha\log(\bm{A}\bm{x}) + 2\beta\|\bm{x}\|^2

    Parameters
    ----------
    z : np.ndarray
        Vector composed of the upper-right elements of the pair-wise distances matrix Z of the data matrix
    A : sparse.csr_array
        Matrix A
    alpha : float
        Hyperparameter for the optimization problem
    beta : float
        Hyperparameter for the optimization problem
    gamma : float
        Step size
    maxit : int
        Maximum number of iterations for the iterative method
    epsilon : float
        Threshold for the stopping condition

    Returns
    -------
    np.ndarray
        The optimal solution (estimated adjacency matrix in vector form)
    """
        
    # init
    x = np.zeros(z.shape)
    v = A.dot(x)
    
    # def prox.
    prox_f = lambda x_f, gamma_: np.maximum(x_f - 2 * gamma_ * z, 0.)
    prox_g = lambda v_g, gamma_: (v_g + np.sqrt(np.square(v_g) + 4 * alpha * gamma_)) / 2
    prox_g_ast = lambda v_g_ast, gamma_: v_g_ast - gamma_ * prox_g(v_g_ast / gamma_, 1 / gamma_)
    nabla_h = lambda x_h: 4 * beta * x_h
    
    At = A.transpose()
    
    for _ in range(maxit):
        # Forward steps (in both primal and dual spaces)
        y1 = x - gamma * (nabla_h(x) + At.dot(v))
        y2 = v + gamma * A.dot(x)
        
        # Backward steps (in both primal and dual spaces)
        p1 = prox_f(y1, gamma)
        p2 = prox_g_ast(y2, gamma)
        
        # Forward steps (in both primal and dual spaces)
        q1 = p1 - gamma * (nabla_h(p1) + At.dot(p2))
        q2 = p2 + gamma * A.dot(p1)
        
        # check stop condition
        new_x = x - y1 + q1
        new_v = v - y2 + q2        
        div = max(1e-4, np.linalg.norm(x))
        gap = np.linalg.norm(new_x - x)
        if gap / div < epsilon:
            break
        
        # Update solution (in both primal and dual spaces)
        x = new_x
        v = new_v
    
    return x
    

def _vec2matrix(vec: np.ndarray, N: int) -> np.ndarray:
    """
    Convert a vector-form adjacency matrix into matrix form

    Parameters
    ----------
    vec : np.ndarray
        Vector-form adjacency matrix
    N : int
        Number of vertices

    Returns
    -------
    np.ndarray
        Adjacency matrix
    """
    matrix = np.empty((N, N))
    matrix = spatial.distance.squareform(vec)

    return matrix


def sgl(X: np.ndarray, alpha: float=1., beta: float=1e-2, step: float=0.5, maxit: int=10000, epsilon: float=1e-5) -> np.ndarray:
    """
    Execution of static graph learning

    Parameters
    ----------
    X : np.ndarray
        Data matrix consisting of K graph signals; the shape of the matrix is (N, K) (N: number of vertices)
    alpha : float, optional
        Hyperparameter (adjusting the scale of edge weights of the resulting graph, so no adjustment needed), by default 1.
    beta : float, optional
        Hyperparameter, by default 1e-2
    step : float, optional
        Step size (convergence is guaranteed, so no adjustment needed), by default 0.5
    maxit : int, optional
        Maximum number of iterations for the iterative method, by default 10000
    epsilon : float, optional
        Threshold for the stopping condition, by default 1e-5

    Returns
    -------
    np.ndarray
        Estimated adjacency matrix
    """
    z = _calc_z(X)
    S = _create_S(X.shape[0])
    # Lipschitz constant
    lip_const = 4 * beta
    # Spectral norm of S
    spec_norm_S = np.sqrt(2 * (X.shape[0] - 1))
    # Step size for use in PDS
    gamma = step / (1 + lip_const + spec_norm_S)
    # Solve the optimization problem
    w = _PDS_sgl(z, S, alpha, beta, gamma, maxit, epsilon)
    # Convert vector-form adjacency matrix to matrix form
    W = _vec2matrix(w, X.shape[0])
    
    return W