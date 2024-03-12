import numpy as np
from scipy import spatial, sparse


def _calc_z(X: np.ndarray) -> np.ndarray:
    """
    Calculate a vector consisting only of the upper-right elements of
    the pair-wise distances matrix Z of the data matrix X

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (T, N, K)

    Returns
    -------
    np.ndarray
        A vector of dimension TN(N-1)/2
    """
    T, N, K = X.shape
    C = N * (N-1) // 2
    z = np.empty(T * C)
    for t in range(T):
        z_tmp = spatial.distance.pdist(X[t], 'sqeuclidean')
        z_tmp /= np.max(z_tmp)
        z[t * C: (t+1) * C] = z_tmp

    return z


def _create_S_and_P(
    N: int,
    T: int
) -> tuple[sparse.csr_array, sparse.csr_array]:
    """
    Creation of matrices S and P
    Create matrix S that satisfies \bm{S}\bm{w} = \bm{d}, and matrix P
    that satisfies \bm{P}\bm{w} = \bm{w} - hat{\bm{w}}
    Where,
    - \bm{w}^{(t)}: A N(N-1)/2-dimensional vector composed of
        the upper-right elements of the adjacency matrix \bm{W}^{(t)} at time t
    - \bm{d}^{(t)}: A N-dimensional vector of degrees of the graph
        with adjacency matrix \bm{W}^{(t)} at time t
    - \bm{w} = [\bm{w}^{(1)} \bm{w}^{(2)} ... \bm{w}^{(T)}]
    - \bm{d} = [\bm{d}^{(1)} \bm{d}^{(2)} ... \bm{d}^{(T)}]
    - hat{\bm{w}} = [\bm{w}^{(1)} \bm{w}^{(1)} ... \bm{w}^{(T-1)}]
        (vector \bm{w} shifted by one)

    Parameters
    ----------
    N : int
        Number of vertices
    T : int
        Number of observation times

    Returns
    -------
    tuple[sparse.csr_array, sparse.csr_array]
        Matrices S and P
    """
    C = N * (N-1) // 2

    # generate S
    row_indices_S = np.repeat(np.arange(N * T), (N - 1))
    col_indices_S = np.arange(N * (N - 1) * T)
    tmp_array = np.arange(N - 1)
    for i in range(N - 1):
        tmp_array[i + 1:] += N - 2 - i
        tmp_array[:i] += 1
        col_indices_S[(N - 1) * (i + 1): (N - 1) * (i + 2)] = tmp_array
    for i in range(1, T):
        col_indices_S[N * (N-1) * i: N * (N-1) * (i+1)] =\
            col_indices_S[N * (N - 1) * (i - 1): N * (N - 1) * i] + C
    data_S = np.ones(N * (N - 1) * T, dtype=np.float64)
    S = sparse.csr_array(
        (data_S, (row_indices_S, col_indices_S)),
        shape=(N * T, C * T)
        )

    # Pの生成
    row_indices_P = np.repeat(np.arange(C, C * T), 2)
    col_indices_P = np.repeat(np.arange(C * (T-1)), 2)
    col_indices_P[1::2] += C
    data_P = np.ones(N * (N-1) * (T-1))
    data_P[0::2] = -1
    P = sparse.csr_array(
        (data_P, (row_indices_P, col_indices_P)),
        shape=(C * T, C * T)
        )

    return S, P


def _PDS_tvgl(
    z: np.ndarray,
    S: sparse.csr_array,
    P: sparse.csr_array,
    alpha: float,
    beta: float,
    eta: float,
    gamma: float,
    max_iter: int,
    epsilon: float
) -> np.ndarray:
    """
    PDS for solving the optimization problem of TVGL

    Parameters
    ----------
    z : np.ndarray
        Vector composed of the upper-right elements of the pair-wise distances
        matrix Z of the data matrix
    S : sparse.csr_array
        Matrix S
    P : sparse.csr_array
        Matrix P
    alpha : float
        Hyperparameter for the optimization problem
    beta : float
        Hyperparameter for the optimization problem
    eta : float
        Hyperparameter for the optimization problem
    gamma : float
        Step size
    max_iter : int
        Maximum number of iterations
    epsilon : float
        Threshold for the stopping condition

    Returns
    -------
    np.ndarray
        The optimal solution
        (vector form of estimated adjacency matrix columns)
    """

    # init
    x = np.zeros(z.shape)
    v1 = S.dot(x)
    v2 = P.dot(x)

    # def prox.
    def prox_f(x_f, gamma_):
        return np.maximum(x_f - 2 * gamma_ * z, 0.)

    def prox_g1(v_g1, gamma_):
        return (v_g1 + np.sqrt(np.square(v_g1) + 4 * alpha * gamma_)) / 2

    def prox_g2(v_g2, gamma_):
        return np.sign(v_g2) * np.maximum(np.abs(v_g2) - eta * gamma_, 0.)

    def prox_g1_ast(v_g1_ast, gamma_):
        return v_g1_ast - gamma_ * prox_g1(v_g1_ast / gamma_, 1 / gamma_)

    def prox_g2_ast(v_g2_ast, gamma_):
        return v_g2_ast - gamma_ * prox_g2(v_g2_ast / gamma_, 1 / gamma_)

    def nabla_h(x_h):
        return 4 * beta * x_h

    St = S.transpose()
    Pt = P.transpose()

    for _ in range(max_iter):
        # Forward steps (in both primal and dual spaces)
        y0 = x - gamma * (nabla_h(x) + St.dot(v1) + Pt.dot(v2))
        y1 = v1 + gamma * S.dot(x)
        y2 = v2 + gamma * P.dot(x)

        # Backward steps (in both primal and dual spaces)
        p0 = prox_f(y0, gamma)
        p1 = prox_g1_ast(y1, gamma)
        p2 = prox_g2_ast(y2, gamma)

        # Forward steps (in both primal and dual spaces)
        q0 = p0 - gamma * (nabla_h(p0) + St.dot(p1) + Pt.dot(p2))
        q1 = p1 + gamma * S.dot(p0)
        q2 = p2 + gamma * P.dot(p0)

        # check stop condition
        new_x = x - y0 + q0
        new_v1 = v1 - y1 + q1
        new_v2 = v2 - y2 + q2
        div = max(1e-4, float(np.linalg.norm(x)))
        gap = np.linalg.norm(new_x - x)
        if gap / div < epsilon:
            break

        # Update solution (in both primal and dual spaces)
        x = new_x
        v1 = new_v1
        v2 = new_v2

    return x


def _vec2matrix(vec: np.ndarray, N: int, T: int) -> np.ndarray:
    """
    Convert vector-form adjacency matrices to matrix form

    Parameters
    ----------
    vec : np.ndarray
        Vector-form adjacency matrices
    N : int
        Number of vertices
    T : int
        Number of observation times

    Returns
    -------
    np.ndarray
        Series of adjacency matrices (T, N, N)
    """
    matrices = np.empty((T, N, N))
    C = N * (N - 1) // 2
    for t in range(T):
        matrices[t] = spatial.distance.squareform(vec[C * t: C * (t+1)])

    return matrices


def tvgl(
    X: np.ndarray,
    alpha: float = 1.,
    beta: float = 1e-2,
    eta: float = 2.0,
    step: float = 0.5,
    max_iter: int = 10000,
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Execution of time-varying graph learning

    Parameters
    ----------
    X : np.ndarray
        Data matrix with K graph signals at each observation time;
        matrix shape is (T, N, K) (N: number of vertices)
    alpha : float, optional
        Hyperparameter (no need to adjust as it adjusts the scale of
        edge weights of the resulting graph), by default 1.
    beta : float, optional
        Hyperparameter, by default 1e-2
    eta : float, optional
        Hyperparameter, by default 2.0
    step : float, optional
        Step size (convergence is guaranteed, so no adjustment needed),
        by default 0.5
    max_iter : int, optional
        Maximum number of iterations, by default 10000
    epsilon : float, optional
        Threshold for the stopping condition, by default 1e-5

    Returns
    -------
    np.ndarray
        Series of estimated adjacency matrices (T, N, N)
    """
    T, N, K = X.shape
    z = _calc_z(X)
    S, P = _create_S_and_P(N, T)
    # Lipschitz constant
    lip_const = 4 * beta
    # Spectral norm of the linear operator
    A = sparse.vstack((S, P))
    spec_norm_A = sparse.linalg.norm(A, 2)
    # Step size for use in PDS
    gamma = step / (1 + lip_const + spec_norm_A)
    # Solve the optimization problem
    w = _PDS_tvgl(z, S, P, alpha, beta, eta, gamma, max_iter, epsilon)
    # Convert the vector-form adjacency matrix columns to matrix form
    W = _vec2matrix(w, N, T)

    return W
