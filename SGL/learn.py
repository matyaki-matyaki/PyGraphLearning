import numpy as np
from scipy import sparse, spatial, linalg

def calc_z(X: np.ndarray) -> np.ndarray:
    """
    データ行列Xのpair-wise distances matrix Z のupper-right成分を集めたベクトルを計算する

    Parameters
    ----------
    X : np.ndarray
        K個のグラフ信号からなるデータ行列；行列の形は(N, K) (N: 頂点数)

    Returns
    -------
    np.ndarray
        N(N-1)/2次元のベクトルz
    """
    z = spatial.distance.pdist(X, 'sqeuclidean')
    z /= np.max(z)
    return z

def create_S(N: int) -> sparse.csr_array:
    """
    \bm{S}\bm{w} = \bm{d}を満たす行列\bm{S}の作成
    ここで，
    - \bm{w}: 隣接行列\bm{W}のupper-rightの成分を集めたN(N-1)/2次元ベクトル
    - \bm{d}: 隣接行列\bm{W}を持つグラフの次数を集めたN次元ベクトル
    
    Parameters
    ----------
    N : int
        頂点数

    Returns
    -------
    sparse.csr_array
        sparse.csr_array形式の行列S
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


def PDS_sgl(z: np.ndarray, A: sparse.csr_matrix, alpha: float, beta: float, gamma: float, maxit: int, epsilon: float) -> np.ndarray:
    """
    SGLの最適化問題を解くためのPDS（主双対近接分離法）
    PDS can solve: min. f(\bm{x}) + g(\bm{A}\bm{x}) + h(\bm{x}),
    where f, g and h are proper, convex lower-semicontinuous functions,
    and h is a differentiable function having a Lipschitzian gradient with a Lipschitz constant L    

    Parameters
    ----------
    z : np.ndarray
        データ行列Xのpair-wise distances matrix Z のupper-right成分を集めたベクトル
    A : sparse.csr_matrix
        行列A
    alpha : float
        最適化問題のハイパーパラメータ
    beta : float
        最適化問題のハイパーパラメータ
    gamma : float
        ステップサイズ
    maxit : int
        反復法の最大反復数
    epsilon : float
        停止条件用の閾値

    Returns
    -------
    np.ndarray
        最適解（推定された隣接行列；ベクトル形式）
    """

    
    # init
    x = np.zeros(z.shape)
    v = A.dot(x)
    
    prox_f = lambda x_f, gamma_: np.maximum(x_f - 2 * gamma_ * z, 0.)
    prox_g = lambda v_g, gamma_: (v_g + np.sqrt(np.square(v_g) + 4 * alpha * gamma_)) / 2
    prox_g_ast = lambda v_g_ast, gamma_: v_g_ast - gamma_ * prox_g(v_g_ast / gamma_, 1 / gamma_)
    nabla_h = lambda x_h: 4 * beta * x_h
    
    At = A.transpose()
    
    for _ in range(maxit):
        # Forward steps (in both primal and dual spaces)
        y_1 = x - gamma * (nabla_h(x) + At.dot(v))
        y_2 = v + gamma * A.dot(x)
        
        # Backward steps (in both primal and dual spaces)
        p_1 = prox_f(y_1, gamma)
        p_2 = prox_g_ast(y_2, gamma)
        
        # Forward steps (in both primal and dual spaces)
        q_1 = p_1 - gamma * (nabla_h(p_1) + At.dot(p_2))
        q_2 = p_2 + gamma * A.dot(p_1)
        
        # check stop condition
        new_x = x - y_1 + q_1
        new_v = v - y_2 + q_2        
        div = max(1e-4, np.linalg.norm(x))
        gap = np.linalg.norm(new_x - x)
        if gap / div < epsilon:
            break
        
        # Update solution (in both primal and dual spaces)
        x = new_x
        v = new_v
    
    return x
    

def vec2matrix(vec: np.ndarray, N: int) -> np.ndarray:
    """
    ベクトル形式の隣接行列を行列形式に変換

    Parameters
    ----------
    vec : np.ndarray
        ベクトル形式の隣接行列
    N : int
        頂点数

    Returns
    -------
    np.ndarray
        隣接行列
    """
    indices = np.triu_indices(N, 1)
    mat = np.empty((N, N))
    mat[indices[0], indices[1]] = vec
    mat = mat + mat.transpose()
    return mat


def sgl(X: np.ndarray, alpha: float=1., beta: float=1e-2, step: float=0.5, maxit: int=10000, epsilon: float=1e-5) -> np.ndarray:
    """
    静的グラフ学習の実行

    Parameters
    ----------
    X : np.ndarray
        K個のグラフ信号からなるデータ行列；行列の形は(N, K) (N: 頂点数)
    alpha : float, optional
        ハイパーパラメータ（結果となるグラフの辺重みのスケールを調整するパラメータであるので調整不要）, by default 1.
    beta : float, optional
        ハイパーパラメータ, by default 1e-2
    step : float, optional
        ステップサイズ（収束は保証されるので，調整不要）, by default 0.5
    maxit : int, optional
        反復法の最大反復数, by default 10000
    epsilon : float, optional
        停止条件用の閾値, by default 1e-5

    Returns
    -------
    np.ndarray
        推定された隣接行列
    """
    z = calc_z(X)
    S = create_S(X.shape[0])
    # リプシッツ定数
    lip_const = 4 * beta
    # Sのスペクトラルノルム
    spec_norm_S = np.sqrt(2 * (X.shape[0] - 1))
    # PDSで使用するステップサイズ
    gamma = step / (1 + lip_const + spec_norm_S)
    # 最適化問題を解く
    w = PDS_sgl(z, S, alpha, beta, gamma, maxit, epsilon)
    # ベクトル形式の隣接行列を行列形式に変換
    W = vec2matrix(w, X.shape[0])
    
    return W
    
    
    

    