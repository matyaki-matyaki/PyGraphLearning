import matplotlib.pyplot as plt
from pathlib import Path

from pygraphlearning.graph_learning import GraphLearning
from pygraphlearning.utils.time_varying.time_varying_graph_model \
    import TimevaryingErdosRenyiGraph

# produce ground-truth graph and graph signals
timevarying_graph = TimevaryingErdosRenyiGraph(N=36, T=200)
X = timevarying_graph.generate_graph_signals_eachtime(K=100, sigma=0.25)
print(f'{timevarying_graph.W.shape=}, {X.shape=}')
# timevarying_graph.W.shape=(200, 36, 36), X.shape=(200, 36, 100)

# produce GraphLearning instance
gl = GraphLearning(
    graph_type='time-varying',
    beta=1e-4,
    eta=0.2
)

# learn time-varying graph
W_pred = gl.fit_transform(X)

# show result
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(timevarying_graph.W[i])
    plt.title(f'W_gt[{i}]')
    plt.axis('off')
    plt.subplot(2, 5, i + 6)
    plt.imshow(W_pred[i])
    plt.title(f'W_pred[{i}]')
    plt.axis('off')
plt.suptitle('Result (Time Varying Graph Learning)')
plt.tight_layout(pad=3.0)
plt.savefig(Path('examples', 'out', 'result_tvgl.png'))
