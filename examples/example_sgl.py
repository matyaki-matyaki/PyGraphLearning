import matplotlib.pyplot as plt
from pathlib import Path

from pygraphlearning.graph_learning import GraphLearning
from pygraphlearning.utils.static.static_graph_model \
    import StaticErdosRenyiGraph

# produce ground-truth graph and graph signals
static_graph = StaticErdosRenyiGraph(N=36, p=0.05)
X = static_graph.generate_graph_signals(K=100, sigma=0.25)
print(f'{static_graph.W.shape=}, {X.shape=}')
# static_graph.W.shape=(36, 36), X.shape=(36, 100)

# produce GraphLearning instance
gl = GraphLearning(
    graph_type='static',
    beta=1e-4
)

# learn static graph
W_pred = gl.fit_transform(X)

# show result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(static_graph.W)
plt.title('Ground Truth')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(W_pred)
plt.title('Estimated')
plt.axis('off')
plt.suptitle('Result (Static Graph Learning)')
plt.savefig(Path('examples', 'out', 'result_sgl.png'))
