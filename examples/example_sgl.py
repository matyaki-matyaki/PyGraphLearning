import matplotlib.pyplot as plt

from graph_learning.graph_learn import GraphLearning
from graph_learning.utils_sgl.toy_graph_static import StaticErdosRenyiGraph

N = 100 # number of nodes
K = 1000 # number of data observed at each node

# To generate the ground-truth weighted adjacency matrix
G = StaticErdosRenyiGraph(N) # ground-truth graph (random graph)
X = G.generate_graph_signals(K) # observed data matrix

# conduct static graph learning
W = GraphLearning(beta=1e-3).fit_transform(X)

# Displaying the images side by side
plt.figure(figsize=(12, 6))

# Displaying G.W
plt.subplot(1, 2, 1)
plt.imshow(G.W)
plt.title("Ground-truth")

# Displaying W
plt.subplot(1, 2, 2)
plt.imshow(W)
plt.title("Estimated")

plt.show()