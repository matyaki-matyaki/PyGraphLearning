import matplotlib.pyplot as plt

from learn import sgl
# To generate the ground-truth weighted adjacency matrix
from static_graph import StaticErdosRenyiGraph

N = 100 # number of nodes
K = 10000 # number of data observed at each node

G = StaticErdosRenyiGraph(N) # ground-truth graph (random graph)
X = G.generate_graph_signal(K) # observed data matrix

# conduct static graph learning
W = sgl(X, beta=1e-3)

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