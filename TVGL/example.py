import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

from learn import tvgl
# To generate the ground-truth weighted adjacency matrix
from timevarying_graph import TimevaryingErdosRenyiGraph

N = 40 # number of nodes
K = 100000 # number of data observed at each node
T = 200  # number of times

G = TimevaryingErdosRenyiGraph(N, T) # ground-truth graph (random graph)
X = G.generate_graph_signals_eachtime(K) # observed data matrix

# conduct static graph learning
W = tvgl(X, eta=1)

# Displaying the images side by side
plt.figure(figsize=(12, 6))

t1, t2 = 5, 6
# Displaying G.W
plt.subplot(2, 2, 1)
plt.imshow(G.W[t1])
plt.title(f"Ground-truth (t={t1})")
plt.subplot(2, 2, 2)
plt.imshow(G.W[t2])
plt.title(f"Ground-truth (t={t2})")

# Displaying W
plt.subplot(2, 2, 3)
plt.imshow(W[t1])
plt.title(f"Estimated (t={t1})")
plt.subplot(2, 2, 4)
plt.imshow(W[t2])
plt.title(f"Estimated (t={t2})")

plt.show()