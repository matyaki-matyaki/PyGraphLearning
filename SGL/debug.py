import matplotlib.pyplot as plt
import numpy as np

from learn import sgl 
from static_graph import StaticErdosRenyiGraph

N = 500
K = 100000

G = StaticErdosRenyiGraph(N)
X = G.generate_graph_signal(K)

W = sgl(X, beta=0)
print(W.shape)

# Displaying the images side by side
plt.figure(figsize=(12, 6))   # Adjust the figure size as needed

# Displaying G.W
plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
plt.imshow(G.W)
plt.title("G.W")

# Displaying W
plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
plt.imshow(W)
plt.title("W")

# 差の表示
plt.subplot(1, 3, 3)
plt.imshow(np.abs(G.W / np.linalg.norm(G.W) - W / np.linalg.norm(W)))
plt.title("Gap")

plt.show()

