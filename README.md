# Static and Time-Varying Graph Learning by Python

This repository contains Python implementations for graph learning methods, which are used to estimate the underlying graph structure (weighted adjacency matrix) using data observed at each vertex. In Static Graph Learning (SGL), it is possible to infer the structure of graphs that do not change over time. In Time-Varying Graph Learning (TVGL), the methods allow for the estimation of graph structures that change over time.

## Installation

```
git clone https://github.com/matyaki-matyaki/PyGraphLearning
cd PyGraphLearning
pip install -r requirements.txt
```

## Usage

For usage examples, please refer to the example.py file in this repository.

### SGL

#### Input　and Output of SGL

- Input: Data matrix (np.ndarray) of shape (N, K), where N is the number of vertices, and K is the number of data points observed at each vertex.
- Output: Adjacency matrix (np.ndarray) of shape (N, N).

#### Example Execution of SGL

To run the example of SGL, execute the following command:

```
python SGL/example.py
```

The hyperparameters for SGL are $\alpha$ and $\beta$. 
$\alpha (>0)$ is a parameter that governs the overall scale of the adjacency matrix, so it does not need to be tuned for understanding the structure of the graph. On the other hand, $\beta (\geq 0)$ is a parameter that controls the sparsity of the graph, with smaller values resulting in a sparser graph.

#### Example Output of SGL

Below is an example of the output you can expect after running SGL/example.py:

<img src="https://drive.google.com/uc?=view&id=1j8KZiqc5hRLqPNYUQtP7ftfFphCeuwyW" width= 100%>

### TVGL

#### Input and Output of TVGL

- Input: Data matrix (np.ndarray) of shape (T, N, K), where T is the number of observation time points, N is the number of vertices, and K is the number of data points observed at each vertex at each time point.
- Output: Sequence of adjacency matrices (np.ndarray) of shape (T, N, N).

#### Example Execution of TVGL

To run the example of TVGL, execute the following command:

```
python TVGL/example.py
```

The hyperparameters for TVGL are $\alpha$, $\beta$, and $\eta$, where the effects of $\alpha$ and $\beta$ are the same as in SGL. $\eta$ is a parameter that weighs the changes in the graph's structure. When $\eta$ is large, a time-invariant graph is obtained, and when $\eta$=0, the results are equivalent to performing SGL independently at each time point.

#### Example Output of TVGL

Below is an example of the output you can expect after running TVGL/example.py:

<img src="https://drive.google.com/uc?=view&id=1QbdWUrejyRWI2z8eZXIa3zU4nT4pOgBj" width= 100%>

## References
This project is based on the following papers:

- SGL: [[Kalofolias, 2016]](https://proceedings.mlr.press/v51/kalofolias16.html)
- TVGL: [[Yamada+, 2019]](https://ieeexplore.ieee.org/abstract/document/8682762)

## License
MIT
