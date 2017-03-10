import numpy as np


class Graph:
    """A generic graph"""

    def __init__(self, adjmat):
        """Create graph from adjacency matrix"""
        # assuming adjmat is valid

        # number of nodes
        N = adjmat.shape[0]
        self.N = N

        # list of vertices ordered by ids
        self.Vs = np.arange(N)

        # list of lists containing neighbor ids; e.g. Nbs[0] contains [1,3,4]
        self.Nbs = [[]] * N

        for r, row in enumerate(adjmat):
            self.Nbs[r] = self.Vs[np.where(row)[0]]


# singleton and pairwise potentials for MRF; for simplicity they'll be the same across vertices/edges
def gen_node_potential(weights):
    """Generate a singleton (node) potential function from a list of canonical parameters for variable states"""
    f = lambda x: np.exp(weights[x])
    f.size = len(weights)  # size of the discrete variable state space, which is {0,1,...,f.size-1}
    return f


def color_edge_potential(x, y):
    """Edge potential in the pair-wise MRF for the coloring problem"""
    return int(x != y)


def iset_edge_potential(x, y):
    """Edge potential in the pair-wise MRF for the independent-set problem (less restrictive/more interesting than
     the coloring potential)"""
    return int((x + y) <= 1)