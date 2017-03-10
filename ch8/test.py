import numpy as np
from BP import *

# tests of belief propagation for solving the independent set problem; a set of vertices is called an independent set
# if there're no edges between any of them. We let each vertex be a binary random variable and encode the independent
# set constraint with the edge potential psi(x,y)=I(x+y<=1), where I is the indicator function; we also place uniform
# potential over vertices; together they define the uniform distributions over independent sets of a graph. Then the
# partition function is precisely the number of independent sets. [This is the forward computation from canonical
# parameters to mean parameters).

# trees
# 0---1
# |
# 2
A = np.array([[0, 1, 1],
              [1, 0, 0],
              [1, 0, 0]])

# 0---1
# |   |
# 2   3
B = np.array([[0, 1, 1, 0],
              [1, 0, 0, 1],
              [1, 0, 0, 0],
              [0, 1, 0, 0]])

# # 0---1---2
# # |   |
# # 3   4
# C = np.array([[0, 1, 0, 1, 0],
#               [1, 0, 1, 0, 1],
#               [0, 1, 0, 0, 0],
#               [1, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0]])

node_potential = gen_node_potential([0, 0])
edge_potential = iset_edge_potential

GB = Graph(B)
print("2-pass sum-product on the tree: ")
tree_sum_product(GB, node_potential, edge_potential)  # exact answer
print()
print("Non-recursive sum-product on the tree:")
A = sum_product(GB, node_potential, edge_potential, 10)  # the non-recursive implementation should work the same
print('There are ' + str(np.exp(A)) + ' independent sets in this graph')

# loopy
# 0---1
# | \ |
# 2   3
Y = np.array([[0, 1, 1, 1],
              [1, 0, 0, 1],
              [1, 0, 0, 0],
              [1, 1, 0, 0]])

print()
print("Non-recursive sum-product on general loopy graph:")
GY = Graph(Y)
A = sum_product(GY, node_potential, edge_potential, 30)  # loopy belief propagation; will be approximate
print('There are approximately ' + str(np.exp(A)) + ' independent sets in this graph')
