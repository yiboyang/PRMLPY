"""(Loopy) Belief propagation for pair-wise Markov random field"""

import numpy as np
from MRF import *
import logging


def calc_message(graph, phi, psi, M, j, i, op=sum):
    # calculate the message m_ji, from j to i, using the central storage M of messages
    # calculates either sum-product message (op=sum) or max-product message(op=max)
    # we also normalize each message to avoid overflow/underflow
    # phi, psi are callables; phi is the singleton potential; psi is the pair-wise potential

    N = graph.N  # number of vertices; there can be at most N^2 messages in a graph
    K = phi.size  # number of states of x_i

    nbr_msgs = [M[n][j] for n in range(N) if n != i]  # all children messages into j
    nbr_msg = np.product(nbr_msgs, axis=0)  # a vector containing the entry-wise products

    msg = np.empty(K)

    for i in range(K):
        # key equation: m_ji(x_i) = op_xj phi(x_j) psi(x_i,x_j) prod_{k in neighbors of j except i}(m_kj(x_j))
        msg[i] = op(phi(j) * psi(i, j) * nbr_msg[j] for j in range(K))

    return msg / msg.sum()


def tree_sum_product(graph, phi, psi):
    """Recursive tree implementation for testing (will loop forever if graph contains loop!);
    exact inference in two (forward+backward) passes"""

    N = graph.N  # number of vertices; there can be at most N^2 messages in a graph
    K = phi.size  # number of states of x_i
    phi_vals = np.array([phi(k) for k in range(K)])  # cache the array of node potential values

    # we store all the messages in a 3D array; the (i,j,:)th entry is a vector
    # that store the values of the function m_ij(x_j) for all the K states of x_j;
    # the ith horizontal slice contains the messages from node i; the jth vertical
    # slice contains the messages to node j
    # initialize to unit messages; crucial (so that multiplying messages from non-neighbors has no effect)
    M = np.ones((N, N, K))

    def collect(i, j):
        # let node i collect all messages from the subtree rooted at j
        for k in graph.Nbs[j]:  # calculate children messages first, if any
            if k == i: continue
            collect(j, k)
        # send message
        M[j][i] = calc_message(graph, phi, psi, M, j, i)

    def distribute(i, j):
        # distribute message from node i to all the nodes in the subtree rooted at j
        M[i][j] = calc_message(graph, phi, psi, M, i, j)
        for k in graph.Nbs[j]:  # distribute messages to children, if any
            if k == i: continue
            distribute(j, k)

    # arbitrarily select a root
    f = graph.Vs[0]

    for e in graph.Nbs[f]:  # first pass; messages flow inward from leaves to root
        collect(f, e)
    for e in graph.Nbs[f]:  # second pass; messages flow outward from root to leaves
        distribute(f, e)

    # messages have been passed both ways; ready to calculate marginal distributions for all nodes
    print("Node marginal probabilities:")
    for i in graph.Vs:
        nbr_msg = np.product(M[:, i], axis=0)  # a vector containing the entry-wise product of neighbor messages
        marg = phi_vals * nbr_msg  # product of self-potential and messages
        marg /= marg.sum()  # normalize
        print(i, marg)

    # also calculate the marginals on edges
    # each marginal will be a vector of length K^2, for each possible combination of x_i and x_j
    # e.g. if K=3, the combinations will be (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)

    # we calculate the product of potentials, phi(x_i)*phi(x_j)*psi(x_i,x_j) ahead of time because it's the same
    # for all pairs (x_i, x_j) in our model; all stored in vectors of length K^2
    phi_i_vals = np.repeat(phi_vals, K)  # 0,0,0,1,1,1,2,2,2
    phi_j_vals = np.tile(phi_vals, K)  # 0,1,2,0,1,2,0,1,2
    psi_vals = np.array([psi(i, j) for i in range(K) for j in range(K)])  # pair-wise potentials for all x_i x_j
    potential_prod = psi_vals * phi_i_vals * phi_j_vals

    print("Edge marginal probabilities:")
    for i in graph.Vs:
        for j in graph.Nbs[i]:
            if i > j:   continue  # avoid calculating the same edge marginal twice
            i_nbr_msgs = [M[n][i] for n in range(N) if n != j]
            i_nbr_msg = np.product(i_nbr_msgs, axis=0)  # the product of messages sent to i from all of its children

            j_nbr_msgs = [M[n][j] for n in range(N) if n != i]
            j_nbr_msg = np.product(j_nbr_msgs, axis=0)  # the product of messages sent to j from all of its children

            # Cartesian product of i_nbr_msg and j_nbr_msg, (K^2) x 2 matrix
            nbr_msgs = np.array([(i, j) for i in i_nbr_msg for j in j_nbr_msg])
            nbr_msg = np.product(nbr_msgs, axis=1)  # K^2 vector of neighbor message products for all x_i,x_j configs

            marg = potential_prod * nbr_msg
            marg /= marg.sum()  # normalize
            print(i, j, marg)


def sum_product(graph, phi, psi, its, calc_bethe=True):
    """sum-product, not recursive; in each iteration, go through all nodes one by one (in no particular order),
    let the selected node pass messages to all its neighbors; all the messages are continually updated.
    Converges and is exact for trees; no guarantee for loopy graphs; optionally also calculate the Bethe free
    energy after each iteration"""
    N = graph.N  # number of vertices; there can be at most N^2 messages at a time
    K = phi.size  # number of states of x_i

    phi_vals = np.array([phi(k) for k in range(K)])  # cache the array of node potential values

    # cache the product of potentials, phi(x_i)*phi(x_j)*psi(x_i,x_j) ahead of time because it's the same
    # for all pairs (x_i, x_j) in our model; all stored in vectors of length K^2
    phi_i_vals = np.repeat(phi_vals, K)  # 0,0,0,1,1,1,2,2,2
    phi_j_vals = np.tile(phi_vals, K)  # 0,1,2,0,1,2,0,1,2
    psi_vals = np.array([psi(i, j) for i in range(K) for j in range(K)])  # pair-wise potentials for all x_i x_j
    edge_potential_prods = psi_vals * phi_i_vals * phi_j_vals

    # we store all the messages in a 3D array; the (i,j,:)th entry is a vector
    # that store the values of the function m_ij(x_j) for all the K states of x_j;
    # the ith horizontal slice contains the messages from node i; the jth vertical
    # slice contains the messages sent to node j
    # initialize to unit messages; crucial (so that multiplying messages from non-neighbors has no effect)
    M = np.ones((N, N, K))
    M_prev = M.copy()  # the messages from the previous iteration; used for checking convergence

    if calc_bethe:  # pre-calculation/caching for Bethe free energy calculation
        # find theta(canonical parameters; same as log potentials)
        logging.debug('null node configs: ', np.where(phi_vals == 0)[0])
        node_theta = np.log(phi_vals)  # length K; shared by all nodes
        node_theta[node_theta == -np.inf] = 0

        logging.debug('null edge configs: ', np.where(psi_vals == 0)[0])
        edge_theta = np.log(psi_vals)  # length K^2; shared by all edges
        edge_theta[edge_theta == -np.inf] = 0

        # need to keep track of the mus (marginals)
        # we use a 2D array (NxN); the main diagonals store node marginals, others store edge
        # marginals; (we only use the upper-triangular part)
        Marg = [[0 for _ in range(N)] for _ in range(N)]

    for t in range(its):
        for i in graph.Vs:
            for j in graph.Nbs[i]:
                M[i][j] = calc_message(graph, phi, psi, M, i, j)  # always use the same message bank

        if np.sum(np.abs(M - M_prev)) < 1e-4:
            print("convergence after " + str(t) + " iters")
            break
        elif t + 1 == its:
            print("no convergence after " + str(its) + " iters")
            break

        M_prev = M.copy()

        print("iter: ", t)

        # calculate marginals on nodes p(x_i)
        print("Node marginal probabilities:")
        for v in graph.Vs:
            nbr_msg = np.product(M[:, v], axis=0)  # a vector containing the entry-wise product of neighbor messages
            marg = phi_vals * nbr_msg  # product of self-potential and messages
            marg = marg / marg.sum()  # normalize
            if calc_bethe:
                Marg[v][v] = marg
            print(v, marg)

        # calculate marginals on edges p(x_i,x_j)
        print("Edge marginal probabilities:")
        for i in graph.Vs:
            for j in graph.Nbs[i]:
                if i > j:   continue  # avoid calculating the same edge marginal twice
                i_nbr_msgs = [M[n][i] for n in range(N) if n != j]
                i_nbr_msg = np.product(i_nbr_msgs, axis=0)  # the product of messages sent to i from all of its children

                j_nbr_msgs = [M[n][j] for n in range(N) if n != i]
                j_nbr_msg = np.product(j_nbr_msgs, axis=0)  # the product of messages sent to j from all of its children

                # Cartesian product of i_nbr_msg and j_nbr_msg, (K^2) x 2 matrix
                nbr_msgs = np.array([(i, j) for i in i_nbr_msg for j in j_nbr_msg])
                nbr_msg = np.product(nbr_msgs, axis=1)

                marg = edge_potential_prods * nbr_msg
                marg /= marg.sum()  # normalize

                if calc_bethe:
                    Marg[i][j] = marg
                print(i, j, marg)

        if calc_bethe:
            # calculate the inner product of theta (canonical params) and mu (marginals)
            inner_prod = 0
            for v in graph.Vs:  # i.e. in range(N)
                inner_prod += node_theta.dot(Marg[v][v])
            for i in graph.Vs:
                for j in graph.Nbs[i]:
                    if i > j:   continue  # only using upper triangular of Marg
                    inner_prod += edge_theta.dot(Marg[i][j])

            # calculate Bethe (tree) approximate entropy
            ent = 0
            for v in graph.Vs:  # node entropy
                logging.debug('null node marg: ', np.where(Marg[v][v] == 0)[0])
                log_v = np.log(Marg[v][v])
                log_v[log_v == -np.inf] = 0
                ent -= log_v.dot(Marg[v][v])
            for i in graph.Vs:  # edge mutual information
                for j in graph.Nbs[i]:
                    if i > j:   continue  # only using upper triangular of Marg

                    # product of node marginals for all x_i,x_j configs (denominator in mutual information)
                    marg_prods = np.array([u * v for u in Marg[i][i] for v in Marg[j][j]])
                    logging.debug('null node marg prod: ', np.where(marg_prods == 0)[0])
                    logging.debug('null edge marg: ', np.where(Marg[i][j] == 0)[0])

                    log_ij = np.log(Marg[i][j]) - np.log(marg_prods)
                    log_ij[log_ij == -np.inf] = 0
                    ent -= log_ij.dot(Marg[i][j])

            # Bethe free energy; should approach the log partition function; something's terribly wrong if it's negative
            bfe = inner_prod + ent
            print('BFE =', bfe)

    return bfe if calc_bethe else 0
