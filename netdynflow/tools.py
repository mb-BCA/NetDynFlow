# -*- coding: utf-8 -*-
# Copyright (c) 2021, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris
# <galib@Zamora-Lopez.xyz>
#
# Released under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Miscelaneous tools and helpers
==============================
Several extra functions to help in the analysis of networks using the dynamic
communicability or flow
Randomization of (weighted) networks
-------------------------------------
RewireLinkWeights
    Randomly re-allocates the link weights of an input network.
RandomiseWeightedNetwork:
    Randomises a (weighted) connectivity matrix.
!!!!!!!!!!!!
GORKA: Shall we may name this module as 'benchmarks' or 'nullmodels'?
!!!!!!!!!!!!
GORKA: I am thinking to import pyGAlib for some utilities and network generation
functions. On the one hand, I don't want to enforce NetDynFlow users to use
pyGAlib, just because it is my library. But on the other hand, I don't think
that copy/pasting functions from pyGAlib into this module is robust. Replicating
functions may lead to bugs. Better to have only one function to do a job and,
if that one is broken, it only needs to be fixed once.
What shall we do?
!!!!!!!!!!!!
"""

import numpy as np
import numpy.random
# WARNING! Numba is not listed in 'requirements.txt'
from numba import jit


## MISCELLANEOUS FUNCTIONS #####################################################
def Reciprocity(adjmatrix):
    """Computes the fraction of reciprocal links to total number of links.
    Both weighted and unweighted input matrices are permitted. Weights
    are ignored for the calculation.
    Parameters
    ----------
    adjmatrix : ndarray of rank-2
        The adjacency matrix of the network.
    Returns
    -------
    reciprocity : float
        A scalar value between 0 (for acyclic directed networks) and 1 (for
        fully reciprocal).
    """
    # 0) PREPARE FOR COMPUTATIONS
    adjmatrix = adjmatrix.astype('bool')

    # 1) COMPUTE THE RECIPROCITY
    L = adjmatrix.sum()
    if L == 0:
        reciprocity = 0
    else:
        # Find the assymmetric links
        # Rest = np.abs(adjmatrix - adjmatrix.T)
        Rest = np.abs(adjmatrix ^ adjmatrix.T)
        Lsingle = 0.5*Rest.sum()
        reciprocity = np.float(L-Lsingle) / L

    return reciprocity


## RANDOMIZATION OF (WEIGHTED) NETWORKS ########################################
def RewireLinkWeights(con):
    """
    Randomly re-allocates the link weights of an input network.

    The function does not alter the position of the links, it only shuffles
    the weights. Thus, if 'con' is an unweighted adjacency matrix, the
    function will simply return a copy of 'con'.
    Parameters
    ----------
    con : ndarray
        Adjacency matrix of the (weighted) network.
    Returns
    -------
    rewmatrix : ndarray
        A connectivity matrix with links between same nodes as 'con' but the
        link weights shuffled.
    """
    # 0) SECURITY CHECKS
    if not type(con) == numpy.ndarray:
        raise TypeError('Please enter the connectivity matrix as a numpy array.')

    # 1) EXTRACT THE NEEDED INFORMATION FROM THE con MATRIX
    N = len(con)
    nzidx = con.nonzero()
    weights = con[nzidx]

    # 2) GENERATE THE NEW NETWORK WITH THE WEIGHTS SHUFFLED
    numpy.random.shuffle(weights)
    rewmatrix = np.zeros((N,N), dtype=con.dtype)
    rewmatrix[nzidx] = weights

    return rewmatrix

@jit
def RandomiseWeightedNetwork1(con):
    # GORKA: This version seems to be faster, with and without Numba.
    # At least, it is never slower
    """
    Randomises a (weighted) connectivity matrix.
    The function returns a random connectivity matrix with the same number of
    links as the input matrix. The resulting connectivity has the same link
    weights of the input matrix (thus total weight is also conserved) but the
    input / output strengths of the nodes are not conserved. If 'con' is an
    unweighted adjacency matrix, the function returns an Erdos-Renyi-like
    random graph, of same size and number of links as 'con'.
    If the binarisation of 'con' is a symmetric matrix, the result will also be
    symmetric. Otherwise, if 'con' represents a directed network, the result
    will be directed.
    !!!!!!!
    GORKA: In the current version, if the underlying graph is undirected but the
    weights are asymmetric, the function won't work. The result will be a
    symmetric matrix and the total weight will likely not be conserved !!!
    !!!!!!!
    Parameters
    ----------
    con : ndarray
        Adjacency matrix of the (weighted) network.
    Returns
    -------
    rewcon : ndarray
        A connectivity matrix with links between same nodes as 'con' but the
        link weights shuffled.
    """
    # 0) SECURITY CHECKS
    if not type(con) == numpy.ndarray:
        raise TypeError('Please enter the connectivity matrix as a numpy array.')

    # 1) EXTRACT THE NEEDED INFORMATION FROM THE con MATRIX
    N = len(con)

    # Get whether 'con' is directed and calculate the number of links
    if Reciprocity(con) == 1.0:
        directed = False
        L = int( round(0.5*con.astype(bool).sum()) )
    else:
        directed = True
        L = con.astype(bool).sum()

    # Get the list of weights
    if directed:
        nzidx = con.nonzero()
        weights = con[nzidx]
    else:
        nzidx = np.triu(con, k=1).nonzero()
        weights = con[nzidx]

    # Get whether 'con' allows self-loops (non-zero diagonal elements)
    if con.trace() == 0:
        selfloops = False
    else:
        selfloops = True

    # 2) GENERATE THE NEW NETWORK WITH THE WEIGHTS SHUFFLED
    # Initialise the matrix. Give same dtype as 'con'
    rewcon = np.zeros((N,N), dtype=con.dtype)

    # Shuffle the list of weights
    numpy.random.shuffle(weights)

    # Finally, add the links at random
    counter = 0
    while counter < L:
        # 2.1) Pick up two nodes at random
        source = int(N * numpy.random.rand())
        target = int(N * numpy.random.rand())

        # 2.2) Check if they can be linked, otherwise look for another pair
        if rewcon[source,target]: continue
        if source == target and not selfloops: continue

        rewcon[source,target] = weights[counter]
        if not directed:
            rewcon[target,source] = weights[counter]

        counter += 1

    return rewcon

@jit
def RandomiseWeightedNetwork2(con):
    """
    Randomises a (weighted) connectivity matrix.
    The function returns a random connectivity matrix with the same number of
    links as the input matrix. The resulting connectivity has the same link
    weights of the input matrix (thus total weight is also conserved) but the
    input / output strengths of the nodes are not conserved. If 'con' is an
    unweighted adjacency matrix, the function returns an Erdos-Renyi-like
    random graph, of same size and number of links as 'con'.
    If the binarisation of 'con' is a symmetric matrix, the result will also be
    symmetric. Otherwise, if 'con' represents a directed network, the result
    will be directed.
    !!!!!!!
    GORKA: In the current version, if the underlying graph is undirected but the
    weights are asymmetric, the function won't work. The result will be a
    symmetric matrix and the total weight will likely not be conserved !!!
    !!!!!!!
    Parameters
    ----------
    con : ndarray
        Adjacency matrix of the (weighted) network.
    Returns
    -------
    rewcon : ndarray
        A connectivity matrix with links between same nodes as 'con' but the
        link weights shuffled.
    """
    # 0) SECURITY CHECKS
    if not type(con) == numpy.ndarray:
        raise TypeError('Please enter the connectivity matrix as a numpy array.')

    # 1) EXTRACT THE NEEDED INFORMATION FROM THE con MATRIX
    N = len(con)
    nzidx = con.nonzero()
    weights = con[nzidx]

    # Get whether 'con' is directed and calculate the number of links
    if Reciprocity(con) == 1.0:
        directed = False
        L = int( round(0.5*con.astype(bool).sum()) )
    else:
        directed = True
        L = con.astype(bool).sum()

    # Get whether 'con' allows self-loops (non-zero diagonal elements)
    if con.trace() == 0:
        selfloops = False
    else:
        selfloops = True

    # 2) GENERATE THE NEW NETWORK WITH THE WEIGHTS SHUFFLED
    # Initialise the matrix. Give same dtype as 'con'
    rewcon = np.zeros((N,N), dtype=con.dtype)

    # Finally, add the links at random
    counter = 0
    while counter < L:
        # 2.1) Pick up two nodes at random
        source = int(N * numpy.random.rand())
        target = int(N * numpy.random.rand())

        # 2.2) Check if they can be linked, otherwise look for another pair
        if rewcon[source,target]: continue
        if source == target and not selfloops: continue

        # 2.3) If the nodes are linkable, place the link
        rewcon[source,target] = 1
        if not directed:
            rewcon[target,source] = 1

        counter += 1

    # Shuffle the list of weights
    numpy.random.shuffle(weights)
    newnzidx = rewcon.nonzero()
    rewcon[newnzidx] = weights

    return rewcon


##
