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
Network and surrogate generation module
=======================================

Functions to construct synthetic networks and generate surrogates out of given
binary or weighted networks.

Generation and randomization of binary graphs
---------------------------------------------
These functions are all imported from the GAlib library. Please see doctsring
of module 'galib.models' for further documentation and a list of functions.  ::

    >>> import galib
    >>> help(galib.models)

Deterministic network models
----------------------------
Function1
    Short description of the function.
Function2
    Short description of the function.

Random network models
----------------------
Function1
    Short description of the function.
Function2
    Short description of the function.

Weighted surrogate networks
---------------------------
ShuffleLinkWeights
    Randomly re-allocates the weights associated to the links.

"""
# Standard library imports
from __future__ import division, print_function
# Third party packages
import numpy as np
import numpy.linalg
import scipy.linalg
from numba import jit
# Import GAlib for graph analysis and graph generation tools
### ACHTUNG!! For now I am importing GAlib but, for the long run, we must
### decide whether we want GAlib as a dependency of NetDynFlow, or we
### prefer to copy/paste the useful functions from GAlib into this module and
### therefore make NetDynFlow independent of GAlib.
import galib
from galib.models import*


## DETERMINISTIC GRAPH MODELS ##################################################
# NOTE: See GAlib.models package


## RANDOM GRAPH MODELS #########################################################
# NOTE: See GAlib.models package


## GENERATION OF SURROGATE NETWORKS ############################################
def ShuffleLinkWeights(con):
    """
    Randomly re-allocates the link weights of an input network.

    The function does not alter the position of the links, it only shuffles
    the weights associated to the links. Therefore, the binarised version
    is preserved.

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
    con_shape = np.shape(con)
    if (len(con_shape) != 2) or (con_shape[0] != con_shape[1]):
        raise ValueError("Input not aligned. 'con' should be a 2D array of shape (N x N).")

    # 1) EXTRACT THE CONSTRAINTS FROM THE con MATRIX
    N = con_shape[0]
    nzidx = con.nonzero()
    weights = con[nzidx]

    # 2) GENERATE THE NEW NETWORK WITH THE WEIGHTS SHUFFLED
    numpy.random.shuffle(weights)
    rewmatrix = np.zeros((N,N), dtype=con.dtype)
    rewmatrix[nzidx] = weights

    return rewmatrix

@jit
def RandomiseWeightedNetwork(con):
    """
    Randomises a connectivity matrix and its weights.

    Returns a random connectivity matrix with the same number of links and
    same link weights as the input matrix 'con'. Therefore, both the total
    weight and the link weight distribution are conserved but the input/output
    degrees and node strengths are not.

    The function identifies some properties of 'con' in order to conserve
    elementary properties of 'con'. For example:
    1) The resulting random weighted network will only contain self-connections
    (non-zero diagonal entries) if 'con' contains self-connections.
    2) If 'con' is an unweighted adjacency matrix (directed or undirected), the
    result is an Erdos-Renyi-type random graph (directed or undirected),
    of same size and number of links as 'con'.
    3) If 'con' is an undirected network but contains asymmetric link weights,
    the result will be an undirected random graph with asymmetric weights.
    4) If 'con is a directed weighted network, the result will be a directed and
    weighted network. In this case, weights cannot be symmetric.

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
    con_shape = np.shape(con)
    if (len(con_shape) != 2) or (con_shape[0] != con_shape[1]):
        raise ValueError("Input not aligned. 'con' should be a 2D array of shape (N x N).")

    # 1) EXTRACT INFORMATION NEEDED FROM THE con MATRIX
    N = con_shape[0]

    # Find out whether 'con' is symmetric
    if abs(con - con.T).sum() == 0:
        symmetric = True
    else:
        symmetric = False

    # Find out whether 'con' is directed and calculate the number of links
    if Reciprocity(con) == 1.0:
        directed = False
        L = int( round(0.5*con.astype(bool).sum()) )
    else:
        directed = True
        L = con.astype(bool).sum()

    # Find out whether 'con' allows self-loops (non-zero diagonal elements)
    if con.trace() == 0:
        selfloops = False
    else:
        selfloops = True

    # Get the weights, as a 1D array
    if symmetric:
        nzidx = np.triu(con, k=1).nonzero()
        weights = con[nzidx]
    else:
        nzidx = con.nonzero()
        weights = con[nzidx]

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

        # 2.3) Perform the rewiring
        rewcon[source,target] = weights[counter]
        if not directed and symmetric:
            rewcon[target,source] = weights[counter]
        elif not directed and not symmetric:
            rewcon[target,source] = weights[-(counter+1)]
        counter += 1

    return rewcon




##
