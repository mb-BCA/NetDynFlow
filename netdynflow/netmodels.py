# -*- coding: utf-8 -*-
# Copyright (c) 2022, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris
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
These functions are all imported from the GAlib library (https://github.com/gorkazl/pyGAlib)
Please see doctsring of module "galib.models" for a list of functions.  ::

    >>> import galib
    >>> help(galib.models)

Surrogates for weighted networks
--------------------------------
ShuffleLinkWeights
    Randomly re-allocates the weights associated to the links.

RandomiseWeightedNetwork
    Randomises a connectivity matrix and its weights.

Spatially embedded (weighted) networks
--------------------------------------
SpatialWeightSorting
    Sorts the link weights of a network by the spatial distance between nodes.
SpatialLatticeFromNetwork
    Generates spatial weighted lattices with same weights as `con`.


"""

# Standard library imports

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
    newcon : ndarray of rank-2 and shape (N x N).
        A connectivity matrix with links between same nodes as `con` but the
        link weights shuffled.

    """
    # 0) SECURITY CHECKS
    if not type(con) == numpy.ndarray:
        raise TypeError( "Please enter the connectivity matrix as a numpy array." )
    con_shape = np.shape(con)
    if (len(con_shape) != 2) or (con_shape[0] != con_shape[1]):
        raise ValueError( "Input not aligned. 'con' should be a 2D array of shape (N x N)." )

    # 1) EXTRACT THE CONSTRAINTS FROM THE con MATRIX
    N = con_shape[0]
    nzidx = con.nonzero()
    weights = con[nzidx]

    # 2) GENERATE THE NEW NETWORK WITH THE WEIGHTS SHUFFLED
    numpy.random.shuffle(weights)
    newcon = np.zeros((N,N), dtype=con.dtype)
    newcon[nzidx] = weights

    return newcon

@jit
def RandomiseWeightedNetwork(con):
    """
    Randomises a connectivity matrix and its weights.

    Returns a random connectivity matrix (Erdos-Renyi-type) with the same number
    of links and same link weights as the input matrix `con`. Therefore, both
    the total weight (sum of link weights) and the distribution of link weights
    are conserved, but the input/output degrees of the nodes, or their individual
    strengths, are not conserved.

    The function identifies some properties of `con` in order to conserve
    elementary properties of `con`. For example:
    (1) The resulting random weighted network will only contain self-connections
    (non-zero diagonal entries) if `con` contains self-connections.
    (2) If `con` is an unweighted adjacency matrix (directed or undirected), the
    result is an Erdos-Renyi-type random graph (directed or undirected),
    of same size and number of links as `con`.
    (3) If `con` is an undirected network but contains asymmetric link weights,
    the result will be an undirected random graph with asymmetric weights.
    (4) If `con` is a directed weighted network, the result will be a directed
    and weighted network. In this case, weights cannot be symmetric.

    Parameters
    ----------
    con : ndarray
        Adjacency matrix of the (weighted) network.

    Returns
    -------
    newcon : ndarray of rank-2 and shape (N x N)
        A connectivity matrix with links between same nodes as `con` but the
        link weights shuffled.

    """
    # 0) SECURITY CHECKS
    if not type(con) == numpy.ndarray:
        raise TypeError( "Please enter the connectivity matrix as a numpy array." )
    con_shape = np.shape(con)
    if (len(con_shape) != 2) or (con_shape[0] != con_shape[1]):
        raise ValueError( "Input not aligned. 'con' should be a 2D array of shape (N x N)." )

    # 1) EXTRACT INFORMATION NEEDED FROM THE con MATRIX
    N = con_shape[0]

    # Find out whether con is symmetric
    if abs(con - con.T).sum() == 0:
        symmetric = True
    else:
        symmetric = False

    # Find out whether con is directed and calculate the number of links
    if Reciprocity(con) == 1.0:
        directed = False
        L = int( round(0.5*con.astype(bool).sum()) )
    else:
        directed = True
        L = con.astype(bool).sum()

    # Find out whether `con` allows self-loops (non-zero diagonal elements)
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
    # Initialise the matrix. Give same dtype as `con`
    newcon = np.zeros((N,N), dtype=con.dtype)

    # Shuffle the list of weights
    numpy.random.shuffle(weights)

    # Finally, add the links at random
    counter = 0
    while counter < L:
        # 2.1) Pick up two nodes at random
        source = int(N * numpy.random.rand())
        target = int(N * numpy.random.rand())

        # 2.2) Check if they can be linked, otherwise look for another pair
        if newcon[source,target]: continue
        if source == target and not selfloops: continue

        # 2.3) Perform the rewiring
        newcon[source,target] = weights[counter]
        if not directed and symmetric:
            newcon[target,source] = weights[counter]
        elif not directed and not symmetric:
            newcon[target,source] = weights[-(counter+1)]
        counter += 1

    return newcon


## SPATIALLY EMBEDDED SURROGATES ###############################################
def SpatialWeightSorting(con, distmat, descending=True):
    """Sorts the link weights of a network by the spatial distance between nodes.

    The function reads the weights from a connectivity matrix and re-allocates
    them according to the euclidean distance between the nodes. The sorting
    conserves the position of the links, therefore, if `con` is a binary graph,
    the function will return a copy of `con`. The distance between nodes shall
    be given as input `distmat`.

    If descending = True, the larger weigths are assigned to the links between
    closer nodes, and the smaller weights to the links between distant nodes.

    If descending = False, the larger weights are assigned to the links between
    distant nodes, and the smaller weights to links between close nodes.

    Parameters
    ----------
    con : ndarray, rank-2.
        Adjacency matrix of the (weighted) network.
    distmat : ndarray, rank-2.
        A matrix containing the spatial distance between all pair of ROIs.
        This can be either the euclidean distance, the fiber length or any
        other geometric distance.
    descending : boolean, optional.
        Determines whether links weights are assigend in descending or in
        ascending order, according to the euclidean distance between the nodes.

    Returns
    -------
    newcon : ndarray of rank-2 and shape (N x N).
        Connectivity matrix with weights sorted according to spatial distance
        between the nodes.

    """
    # 0) SECURITY CHECKS
    con_shape = np.shape(con)
    dist_shape = np.shape(distmat)
    if con_shape != dist_shape:
        raise ValueError( "Data not aligned. 'con' and 'distmat' of same shape expectted. " )

    # 1) EXTRACT THE NEEDED INFORMATION FROM THE con MATRIX
    N = len(con)
    # The indices of the links and their weights, distance
    nzidx = con.nonzero()
    weights = con[nzidx]
    distances = distmat[nzidx]

    # 2) SORT THE WEIGHTS IN DESCENDING ORDER
    weights.sort()
    if descending:
        weights = weights[::-1]

    # Get the indices that would sort the links by distance
    sortdistidx = distances.argsort()
    newidx = (nzidx[0][sortdistidx], nzidx[1][sortdistidx])

    # 3) CREATE THE NEW CONNECTIVITY WITH THE LINK WEIGHTS SORTED SPATIALLY
    newcon = np.zeros((N,N), np.float)
    newcon[newidx] = weights

    return newcon

def SpatialLatticeFromNetwork(con, distmat, descending=True):
    """Generates spatial weighted lattices with same weights as `con`.

    The function reads the weights from a connectivity matrix and generates a
    spatially embedded weighted lattice, assigning the largest weights in
    descending order to the nodes that are closer from each other. Therefore,
    it requires also the euclidean distance between the nodes is given as input.

    If `con` is a binary graph of L links, the function returns a graph with
    links between the L spatially closest pairs of nodes.

    If `descending = True`, the larger weigths are assigned to the links between
    closer nodes, and the smaller weights to the links between distant nodes.

    If `descending = False`, the larger weights are assigned to the links between
    distant nodes, and the smaller weights to links between close nodes.

    Note
    ----
    Even if `con` is either a directed network or undirected but with asymmetric
    weights, the resulting lattice will be undirected and (quasi-)symmetric
    due to the fact that the spatial distance between two nodes is symmetric.

    Parameters
    ----------
    con : ndarray, rank-2.
        Adjacency matrix of the (weighted) network.
    distmat : ndarray, rank-2.
        A matrix containing the spatial distance between all pair of ROIs.
        This can be either the euclidean distance, the fiber length or any
        other geometric distance.
    descending : boolean, optional.
        Determines whether links weights are assigend in descending or in
        ascending order, according to the euclidean distance between the nodes.

    Returns
    -------
    newcon : ndarray of rank-2 and shape (N x N).
        Connectivity matrix of a weighted lattice.

    """
    # 0) SECURITY CHECKS
    con_shape = np.shape(con)
    dist_shape = np.shape(distmat)
    if con_shape != dist_shape:
        raise ValueError( "Data not aligned. 'con' and 'distmat' of same shape expectted. " )

    # 1) EXTRACT THE NEEDED INFORMATION FROM THE con MATRIX
    N = len(con)

    # Sort the weights of the network
    weights = con.flatten()
    weights.sort()
    if descending:
        weights = weights[::-1]

    # Find the indices that sort the euclidean distances, from shorter to longer
    if descending:
        distmat[np.diag_indices(N)] = np.inf
    else:
        distmat[np.diag_indices(N)] = 0.0
    distances = distmat.ravel()
    sortdistidx = distances.argsort()
    newidx = np.unravel_index( sortdistidx, (N,N) )

    # And finally, create the coonectivity matrix with the weights sorted
    newcon = np.zeros((N,N), np.float)
    newcon[newidx] = weights

    return newcon



##
