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
Analysis of dynamic communicability and flow
============================================
Functions in testing version, before they are ported to their corresponding
module for 'official' release into the package.

"""
# Standard library imports
from __future__ import division, print_function
# Third party packages
import numpy as np
import numpy.linalg
import scipy.linalg
from numba import jit


## METRICS EXTRACTED FROM THE FLOW AND COMMUNICABILITY TENSORS ################
def NNt2tNN(signals):
    """This function transposes a 3D array from shape (N,N,nt) to (nt,N,N),
    where nt might be the number of time-points (or samples) and N = the number
    of nodes (or features). It exists basically because I wil constantly
    forget the right order of the axes need.

    WARNING! Remind that np.transpose() function returns a view of the array,
    not a copy! If you want a copy with the entries properly sorted in memory,
    you will need to call the copy explicitely, e.g:

    arr2 = np.copy( tNN2NNt(arr1), order='C' )

    """
    ## TODO:
    ## 1. Check namings to be coherent with rest of the package.
    ## 2. Write the opposit function tNN2NNt()
    ## 3. Properly document both functions.
    ## 4. Move them to the tools.py module.

    # Security checks
    assert len(np.shape(signals)) == 3, "3D array required."
    n0, n1, n2 = np.shape(signals)
    if n0 != n1:
        raise TypeError("3D array of shape (N,N,nt) required.")

    # Transpose the array
    newsignals = np.transpose(signals, axes=(2,0,1))
    return newsignals


## RANDOMIZATION OF (WEIGHTED) NETWORKS ########################################
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
    ## Deprecated function. Not desirable output. See the docstring.
    """
    Randomises a (weighted) connectivity matrix.

    The function returns a random connectivity matrix with the same number of
    links as the input matrix.
    ACHTUNG!!! However, the weights are always randomised such
    that they are asymmetric. Therefore, if 'con' is symmetric weighted matrix,
    the function returns an undirected underlying graph, but with the weights
    asymmetric. So, I don't like it.

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
