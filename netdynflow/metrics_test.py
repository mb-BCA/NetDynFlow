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
Functions in testing version, before they are ported to the metrics.py module.

"""
from __future__ import division, print_function

import numpy as np
import numpy.linalg
import scipy.linalg


## METRICS EXTRACTED FROM THE FLOW AND COMMUNICABILITY TENSORS ################
def TTPdistance(tensor, timestep):
    """Returns the time at which links reach peak communicability.

    Write more here ...  TTP is the time the response on node j take to reach
    peak, after a perturbation at node i.

    Parameters
    ----------
    tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape n_nodes x n_nodes x timesteps , where n_nodes is the number of nodes.
    timestep : real valued number.
        Sampling time-step. This has to be the time-step employed to simulate
        the temporal evolution encoded in 'tensor'.

    Returns
    -------
    ttp_matrix : ndarray of rank-2
        An N x N matrix (n = number of nodes) contaning the time-to-peaks for
        all pairs of nodes. TTP is the time the response on node j take to reach
        peak, after a perturbation at node i.
        Analogous to the graph distance matrix in binary graphs.
    average_ttp : real valued.
        The average time-to-peak distance in the network.
        Analogous to the average pathlength of graphs.
    """
    # 0) SECURITY CHECKS
    tensor_shape = np.shape(tensor)
    assert len(tensor_shape) == 3, 'Input not aligned. Tensor of rank-3 expected'
    n1, n2, nt = tensor_shape
    assert n1 == n2, 'Input not aligned. Shape (n_nodes x n_nodes x n_t) expected'

    # Get the indices at which every link peaks
    ttp_matrix = tensor.argmax(axis=2)

    # Convert into time
    tpoints = timestep * arange(nt, dtype=float)
    ttp_matrix = tpoints[ttp_matrix]

    # Calculate the average time-to-peak
    average_ttp = (ttp_matrix.sum() - ttp_matrix.trace()) / (N*(N-1))

    return (ttp_matrix, average_ttp)

def NodeEvolution(tensor, directed=False):
    """Temporal evolution of all nodes' input and output communicability or flow.

    Parameters
    ----------
    tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape n_nodes x n_nodes x timesteps, where n_nodes is the number of nodes.

    Returns
    -------
    nodedyncom : tuple.
        Temporal evolution of the communicability or flow for all nodes.
        The result consists of a tuple of two ndarrays of shape (n_nodes x timesteps)
        each. The first is for the sum of communicability interactions over all
        inputs of each node and the second for its outputs.
    """
    # 0) SECURITY CHECKS
    tensor_shape = np.shape(tensor)
    assert len(tensor_shape) == 3, 'Input not aligned. Tensor of rank-3 expected'
    n1, n2, n_t = tensor_shape
    assert n1 == n2, 'Input not aligned. Shape (n_nodes x n_nodes x timesteps) expected'

    # 1) Calculate the input and output node properties
    innodedyn = tensor.sum(axis=0)
    outnodedyn = tensor.sum(axis=1)
    nodedyn = ( innodedyn, outnodedyn )

    return nodedyn

def Diversity(tensor):
    """Temporal diversity for a networks dynamic communicability or flow.

    Parameters
    ----------
    tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability or flow. A
        tensor of shape N x N x timesteps, where N is the number of nodes.

    Returns
    -------
    diversity : ndarray of rank-1
        Array containing temporal evolution of the diversity.
    """
    # 0) SECURITY CHECKS
    tensor_shape = np.shape(tensor)
    assert len(tensor_shape) == 3, 'Input not aligned. Tensor of rank-3 expected'
    n1, n2, n_t = tensor_shape
    assert n1 == n2, 'Input not aligned. Shape (n_nodes x n_nodes x n_t) expected'

    diversity = np.zeros(n_t, np.float)
    diversity[0] = np.nan
    for i_t in range(1,n_t):
        temp = tensor[:,:,i_t]
        diversity[i_t] = temp.std() / temp.mean()

    return diversity



##
