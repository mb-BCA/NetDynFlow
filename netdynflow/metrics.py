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

Functions to analyse the dynamic communicability or flow "tensors", which have
been previously calculated from a given network.

Metrics derived from the tensors
--------------------------------
TotalEvolution
    Calculates total communicability or flow over time for a network.
NodeEvolution
    Temporal evolution of all nodes' input and output communicability or flow.
Diversity
    Temporal diversity for a networks dynamic communicability or flow.
TimeToPeak
    The time links, nodes or networks need to reach peak flow.


Reference and Citation
----------------------
1. M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).
2. M. Gilson, N. Kouvaris, et al. "Network analysis of whole-brain fMRI
dynamics: A new framework based on dynamic communicability" NeuroImage 201,
116007 (2019).

"""
from __future__ import division, print_function

import numpy as np
import numpy.linalg
import scipy.linalg


## METRICS EXTRACTED FROM THE FLOW AND COMMUNICABILITY TENSORS ################
def TotalEvolution(tensor):
    """
    Calculates total communicability or flow over time for a network.

    Parameters
    ----------
    tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape n_nodes x n_nodes x timesteps , where n_nodes is the number of nodes.

    Returns
    -------
    totaldyncom : ndarray of rank-1
        Array containing temporal evolution of the total communicability.
    """

    # 0) SECURITY CHECKS
    tensor_shape = np.shape(tensor)
    assert len(tensor_shape) == 3, 'Input not aligned. Tensor of rank-3 expected'
    n1, n2, n_t = tensor_shape
    assert n1 == n2, 'Input not aligned. Shape (n_nodes x n_nodes x n_t) expected'

    totaldyncom = tensor.sum(axis=(0,1))

    return totaldyncom

def NodeEvolution(tensor, directed=False):
    """
    Temporal evolution of all nodes' input and output communicability or flow.

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
    """
    Temporal diversity for a networks dynamic communicability or flow.

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

def TimeToPeak(arr, timestep):
    """
    The time links, nodes or networks need to reach peak flow.

    In terms of binary graphs, time-to-peak is equivalen to the pathlength
    between two nodes.

    The function calculates the time-to-peak for either links, nodes or the
    whole network, depending on the input array given.
    - If 'arr' is the (N x N x nt) tensor flow, the output 'ttp_arr' will be
    an N x N matrix with the ttp between every pair of nodes.
    - If 'arr' is the (N x nt) temporal flow of the N nodes, the output
    'ttp_arr' will be an array of length N, containing the ttp of the N nodes.
    - If 'arr' is the array of length nt for the network flow, then 'ttp_arr'
    will be a scalar, indicating the time at which whole-network flow peaks.

    Parameters
    ----------
    arr : ndarray of adaptive shape, according to the case.
        Temporal evolution of the flow. An array of shape N X N X nt for the
        flow of the links, an array of shape N X nt for the flow of the nodes,
        or a 1-dimensional array of length nt for the network flow.
    timestep : real valued number.
        Sampling time-step. This has to be the time-step employed to simulate
        the temporal evolution encoded in 'arr'.

    Returns
    -------
    ttp_arr : ndarray of rank-2
        The time(s) taken for links, nodes or the network to reach peak flow.
        Output shape depends on input.
    """

    # 0) SECURITY CHECKS
    ## TODO: Write a check to verify the curve has a real peak and decays after
    ## the peak. Raise a warning that maybe longer simulation is needed.
    arr_shape = np.shape(arr)
    if arr_shape==3:
        assert arr_shape[0] == arr_shape[1], \
            'Input not aligned. Shape (n_nodes x n_nodes x n_time) expected'

    # 1) Get the indices at which every element peaks
    ttp_arr = arr.argmax(axis=-1)
    # 2) Convert into simulation time
    ttp_arr = timestep * ttp_arr

    return ttp_arr


##
