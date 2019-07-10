# -*- coding: utf-8 -*-
# Copyright (c) 2019, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris
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

This module contains functions to analyse the dynamic communicability or flow
from a given network to extract information of its structure and function.

Metrics derived from the tensors
--------------------------------
TotalEvolution
    Calculates total communicability or flow over time for a network.
NodeEvolution
    Temporal evolution of all nodes' input and output communicability or flow.
Diversity
    Temporal diversity for a networks dynamic communicability or flow.
TTPdistance
    Pair-wise node distance, measured as the time-to-peak of their interaction.
    TO BE WRITTEN AND ADDED !! I NEED TO INCLUD THE TEMPORAL RESOLUTION

Reference and Citation
----------------------
1. M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).
2. M. Gilson, N. Kouvaris, et al. "Analysis of brain network dynamics estimated
from fMRI data: A new framework based on communicability and flow"
bioRxiv (2018). DOI: https://doi.org/10.1101/421883.
"""
from __future__ import division, print_function

# Q: Shall these go here, in the module file, or in the __init__,py file?
# ... or even somewhere else?
__author__ = "Gorka Zamora-Lopez, Mattheiu Gilson and Nikos Kouvaris"
__email__ = "galib@Zamora-Lopez.xyz"
__copyright__ = "Copyright 2018"
__license__ = "GPL"
__update__="18/09/2018"
__version__="0.0.1.dev0"

import numpy as np
import numpy.linalg
import scipy.linalg


## METRICS FROM THE TENSORS ###################################################
def TotalEvolution(dyntensor):
    """Calculates total communicability or flow over time for a network.

    Parameters
    ----------
    dyntensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape timesteps x N x N, where N is the number of nodes.

    Returns
    -------
    totaldyncom : ndarray of rank-1
        Array containing temporal evolution of the total communicability.
    """
    # 0) SECURITY CHECKS
    tensorshape = np.shape(dyntensor)
    assert len(tensorshape) == 3, 'Input not aligned. Tensor of rank-3 expected'
    nsteps, N1, N2 = tensorshape
    assert N1 == N2, 'Input not aligned. Shape (nsteps x N x N) expected'

    totaldyncom = dyntensor.sum(axis=1).sum(axis=1)

    return totaldyncom

def NodeEvolution(dyntensor, directed=False):
    """Temporal evolution of all nodes' input and output communicability or flow.

    Parameters
    ----------
    dyntensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape timesteps x N x N, where N is the number of nodes.

    Returns
    -------
    nodedyncom : ndarray of rank-2 or tuple.
        Temporal evolution of communicability of each node. Array of shape
        (N x timesteps). If 'directed=True', 'nodedyncom' is an array of
        length two, with the input and the output communi
    """
    # 0) SECURITY CHECKS
    tensorshape = np.shape(dyntensor)
    assert len(tensorshape) == 3, 'Input not aligned. Tensor of rank-3 expected'
    nsteps, N1, N2 = tensorshape
    assert N1 == N2, 'Input not aligned. Shape (nsteps x N x N) expected'

    # 1) Calculate the input and output node properties
    innodedyn = dyntensor.sum(axis=1).T
    outnodedyn = dyntensor.sum(axis=2).T
    nodedyn = ( innodedyn, outnodedyn )

    return nodedyn

def Diversity(dyntensor):
    """Temporal diversity for a networks dynamic communicability or flow.

    Parameters
    ----------
    dyntensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability or flow. A
        tensor of shape timesteps x N x N, where N is the number of nodes.

    Returns
    -------
    diversity : ndarray of rank-1
        Array containing temporal evolution of the diversity.
    """
    # 0) SECURITY CHECKS
    tensorshape = np.shape(dyntensor)
    assert len(tensorshape) == 3, 'Input not aligned. Tensor of rank-3 expected'
    nsteps, N1, N2 = tensorshape
    assert N1 == N2, 'Input not aligned. Shape (nsteps x N x N) expected'

    diversity = np.zeros(nsteps, np.float)
    for t in range(nsteps):
        diversity[t] = dyntensor[t].std() / dyntensor[t].mean()

    return diversity



##
