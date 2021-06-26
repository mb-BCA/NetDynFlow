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
def TimeToPeak(tensor, timestep):
    """Returns the time at which links reach peak communicability.

    Write more here ...  TTP is the time the response on node j take to reach
    peak, after a perturbation at node i.

    NOTE: See if you can use the same function for both the node evolution and
    the link evolution, which are of different shape. I think you can achieve
    that by simply setting 'axis=-1' to the argmax() function.

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
    ttp_matrix = tensor.argmax(axis=-1)

    # Convert into time
    tpoints = timestep * np.arange(nt, dtype=np.float)
    ttp_matrix = tpoints[ttp_matrix]

    # Calculate the average time-to-peak
    average_ttp = (ttp_matrix.sum() - ttp_matrix.trace()) / (n1*(n1-1))

    return (ttp_matrix, average_ttp)




##
