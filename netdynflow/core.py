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
Calculation of dynamic communicability and flow
===============================================

This module contains functions to calculate the temporal evolution of the
dynamic communicability and flow. Networks are treated as connectevity
matrices, represented as 2D NumPy arrays. Given a connectivity matrix, dynamic
communicability and flow return a series of matrices arranged into a tensor
(a numpy array of rank-3), each describing the state of the network at
consecutive time points.

Generation of main tensors
--------------------------
DynCom
    Returns the temporal evolution of a network's dynamic communicability.
DynFlow
    Returns the extrinsinc flow on a network over time for a given input.
FullFlow
    Returns the complete flow on a network over time for a given input.
IntrinsicFlow
    Returns the intrinsic flow on a network over time for a given input.

Reference and Citation
----------------------
1. M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).
2. M. Gilson, N. Kouvaris, et al. "Analysis of brain network dynamics estimated
from fMRI data: A new framework based on communicability and flow"
bioRxiv (2018). DOI: https://doi.org/10.1101/421883.


...moduleauthor:: Gorka Zamora-Lopez <galib@zamora-lopez.xyz>

"""
from __future__ import division, print_function

import numpy as np
import numpy.linalg
import scipy.linalg

__all__ = ['JaccobianMOU', 'DynCom', 'DynFlow', 'FullFlow', 'IntrinsicFlow']


## THE MAIN TENSORS ##########################################################
def DynCom(con_matrix, tau_const, tmax=20, timestep=0.1, normed=True):
    """Returns the temporal evolution of a network's dynamic communicability.

    Parameters
    ----------
    con_matrix : ndarray of rank-2
        The adjacency matrix of the network.
    tau_const : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning: Not an integration step, just the desired sampling rate.
    normed : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    dyncom_tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape (tmax*timestep) x N x N, where N is the number of nodes.
    """
    assert len(con_matrix.shape) == 2 and con_matrix.shape[0] == con_matrix.shape[1], "con_matrix should be a square matrix"
    n_nodes = con_matrix.shape[0]

    # INFOS HERE
    dyncom_tensor = DynFlow(con_matrix, tau_const, np.eye(n_nodes, dtype=np.float), tmax, timestep, normed)
    return dyncom_tensor

def DynFlow(con_matrix, tau_const, sigma_mat, tmax=20, timestep=0.1, normed=True, type='extrinsic'):
    # CONVERT THIS FUNCTION INTO A gen_dyn_tensor() FUNCTION, EXTRACT THE
    # REST OF FUNCTIONS FROM THIS ONE !!
    """Returns the extrinsinc flow on a network over time for a given input.

    Parameters
    ----------
    con_matrix : ndarray of rank-2
        The adjacency matrix of the network.
    tau_const : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma_mat : ndarray of rank-2
        The covariance matrix of fluctuating inputs.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning: Not an integration step, just the desired sampling rate.
    normed : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    flow_tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape (tmax*timestep) x n_nodes x n_nodes, where n_nodes is the number of nodes.
    """
    # 0) SECURITY CHECKS
    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep > tmax: raise ValueError("Incompatible values, timestep < tmax given")
    assert len(con_matrix.shape) == 2 and con_matrix.shape[0] == con_matrix.shape[1], "con_matrix should be a square matrix"
    assert np.shape(con_matrix) == np.shape(sigma_mat), "Connectivity and covariance matrices not aligned."
    con_matrix = con_matrix.astype(float)

    # Number of nodes
    n_nodes = con_matrix.shape[0]

    # 1) CALCULATE THE DYNAMIC COMMUNICABILITY
    # 1.1) Define the Jacobian matrix
    # The matrix jacobian_diag is diagonal with elements related to the time constant(s)
    if np.shape(tau_const):
        # In case tau_const was an array-like data with 1 dimension
        assert type(tau_const) == numpy.ndarray and len(tau_const.shape) == 1, "tau_const should be a float or 1D array of floats"
        assert tau_const.shape[0] == n_nodes, "Data not aligned. 'con_matrix and tau_const not of same length"
        jacobian_diag = -np.ones(n_nodes, dtype=np.float) / tau_const
    else:
        # In case tau_const was just a number
        jacobian_diag = -np.ones(n_nodes, dtype=np.float) / tau_const
    scaling_factor = (-1./jacobian_diag).sum()

    # The Jacobian matrix is jacobian_diag on the diagonal and the connectivity elsewhere
    jacobian = np.diag(jacobian_diag) + con_matrix

    # 1.2) Calculate the extrinsic flow over integration time
    # number of discrete time steps
    n_t = int(tmax / timestep) + 1
    # Calculate the matrix square root of the symmetric matrix sigma_mat
    sigma_sqrt_mat = scipy.linalg.sqrtm(sigma_mat)
    flow_tensor = np.zeros((n_t,n_nodes,n_nodes), dtype=np.float)
    for i_t in range(n_t):
        t = i_t * timestep
        # Calculate the term for jacobian_diag without using expm(), to speed up
        jacobian_diag_t = np.diag( np.exp(jacobian_diag * t) )
        # Calculate the dynamic communicability at time t
        flow_tensor[i_t] = np.dot( sigma_mat, (scipy.linalg.expm(jacobian * t) - jacobian_diag_t) )

    # 1.3) Normalise by the scaling factor
    if normed:
        flow_tensor /= scaling_factor

    return flow_tensor


def IntrinsicFlow(con_matrix, tau_const, sigma_mat, tmax=20, timestep=0.1, normed=True):
    """Returns the intrinsic flow on a network over time for a given input.

    Parameters
    ----------
    con_matrix : ndarray of rank-2
        The adjacency matrix of the network.
    tau_const : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma_mat : ndarray of rank-2
        The matrix of Gaussian noise covariances.
    tmax : real valued number, positive (optional)
        Final time for integration.dyncom
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning: Not an integration step, just the desired sampling rate.
    normed : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    flow_tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape (tmax*timestep) x N x N, where N is the number of nodes.
    """
    # 0) SECURITY CHECKS
    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep > tmax: raise ValueError("Incompatible values, timestep < tmax given")
    assert len(con_matrix.shape) == 2 and con_matrix.shape[0] == con_matrix.shape[1], "con_matrix should be a square matrix"
    assert np.shape(con_matrix) == np.shape(sigma_mat), "Connectivity and covariance matrices not aligned."
    con_matrix = con_matrix.astype(float)

    # Number of nodes
    n_nodes = con_matrix.shape[0]

    # 1) CALCULATE THE DYNAMIC COMMUNICABILITY
    # 1.1) Define the Jacobian matrix
    # The matrix jacobian_diag is diagonal with elements related to the time constant(s)
    if np.shape(tau_const):
        # In case tau_const was an array-like data with 1 dimension
        assert type(tau_const) == numpy.ndarray and len(tau_const.shape) == 1, "tau_const should be a float or 1D array of floats"
        assert tau_const.shape[0] == n_nodes, "Data not aligned. con_matrix and tau_const not of same length"
        jacobian_diag = -np.ones(n_nodes, dtype=np.float) / tau_const
    else:
        # In case tau_const was just a number
        jacobian_diag = -np.ones(n_nodes, dtype=np.float) / tau_const
    scaling_factor = (-1./jacobian_diag).sum()

    # The Jacobian matrix is jacobian_diag on the diagonal and the connectivity elsewhere
    jacobian = np.diag(jacobian_diag) + con_matrix

    # 1.2) Calculate the extrinsic flow over integration time
    # number of discrete time steps
    n_t = int(tmax / timestep) + 1
    # Calculate the matrix square root of the symmetric matrix sigma_mat
    sigma_sqrt_mat = scipy.linalg.sqrtm(sigma_mat)
    flow_tensor = np.zeros((n_t,n_nodes,n_nodes), dtype=np.float)
    for i_t in range(n_t):
        t = i_t * timestep
        # Calculate the term for jacobian_diag without using expm(), to speed up
        jacobian_diag_t = np.diag( np.exp(jacobian_diag * t) )
        # Calculate the dynamic communicability at time t.
        flow_tensor[i_t] = np.dot( sigma_mat, jacobian_diag_t)

    # 1.3) Normalise by the scaling factor
    if normed:
        flow_tensor /= scaling_factor

    return flow_tensor


def FullFlow(con_matrix, tau_const, sigma_mat, tmax=20, timestep=0.1,
                                            normed=True):
    """Returns the complete flow on a network over time for a given input.

    Parameters
    ----------
    con_matrix : ndarray of rank-2
        The adjacency matrix of the network.
    tau_const : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma_mat : ndarray of rank-2
        The matrix of Gaussian noise covariances.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step. NOT an integration step, but the sampling step.
    normed : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    flow_tensor : ndarray of rank-3
        Temporal evolution of the network's flow. A tensor of shape
        (tmax*timestep) x N x N, where N is the number of nodes.
    """
    flow_tensor = DynFlow(con_matrix, tau_const, np.eye(n_nodes), tmax, timestep, normed) \
               + IntrinsicFlow(con_matrix, tau_const, np.eye(n_nodes), tmax, timestep, normed)

    return flow_tensor


##
