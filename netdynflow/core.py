# -*- coding: utf-8 -*-
# Copyright (c) 2023, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris
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
pair-wise, conditional flows in a network due to perturbations. The location
and intensity of the initial perturbation can be defined by the users.
Networks are treated as connectivity matrices, represented as 2D NumPy arrays.
Given a connectivity matrix, the subsequent flows are returned as a series of
matrices arranged into a tensor (a numpy array of rank-3), each describing the
state of the network at consecutive time points.

Helper functions
----------------
JacobianMOU
    Calculates the Jacobian matrix for the MOU dynamic system.

Generation of main tensors
--------------------------
CalcTensor
    Generic function to create time evolution of the flows.
DynFlow
    Pair-wise conditional flows on a network over time for a given input.
IntrinsicFlow
    Returns the flow dissipated through each node over time.
FullFlow
    Returns the complete flow on a network over time for a given input.

Reference and Citation
----------------------
1. M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).
2. M. Gilson, N. Kouvaris, et al. "Network analysis of whole-brain fMRI
dynamics: A new framework based on dynamic communicability" NeuroImage 201,
116007 (2019).

"""
# Standard libary imports

# Third party packages
import numpy as np
import numpy.linalg
import scipy.linalg

__all__ = ['JacobianMOU', 'DynFlow', 'FullFlow', 'IntrinsicFlow']


## USEFUL FUNCTIONS ##########################################################
def JacobianMOU(con, tau):
    """Calculates the Jacobian matrix for the MOU dynamic system.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    tau : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive values expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.

    Returns
    -------
    jac : ndarray of rank-2
        The Jacobian matrix of shape N x N for the MOU dynamical system.
    """
    # 0) SECURITY CHECKS
    # Check the input connectivity matrix
    con_shape = np.shape(con)
    if len(con_shape) != 2:
        raise ValueError( "'con' not a matrix." )
    if con_shape[0] != con_shape[1]:
        raise ValueError( "'con' not a square matrix." )
    # Make sure con is a ndarray of dtype = np.float64
    con = np.array(con, dtype=np.float)
    N = con_shape[0]

    # Check the tau constant, in case it is a 1-dimensional array-like.
    tau_shape = np.shape(tau)
    if tau_shape:
        if len(tau_shape) != 1:
            raise ValueError( "tau must be either a float or a 1D array." )
        if tau_shape[0] != N:
            raise ValueError( "'con' and tau not aligned." )
        # Make sure tau is a ndarray of dytpe = np.float64
        tau = np.array(tau, dtype=np.float)
    else:
        tau = tau * np.ones(N, dtype=np.float)

    # 1) CALCULATE THE JACOBIAN MATRIX
    jacdiag = -np.ones(N, dtype=np.float) / tau
    jac = np.diag(jacdiag) + con

    return jac


## GENERATION OF THE MAIN TENSORS #############################################
def CalcTensor(con, tau, sigma, tmax=20, timestep=0.1,
                                                normed=False, case='DynFlow'):
    """Generic function to create time evolution of the flows.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    tau : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma : ndarray of rank-2
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
        Temporal evolution of the network's dynamic flow. A tensor of shape
        (nt,N,N), where N is the number of nodes and nt = tmax * timestep is
        the number of time steps.
    """
    # 0) SECURITY CHECKS
    caselist = ['DynFlow', 'FullFlow', 'IntrinsicFlow']
    if case not in caselist:
        raise ValueError( "Please enter one of accepted cases: %s" %str(caselist) )

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep > tmax: raise ValueError("Incompatible values, timestep < tmax given")

    # 1) CALCULATE THE JACOBIAN MATRIX
    jac = JacobianMOU(con, tau)
    jacdiag = np.diagonal(jac)
    N = len(jac)

    # 2) CALCULATE THE DYNAMIC FLOW
    # 2.1) Calculate the extrinsic flow over integration time
    nt = int(tmax / timestep) + 1
    sigma_sqrt = scipy.linalg.sqrtm(sigma)

    flow_tensor = np.zeros((nt,N,N), dtype=np.float)

    if case == 'DynFlow':
        for i_t in range(nt):
            t = i_t * timestep
            # Calculate the term for jacdiag without using expm(), to speed up
            jacdiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the jaccobian at given time
            jac_t = scipy.linalg.expm(jac * t)
            # Calculate the dynamic communicability at time t
            flow_tensor[i_t] = np.dot( sigma_sqrt, jac_t - jacdiag_t )

    elif case == 'IntrinsicFlow':
        for i_t in range(nt):
            t = i_t * timestep
            # Calculate the term for jacdiag without using expm(), to speed up
            jacdiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the dynamic communicability at time t.
            flow_tensor[i_t] = np.dot( sigma_sqrt, jacdiag_t)

    elif case == 'FullFlow':
        for i_t in range(nt):
            t = i_t * timestep
            # Calculate the jaccobian at given time
            jac_t = scipy.linalg.expm(jac * t)
            # Calculate the non-normalised flow at time t.
            flow_tensor[i_t] = np.dot( sigma_sqrt, jac_t )

    # 2.2) Normalise by the scaling factor
    if normed:
        scaling_factor = np.abs(1./jacdiag).sum()
        flow_tensor /= scaling_factor

    return flow_tensor


## Wrappers using CalcTensor() ___________________________________________
def DynFlow(con, tau, sigma, tmax=20, timestep=0.1, normed=False):
    """Pair-wise conditional flows on a network over time for a given input.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    tau : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma : ndarray of rank-2
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
    dynflow_tensor : ndarray of rank-3
        Temporal evolution of the network's pair-wise conditional flows.
        A tensor of shape (nt,N,N), where N is the number of nodes and
        nt = tmax * timestep is the number of time steps.
    """
    dynflow_tensor = CalcTensor(con, tau, sigma, tmax=tmax,
                    timestep=timestep, normed=normed, case='DynFlow')

    return dynflow_tensor

def IntrinsicFlow(con, tau, sigma, tmax=20, timestep=0.1, normed=False):
    """Returns the flow dissipated through each node over time.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    tau : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma : ndarray of rank-2
        The matrix of Gaussian noise covariances.
    tmax : real valued number, positive (optional)
        Final time for integration.dynflow
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning: Not an integration step, just the desired sampling rate.
    normed : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    flow_tensor : ndarray of rank-3
        Temporal evolution of disspation through the nodes of the network.
        A tensor of shape (nt,N,N), where N is the number of nodes and
        nt = tmax * timestep is the number of time steps.
    """
    flow_tensor = CalcTensor(con, tau, sigma, tmax=tmax,
                    timestep=timestep, normed=normed, case='IntrinsicFlow')

    return flow_tensor

def FullFlow(con, tau, sigma, tmax=20, timestep=0.1, normed=False):
    """Returns the complete flow on a network over time for a given input.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    tau : real valued number, or ndarray of rank-1
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma : ndarray of rank-2
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
        Temporal evolution of the network's full-flow. A tensor of shape
        (nt,N,N), where N is the number of nodes and nt = tmax * timestep is
        the number of time steps.
    """
    flow_tensor = CalcTensor(con, tau, sigma, tmax=tmax,
                            timestep=timestep, normed=normed, case='FullFlow')

    return flow_tensor


##
