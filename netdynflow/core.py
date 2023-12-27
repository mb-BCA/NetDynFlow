# -*- coding: utf-8 -*-
# Copyright (c) 2024, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris
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
TransitionMatrix
    Returns the transition probability matrix for random walks.
JacobianMOU
    Calculates the Jacobian matrix for the MOU dynamic system.

Generation of main tensors
--------------------------
RespMatrices_DiscreteCascade
    Computes the pair-wise responses over time for the discrete cascade model.
RespMatrices_RandomWalk
    Computes the pair-wise responses over time for the simple random walk model.
RespMatrices_ContCascade
    Computes the pair-wise responses over time for the continuous cascade model.
RespMatrices_LeakyCascade
    Computes the pair-wise responses over time for the leaky-cascade model.

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

# __all__ = ['JacobianMOU', 'DynFlow', 'FullFlow', 'IntrinsicFlow']


# REORGANISE THIS MODULE TO GENERATE THE RESPONSE MATRICES FOR THE FIVE
# CANONICAL MODELS. FOR THE LEAKY-CASCADE AND THE CONTINUOUS DIFFUSION,
# PROVIDE THE OPTIONS TO COMPUTE THE GREEN'S FUNCTION, THE ONE REGRESSING THE
# DIAGONALS, AND THE RESPONSES ONLY DUE TO THE DIAGONALS. FIND PROPER NAMES
# FOR THOSE CASES.

## USEFUL FUNCTIONS ##########################################################
def TransitionMatrix(con, rwcase='simple'):
    """Returns the transition probability matrix for random walks.

    - If rwcase='simple'
    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the transition probability matrix for a simple
    random walk is computed as Tij = Aij / deg(i), where deg(i) is the output
    (weighted) degree of node i.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    rwcase : string (optional)
        Default 'simple' returns the transition probability matrix for the
        simple random walk.
        NOTE: For now only the simple random walk is supported. Optional
        parameter available to cover different calsses of random walks in future
        releases.

    Returns
    -------
    tp_matrix : ndarray of rank-2
        The transition probability matrix of shape N x N.
    """

    # 0) SECURITY CHECKS
    con_shape = np.shape(con)
    if len(con_shape) != 2:
        raise ValueError( "'con' not a matrix." )
    if con_shape[0] != con_shape[1]:
        raise ValueError( "'con' not a square matrix." )

    caselist = ['simple']
    if rwcase not in caselist:
        raise ValueError( "Please enter one of accepted cases: %s" %str(caselist) )

    # 1) COMPUTE THE TRANSITION PROBABILITY MATRIX
    N = con_shape[0]
    tp_matrix = con.copy().astype(float)

    if rwcase=='simple':
        indegree = con.sum(axis=0)
        for i in range(N):
            # Avoids NaN values in tp_matrix if node is disconnected
            if indegree[i]:
                tp_matrix[i] = con[i] / indegree[i]

    return tp_matrix

def JacobianMOU(con, tau):
    """Calculates the Jacobian matrix for the MOU dynamic system.

    TODO: RETHINK THE NAME OF THIS FUNCTION. MERGE DIFFERENT JACOBIAN GENERATOR
    FUNCTIONS INTO A SINGLE FUNCTION ?

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
    # Make sure con is a ndarray of dtype = float64
    con = np.array(con, dtype=float)
    N = con_shape[0]

    # Check the tau constant, in case it is a 1-dimensional array-like.
    tau_shape = np.shape(tau)
    if tau_shape:
        if len(tau_shape) != 1:
            raise ValueError( "tau must be either a float or a 1D array." )
        if tau_shape[0] != N:
            raise ValueError( "'con' and tau not aligned." )
        # Make sure tau is a ndarray of dytpe = float64
        tau = np.array(tau, dtype=float)
    else:
        tau = tau * np.ones(N, dtype=float)

    # 1) CALCULATE THE JACOBIAN MATRIX
    jacdiag = -np.ones(N, dtype=float) / tau
    jac = np.diag(jacdiag) + con

    return jac


## GENERATION OF THE MAIN TENSORS #############################################

# TODO: DECIDE BETTER NAMES FOR THESE FUNCTIONS. TRY GIVE THEM SHORTER NAMES.

def RespMatrices_DiscreteCascade(con, sigma=None, tmax=10):
    """Computes the pair-wise responses over time for the discrete cascade model.

    TODO: WHAT SHOULD WE DO WITH THE 'SIGMA' PARAMETER ? FOR THE MOU CASE
    I WANTED TO LEAVE IT THERE FOR GENERALITY. BUT, DOES IT MAKE SENSE TO
    CALL THIS EQUATION WITH A (CORRELATED) ADDITIVE GAUSSIAN NOISE ?

    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the response matrices Rij(t) encode the temporal
    response observed at node j due to a short stimulus applied on node i at
    time t=0.
    The discrete cascade is the simplest linear propagation model for
    DISCRETE VARIABLE and DISCRETE TIME in a network. It is represented by
    the following iterative equation:

            x(t+1) = A x(t) .

    This system is NON-CONSERVATIVE and leads to DIVERGENT dynamics. If all
    entries of A are positive, e.g, A is a binary graph, the both the solutions
    x_i(t) and the responses Rij(t) rapidly explode.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    sigma : None or ndarray of rank-1 (optional)
        The covariance matrix of the inputs.
        - The default value 'sigma=None' applies an input of amplitude 1.0
        to all nodes.
        - If a vector v of length N is entered, each node will receive an initial
        input of amplitude v_i.
    tmax : real valued number, positive (optional)
        Final time for integration.

    Returns
    -------
    resp_matrices : ndarray of rank-3
        Temporal evolution of the pair-wise responses. A tensor of shape
        (nt,N,N), where N is the number of nodes and nt = tmax*timestep + 1 is
        the number of time steps. The first time point contains the matrix of
        inputs (sigma).
    """
    # 0) SECURITY CHECKS
    N = len(con)
    if sigma is None: sigma_matrix = np.identity(N, dtype=float)
    elif len(np.shape(sigma)) == 1: sigma_matrix = sigma * np.identity(N, dtype=float)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")

    # 1) CALCULATE THE RESPONSE MATRICES
    # 1.1) Define some helper parameters
    nt = int(tmax) + 1
    # TODO: IN THIS CASE, DO WE NEED THIS NORMALIZATION ?
    sigma_sqrt = scipy.linalg.sqrtm(sigma_matrix)

    # Define the tensor where responses will be stored
    resp_matrices = np.zeros((nt,N,N), dtype=float )

    # 1.2) Compute the pair-wise response matrices over time
    resp_matrices[0] = sigma_matrix
    for i_t in range(1,nt):
        resp_matrices[i_t] = np.matmul(resp_matrices[i_t-1], con)

    return resp_matrices

def RespMatrices_RandomWalk(con, sigma=None, tmax=10):
    """Computes the pair-wise responses over time for the simple random walk model.

    TODO: WHAT SHOULD WE DO WITH THE 'SIGMA' PARAMETER ? FOR THE MOU CASE
    I WANTED TO LEAVE IT THERE FOR GENERALITY. BUT, DOES IT MAKE SENSE TO
    CALL THIS EQUATION WITH A (CORRELATED) ADDITIVE GAUSSIAN NOISE ?

    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the transition probability matrix is computed as
    Tij = Aij / deg(i), where deg(i) is the output (weighted) degree of node i.
    The response matrices Rij(t) encode the temporal response observed at
    node j due to a short stimulus applied on node i at time t=0.
    The random walk is the simplest linear propagation model for DISCRETE
    VARIABLE and DISCRETE TIME in a network. It is represented by the following
    iterative equation:

            x(t+1) = T x(t) .

    This system is CONSERVATIVE and leads to CONVERGENT dynamics. At any time
    t > 0 the number of walkers (or agents) found in the network is the same
    as the number of walkers initially seeded.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    sigma : None or ndarray of rank-1 (optional)
        The number of walkers per node initially seeded.
        - The default value 'sigma=None' begins with one walker at each node.
        - If a vector v of length N is entered, each node will be initialised
        with the number of walkers given in v_i.
    tmax : real valued number, positive (optional)
        Final time for integration.

    Returns
    -------
    resp_matrices : ndarray of rank-3
        Temporal evolution of the pair-wise responses. A tensor of shape
        (nt,N,N), where N is the number of nodes and nt = tmax*timestep + 1 is
        the number of time steps. The first time point contains the matrix of
        inputs (sigma).
    """
    # 0) SECURITY CHECKS
    N = len(con)
    if sigma is None: sigma_matrix = np.identity(N, dtype=float)
    elif len(np.shape(sigma)) == 1: sigma_matrix = sigma * np.identity(N, dtype=float)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")

    # 1) CALCULATE THE RESPONSE MATRICES
    # 1.1) Define some helper parameters
    nt = int(tmax) + 1

    # Define the transition probability matrix
    tpmatrix = TransitionMatrix(con, rwcase='simple')

    # Define the tensor where responses will be stored
    resp_matrices = np.zeros((nt,N,N), dtype=float)

    # 1.2) Compute the pair-wise response matrices over time
    resp_matrices[0] = sigma_matrix
    for i_t in range(1,nt):
        resp_matrices[i_t] = np.matmul(resp_matrices[i_t-1], tpmatrix)
        # resp_matrices[i_t] = np.matmul(tpmatrix, resp_matrices[i_t-1])

    return resp_matrices

def RespMatrices_ContCascade(con, sigma=None, tmax=10, timestep=0.1):
    """Computes the pair-wise responses over time for the continuous cascade model.

    TODO: WHAT SHOULD WE DO WITH THE 'SIGMA' PARAMETER ? FOR THE MOU CASE
    I WANTED TO LEAVE IT THERE FOR GENERALITY. BUT, DOES IT MAKE SENSE TO
    CALL THIS EQUATION WITH A (CORRELATED) ADDITIVE GAUSSIAN NOISE ?

    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the response matrices Rij(t) encode the temporal
    response observed at node j due to a short stimulus applied on node i at
    time t=0.
    The continuous-cascade is the simplest linear propagation model for
    CONTINUOUS VARIABLE and CONTINUOUS TIME in a network. It is represented by
    the following differential equation:

            xdot(t) = A x(t) .

    This system is NON-CONSERVATIVE and leads to DIVERGENT dynamics. If all
    entries of A are positive, e.g, A is a binary graph, the both the solutions
    x_i(t) and the responses Rij(t) rapidly explode.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    sigma : None or ndarray of rank-1 or ndarray of rank-2 (optional)
        The covariance matrix of the inputs.
        - The default value 'sigma=None' applies an input of amplitude 1.0
        to all nodes.
        - If a vector v of length N is entered, each node will receive an initial
        input of amplitude v_i.
        - If a matrix M of shape (N,N) is entered, diagonal entries M_ii will
        employed as the amplitudes of the inputs to node i. Extradiagonal
        value M_ij will be considered as correlated noise Gaussian noise. This
        case is left available for situations in which the system is interpreted
        as the multivariate Ornstein-Uhlenbeck process, which is the same
        equation but with additive Gaussian noise applied on the nodes.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning - Not an integration step, just the desired sampling rate.

    Returns
    -------
    resp_matrices : ndarray of rank-3
        Temporal evolution of the pair-wise responses. A tensor of shape
        (nt,N,N), where N is the number of nodes and nt = tmax*timestep + 1 is
        the number of time steps. The first time point contains the matrix of
        inputs (sigma).
    """
    # 0) SECURITY CHECKS
    N = len(con)
    if sigma is None: sigma_matrix = np.identity(N, dtype=float)
    elif len(np.shape(sigma)) == 1: sigma_matrix = sigma * np.identity(N, dtype=float)
    else: sigma_matrix = sigma

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep > tmax: raise ValueError("Incompatible values, timestep < tmax given")

    # 1) CALCULATE THE RESPONSE MATRICES
    # 1.1) Define some helper parameters
    nt = int(tmax / timestep) + 1
    # TODO: IN THIS CASE, DO WE NEED THIS NORMALIZATION ?
    sigma_sqrt = scipy.linalg.sqrtm(sigma_matrix)

    # Define the tensor where responses will be stored
    resp_matrices = np.zeros((nt,N,N), dtype=float )

    # 1.2) Compute the pair-wise response matrices over time
    if sigma is None:
        # Do the calculation a bit faster for the default unit inputs
        for i_t in range(nt):
            t = i_t * timestep
            # Calculate the Green's function at time t.
            resp_matrices[i_t] = scipy.linalg.expm(con * t)
    else:
        # Do the calculation if other inputs are entered besides the default
        for i_t in range(nt):
            t = i_t * timestep
            # Calculate the Green's function at time t.
            greens_t = scipy.linalg.expm(con * t)
            # Calculate the pair-wise responses at time t.
            resp_matrices[i_t] = np.dot(sigma_sqrt, greens_t)

    return resp_matrices

def RespMatrices_LeakyCascade(con, tau, sigma=None, tmax=10, timestep=0.1,
                                                case='regressed', normed=False):
    """Computes the pair-wise responses over time for the leaky-cascade model.

    NOTE: I WOULD RECOMMEND TO REMOVE THE 'normed' OPTIONAL PARAMETER.
    IT DOESN'T MAKE MUCH SENSE IN THE FULL OR THE INTRINSIC CASES. BUT I COULD
    LEAVE IT FOR LEGACY REASONS.

    TODO: DECIDE A BETTER NAME. WILL DEPEND ON HOW TO NAME THE FUNCTIONS FOR THE
    OTHER CANONICAL MODELS. TRY GIVE SHORTER NAMES.

    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the response matrices Rij(t) encode the temporal
    response observed at node j due to a short stimulus applied on node i at
    time t=0.
    The leaky-cascade is the time-continuous and variable-continuous linear
    propagation model represented by the following differential equation:

            xdot(t) = - x(t) / tau + A x(t).

    where tau is a leakage time-constant for a dissipation of the flows through
    the nodes. This model is reminiscent of the multivariate Ornstein-Uhlenbeck
    process, when additive Gaussian white noise is included.
    Given λmax is the largest eigenvalue of the (positive definite) matrix A, then
    - if tau < tau_max = 1 / λmax, then the leakage term dominates in the long
    time and the solutions for all nodes converge to zero.
    - If tau = tau_max, all nodes converge to x_i(t) = 1.
    - And, if tau < tau_max, then time-courses xdot(t) grow exponentially fast.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    tau : real valued number or ndarray of rank-1
        The decay time-constants of the nodes. A 1D array of length N.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigma : None or ndarray of rank-1 or ndarray of rank-2 (optional)
        The covariance matrix of the inputs.
        - The default value 'sigma=None' applies an input of amplitude 1.0
        to all nodes.
        - If a vector v of length N is entered, each node will receive an initial
        input of amplitude v_i.
        - If a matrix M of shape (N,N) is entered, diagonal entries M_ii will
        employed as the amplitudes of the inputs to node i. Extradiagonal
        value M_ij will be considered as correlated noise Gaussian noise. This
        case is left available for situations in which the system is interpreted
        as the multivariate Ornstein-Uhlenbeck process, which is the same
        equation but with additive Gaussian noise applied on the nodes.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning - Not an integration step, just the desired sampling rate.
    case : string (optional)
        TODO: WRITE ME HERE !!
    normed : boolean (optional)
        If True, normalises the tensor by a scaling factor, to make networks
        of different size comparable.

    Returns
    -------
    resp_matrices : ndarray of rank-3
        Temporal evolution of the pair-wise responses. A tensor of shape
        (nt,N,N), where N is the number of nodes and nt = tmax * timestep is
        the number of time steps.
    """
    # 0) SECURITY CHECKS
    N = len(con)

    if sigma is None: sigma = np.identity(N, dtype=float)
    elif len(np.shape(sigma)) == 1: sigma = sigma * np.identity(N, dtype=float)

    # TODO: Add validation of tau. Constant or vector

    caselist = ['regressed', 'full', 'intrinsic']
    if case not in caselist:
        raise ValueError( "Please enter one of accepted cases: %s" %str(caselist) )

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep > tmax: raise ValueError("Incompatible values, timestep < tmax given")

    # 1) CALCULATE THE JACOBIAN MATRIX
    jac = JacobianMOU(con, tau)
    jacdiag = np.diagonal(jac)

    # 1) CALCULATE THE RESPONSE MATRICES
    # 2.1) Calculate the extrinsic flow over integration time
    nt = int(tmax / timestep) + 1
    sigma_sqrt = scipy.linalg.sqrtm(sigma)

    resp_matrices = np.zeros((nt,N,N), dtype=float)

    if case == 'regressed':
        for i_t in range(nt):
            t = i_t * timestep
            # Calculate the term for jacdiag without using expm(), to speed up
            jacdiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the jaccobian at given time
            jac_t = scipy.linalg.expm(jac * t)
            # Calculate the pair-wise responses at time t
            resp_matrices[i_t] = np.dot( sigma_sqrt, jac_t - jacdiag_t )

    elif case == 'intrinsic':
        for i_t in range(nt):
            t = i_t * timestep
            # Calculate the term for jacdiag without using expm(), to speed up
            jacdiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the pair-wise responses at time t
            resp_matrices[i_t] = np.dot( sigma_sqrt, jacdiag_t)

    elif case == 'full':
        for i_t in range(nt):
            t = i_t * timestep
            # Calculate the jaccobian at given time
            jac_t = scipy.linalg.expm(jac * t)
            # Calculate the pair-wise responses at time t
            resp_matrices[i_t] = np.dot( sigma_sqrt, jac_t )

    # 2.2) Normalise by the scaling factor
    if normed:
        scaling_factor = np.abs(1./jacdiag).sum()
        resp_matrices /= scaling_factor

    return resp_matrices


def CalcTensor(con, tau, sigma, tmax=10, timestep=0.1,
                                                normed=False, case='DynFlow'):
    """Generic function to create time evolution of the flows.

    DEPRECATED FUNCTION: USE RespMatrices_LeakyCascade() INSTEAD.
    REMOVE BEFORE RELEASE OF v2.0.0

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
def DynFlow(con, tau, sigma, tmax=10, timestep=0.1, normed=False):
    """Pair-wise conditional flows on a network over time for a given input.

    DEPRECATED FUNCTION: USE RespMatrices_LeakyCascade() INSTEAD.
    REMOVE BEFORE RELEASE OF v2.0.0

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

def IntrinsicFlow(con, tau, sigma, tmax=10, timestep=0.1, normed=False):
    """Returns the flow dissipated through each node over time.

    DEPRECATED FUNCTION: USE RespMatrices_LeakyCascade() INSTEAD.
    REMOVE BEFORE RELEASE OF v2.0.0

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

def FullFlow(con, tau, sigma, tmax=10, timestep=0.1, normed=False):
    """Returns the complete flow on a network over time for a given input.

    DEPRECATED FUNCTION: USE RespMatrices_LeakyCascade() INSTEAD.
    REMOVE BEFORE RELEASE OF v2.0.0

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
