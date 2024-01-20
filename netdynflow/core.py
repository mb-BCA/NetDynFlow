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
LaplacianMatrix
    Calculates the graph Laplacian.

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
RespMatrices_ContinuousDiffusion
    Computes the pair-wise responses over time for the linear diffusive model.


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
# Local imports from NetDynFlow
from . import io_helpers

# __all__ = ['JacobianMOU', 'DynFlow', 'FullFlow', 'IntrinsicFlow']

# FIND PROPER NAMES FOR THESE FUNCTIONS !!

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
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    rwcase : string, optional
        Default 'simple' returns the transition probability matrix for the
        simple random walk.
        NOTE: For now only the simple random walk is supported. Optional
        parameter available to cover different calsses of random walks in future
        releases.

    Returns
    -------
    tp_matrix : ndarray of rank-2
        The transition probability matrix of shape (N,N).
    """

    # 0) HANDLE AND CHECK THE INPUTS. Ensure all arrays are of same dtype
    io_helpers.validate_con(con)
    if con.dtype != np.float64:
        con = con.astype(np.float64)
    N = len(con)

    caselist = ['simple']
    if rwcase not in caselist:
        raise ValueError( "Please enter one of accepted cases: %s" %str(caselist) )

    # 1) COMPUTE THE TRANSITION PROBABILITY MATRIX
    N = con_shape[0]
    tp_matrix = con.copy().astype(float)
    if rwcase=='simple':
        outdegree = con.sum(axis=1)
        for i in range(N):
            # Avoids NaN values in tp_matrix if node is disconnected
            if outdegree[i]:
                tp_matrix[i] = con[i] / outdegree[i]

    return tp_matrix

def JacobianMOU(con, tau):
    """Calculates the Jacobian matrix for the MOU dynamic system.

    TODO: RETHINK THE NAME OF THIS FUNCTION. MERGE DIFFERENT JACOBIAN GENERATOR
    FUNCTIONS INTO A SINGLE FUNCTION ?

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    tau : real value or ndarray (1d) of length N.
        The decay time-constants of the nodes. If a scalar value is entered,
        `tau = c`, then all nodes will be assigned the same value `tau[i] = 2`
        (identical nodes). If an 1d-array is entered, each node i is assigned
        decay time-constant `tau[i]`.

    Returns
    -------
    jac : ndarray (2d) of shape (N,N)
        The Jacobian matrix for the MOU dynamical system.
    """
    # 0) HANDLE AND CHECK THE INPUTS. Ensure all arrays are of same dtype
    io_helpers.validate_con(con)
    N = len(con)
    tau = io_helpers.validate_tau(tau, N)

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if tau.dtype != np.float64:    tau = tau.astype(np.float64)

    # 1) CALCULATE THE JACOBIAN MATRIX
    jac = -1.0/tau * np.identity(N, dtype=float) + con

    return jac

def LaplacianMatrix(con, normed=False):
    """Calculates the graph Laplacian.

    TODO: WRITE THE DESCRIPTION HERE

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    normed : boolean, optional
        If True, it returns the normalised graph Laplacian.

    Returns
    -------
    lap : ndarray (2d) of shape(N,N)
        The graph Laplacian matrix.
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    if con.dtype != np.float64:
        con = con.astype(np.float64)
    N = con_shape[0]

    # 1) CALCULATE THE GRAPH LAPLACIAN MATRIX
    outdegree = con.sum(axis=1)
    lap = - outdegree * np.identity(N, dtype=float) + con

    if normed:
        for i in range(N):
            # Avoids NaN values in tp_matrix if node is disconnected
            if outdegree[i]:
                lap[i] /= outdegree[i]

    return lap


## GENERATION OF THE MAIN TENSORS #############################################
## DISCRETE-TIME CANONICAL MODELS #############################################

# TODO: DECIDE BETTER NAMES FOR THESE FUNCTIONS. TRY GIVE THEM SHORTER NAMES.

def RespMatrices_DiscreteCascade(con, sigma=1.0, tmax=10):
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
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    sigma : scalar, ndarray (1d) of length N, or ndarray (2d) of shape (N,N), optional
        TODO: RE-WRITE AFTER DECIDING WHAT TO DO WITH 'SIGMA'
        The covariance matrix of the inputs.
        - The default value 'sigma=1.0' applies an input of amplitude 1.0
        to all nodes.
        - If a vector v of length N is entered, each node will receive an initial
        input of amplitude v_i.
    tmax : integer, optional
        The duration of the simulation, number of discrete time steps.

    Returns
    -------
    resp_matrices : ndarray (3d) of shape (nt,N,N)
        Temporal evolution of the pair-wise responses. A tensor of shape
        (nt,N,N), where N is the number of nodes and nt = tmax*timestep + 1 is
        the number of time steps. The first time point contains the matrix of
        inputs (sigma).
    """
    # 0) HANDLE AND CHECK THE INPUTS. Ensure all arrays are of same dtype
    io_helpers.validate_con(con)
    N = len(con)
    # sigma = io_helpers.validate_sigma()

    # if sigma is None: sigma_matrix = np.identity(N, dtype=float)
    # elif len(np.shape(sigma)) == 1: sigma_matrix = sigma * np.identity(N, dtype=float)
    #
    # if tmax <= 0.0: raise ValueError("'tmax' must be positive")

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if sigma.dtype != np.float64:   sigma = sigma.astype(np.float64)

    # 1) CALCULATE THE RESPONSE MATRICES
    nt = int(tmax) + 1
    # TODO: IN THIS CASE, DO WE NEED THIS NORMALIZATION ?
    sigma_sqrt = scipy.linalg.sqrtm(sigma_matrix)
    # Define the tensor where responses will be stored
    resp_matrices = np.zeros((nt,N,N), dtype=float )

    # 2) COMPUTE THE PAIR-WISE RESPONSE MATRICES OVER TIME
    resp_matrices[0] = sigma_matrix
    for it in range(1,nt):
        resp_matrices[it] = np.matmul(resp_matrices[it-1], con)

    return resp_matrices

def RespMatrices_RandomWalk(con, sigma=1.0, tmax=10):
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
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    sigma : scalar, ndarray (1d) of length N, or ndarray (2d) of shape (N,N), optional
        TODO: RE-WRITE AFTER DECIDING WHAT TO DO WITH 'SIGMA'
        The number of walkers per node initially seeded.
        - The default value 'sigma=1.0' begins with one walker at each node.
        - If a vector v of length N is entered, each node will be initialised
        with the number of walkers given in v_i.
    tmax : integer, optional
        The duration of the simulation, discrete time steps.

    Returns
    -------
    resp_matrices : ndarray of rank-3
        Temporal evolution of the pair-wise responses. A tensor of shape
        (nt,N,N), where N is the number of nodes and nt = tmax*timestep + 1 is
        the number of time steps. The first time point contains the matrix of
        inputs (sigma).
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    # sigma = io_helpers.validate_sigma()

    # if sigma is None: sigma_matrix = np.identity(N, dtype=float)
    # elif len(np.shape(sigma)) == 1: sigma_matrix = sigma * np.identity(N, dtype=float)
    #
    # if tmax <= 0.0: raise ValueError("'tmax' must be positive")

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if sigma.dtype != np.float64:   sigma = sigma.astype(np.float64)

    # 1) CALCULATE THE RESPONSE MATRICES
    nt = int(tmax) + 1
    # Define the transition probability matrix
    tpmatrix = TransitionMatrix(con, rwcase='simple')
    # Define the tensor where responses will be stored
    resp_matrices = np.zeros((nt,N,N), dtype=float)

    # 2) COMPUTE THE PAIR-WISE RESPONSE MATRICES OVER TIME
    resp_matrices[0] = sigma_matrix
    for it in range(1,nt):
        resp_matrices[it] = np.matmul(resp_matrices[it-1], tpmatrix)
        # resp_matrices[it] = np.matmul(tpmatrix, resp_matrices[it-1])

    return resp_matrices


## CONTINUOUS-TIME CANONICAL MODELS ###########################################
def RespMatrices_ContCascade(con, sigma=1.0, tmax=10, timestep=0.1):
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
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    sigma : scalar, ndarray (1d) of length N, or ndarray (2d) of shape (N,N), optional
        TODO: RE-WRITE AFTER DECIDING WHAT TO DO WITH 'SIGMA'
        The covariance matrix of the inputs.
        - The default value 'sigma=1.0' applies an input of amplitude 1.0
        to all nodes.
        - If a vector v of length N is entered, each node will receive an initial
        input of amplitude v_i.
        - If a matrix M of shape (N,N) is entered, diagonal entries M_ii will
        employed as the amplitudes of the inputs to node i. Extradiagonal
        value M_ij will be considered as correlated noise Gaussian noise. This
        case is left available for situations in which the system is interpreted
        as the multivariate Ornstein-Uhlenbeck process, which is the same
        equation but with additive Gaussian noise applied on the nodes.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Temporal step (resolution) between consecutive calculations of responses.

    Returns
    -------
    resp_matrices : ndarray of rank-3
        Temporal evolution of the pair-wise responses. A tensor of shape
        (nt,N,N), where N is the number of nodes and nt = tmax*timestep + 1 is
        the number of time steps. The first time point contains the matrix of
        inputs (sigma).

    NOTE
    ----
    TODO: WRITE ME HERE, EXPLANATION ABOUT DURATION AND TIME-STEPS ...
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    # sigma = io_helpers.validate_sigma()

    # if sigma is None:
    #     sigma_matrix = np.identity(N, dtype=float)
    # elif len(np.shape(sigma)) == 1:
    #     sigma_matrix = sigma * np.identity(N, dtype=float)
    # else:
    #     sigma_matrix = sigma

    # if tmax <= 0.0: raise ValueError("'tmax' must be positive")

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if sigma.dtype != np.float64:   sigma = sigma.astype(np.float64)

    # 1) CALCULATE THE RESPONSE MATRICES
    nt = int(tmax / timestep) + 1
    # TODO: IN THIS CASE, DO WE NEED THIS NORMALIZATION ?
    sigma_sqrt = scipy.linalg.sqrtm(sigma_matrix)
    # Define the tensor where responses will be stored
    resp_matrices = np.zeros((nt,N,N), dtype=float )

    # 2) COMPUTE THE PAIR-WISE RESPONSE MATRICES OVER TIME
    if sigma is None:
        # Do the calculation a bit faster for the default unit inputs
        for it in range(nt):
            t = it * timestep
            # Calculate the Green's function at time t.
            resp_matrices[it] = scipy.linalg.expm(con * t)
    else:
        # Do the calculation if other inputs are entered besides the default
        for it in range(nt):
            t = it * timestep
            # Calculate the Green's function at time t.
            greens_t = scipy.linalg.expm(con * t)
            # Calculate the pair-wise responses at time t.
            resp_matrices[it] = np.dot(sigma_sqrt, greens_t)

    return resp_matrices

def RespMatrices_LeakyCascade(con, tau, sigma=1.0, tmax=10, timestep=0.1,
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
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    tau : real value or ndarray (1d) of length N.
        The decay time-constants of the nodes. If a scalar value is entered,
        `tau = c`, then all nodes will be assigned the same value `tau[i] = 2`
        (identical nodes). If an 1d-array is entered, each node i is assigned
        decay time-constant `tau[i]`.
    sigma : scalar, ndarray (1d) of length N, or ndarray (2d) of shape (N,N), optional
        TODO: RE-WRITE AFTER DECIDING WHAT TO DO WITH 'SIGMA'
        The covariance matrix of the inputs.
        - The default value 'sigma=1.0' applies an input of amplitude 1.0
        to all nodes.
        - If a vector v of length N is entered, each node will receive an initial
        input of amplitude v_i.
        - If a matrix M of shape (N,N) is entered, diagonal entries M_ii will
        employed as the amplitudes of the inputs to node i. Extradiagonal
        value M_ij will be considered as correlated noise Gaussian noise. This
        case is left available for situations in which the system is interpreted
        as the multivariate Ornstein-Uhlenbeck process, which is the same
        equation but with additive Gaussian noise applied on the nodes.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Temporal step (resolution) between consecutive calculations of responses.
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

    NOTE
    ----
    TODO: WRITE ME HERE, EXPLANATION ABOUT DURATION AND TIME-STEPS ...
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    tau = io_helpers.validate_tau(tau, N)
    # sigma = io_helpers.validate_sigma()

    # if sigma is None:
    #     sigma_matrix = np.identity(N, dtype=float)
    # elif len(np.shape(sigma)) == 1:
    #     sigma_matrix = sigma * np.identity(N, dtype=float)
    # else:
    #     sigma_matrix = sigma

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if tau.dtype != np.float64:     tau = tau.astype(np.float64)
    if sigma.dtype != np.float64:   sigma = sigma.astype(np.float64)

    caselist = ['regressed', 'full', 'intrinsic']
    if case not in caselist:
        raise ValueError( "Please enter one of accepted cases: %s" %str(caselist) )


    # 1) CALCULATE THE JACOBIAN MATRIX
    jac = JacobianMOU(con, tau)
    jacdiag = np.diagonal(jac)

    # 2) CALCULATE THE RESPONSE MATRICES
    # 2.1) Calculate the extrinsic flow over integration time
    nt = int(tmax / timestep) + 1
    sigma_sqrt = scipy.linalg.sqrtm(sigma)
    resp_matrices = np.zeros((nt,N,N), dtype=float)

    if case == 'regressed':
        for it in range(nt):
            t = it * timestep
            # Calculate the term for jacdiag without using expm(), to speed up
            jacdiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the jaccobian at given time
            jac_t = scipy.linalg.expm(jac * t)
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.dot( sigma_sqrt, jac_t - jacdiag_t )

    elif case == 'intrinsic':
        for it in range(nt):
            t = it * timestep
            # Calculate the term for jacdiag without using expm(), to speed up
            jacdiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.dot( sigma_sqrt, jacdiag_t)

    elif case == 'full':
        for it in range(nt):
            t = it * timestep
            # Calculate the jaccobian at given time
            jac_t = scipy.linalg.expm(jac * t)
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.dot( sigma_sqrt, jac_t )

    # 2.2) Normalise by the scaling factor
    if normed:
        scaling_factor = np.abs(1./jacdiag).sum()
        resp_matrices /= scaling_factor

    return resp_matrices

def RespMatrices_ContinuousDiffusion(con, sigma=1.0, tmax=10, timestep=0.1,
                                                case='regressed', normed=False):
    """Computes the pair-wise responses over time for the linear diffusive model.

    TODO: DECIDE A BETTER NAME. WILL DEPEND ON HOW TO NAME THE FUNCTIONS FOR THE
    OTHER CANONICAL MODELS. TRY GIVE SHORTER NAMES.

    Given a connectivity matrix A, where Aij represents the (weighted)
    connection from i to j, the response matrices Rij(t) encode the temporal
    response observed at node j due to a short stimulus applied on node i at
    time t=0.
    The continuous diffusion is the simplest time-continuous and variable-
    continuous linear propagation model with diffusive coupling. It is
    represented by the following differential equation:

            xdot(t) = -D x(t) + A x(t) = L x(t).

    where D is a diagonal matrix containing the (output) degrees of the nodes
    in the diagonal, and L = -D + A is the graph Laplacian matrix. This model
    is reminiscent of the continuous leaky cascade but considering that
    tau(i) = 1./deg(i). As such, the input and the leaked flows are balanced
    at each node.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    sigma : scalar, ndarray (1d) of length N, or ndarray (2d) of shape (N,N), optional
        TODO: RE-WRITE AFTER DECIDING WHAT TO DO WITH 'SIGMA'
        The covariance matrix of the inputs.
        - The default value 'sigma=1.0' applies an input of amplitude 1.0
        to all nodes.
        - If a vector v of length N is entered, each node will receive an initial
        input of amplitude v_i.
        - If a matrix M of shape (N,N) is entered, diagonal entries M_ii will
        employed as the amplitudes of the inputs to node i. Extradiagonal
        value M_ij will be considered as correlated noise Gaussian noise. This
        case is left available for situations in which the system is interpreted
        as the multivariate Ornstein-Uhlenbeck process, which is the same
        equation but with additive Gaussian noise applied on the nodes.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Temporal step (resolution) between consecutive calculations of responses.
    case : string (optional)
        TODO: WRITE ME HERE !!
    normed : boolean (optional)
        If True, it employs the normalised graph Laplacian L' = D^-1 L.

    Returns
    -------
    resp_matrices : ndarray of rank-3
        Temporal evolution of the pair-wise responses. A tensor of shape
        (nt,N,N), where N is the number of nodes and nt = tmax * timestep is
        the number of time steps.

    NOTE
    ----
    TODO: WRITE ME HERE, EXPLANATION ABOUT DURATION AND TIME-STEPS ...
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    # sigma = io_helpers.validate_sigma()

    # if sigma is None:
    #     sigma_matrix = np.identity(N, dtype=float)
    # elif len(np.shape(sigma)) == 1:
    #     sigma_matrix = sigma * np.identity(N, dtype=float)
    # else:
    #     sigma_matrix = sigma

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if sigma.dtype != np.float64:   sigma = sigma.astype(np.float64)


    caselist = ['regressed', 'full', 'intrinsic']
    if case not in caselist:
        raise ValueError( "Please enter one of accepted cases: %s" %str(caselist) )


    # 1) CALCULATE THE JACOBIAN MATRIX
    # NOTE: The graph Laplacian is the Jacobian matrix of the linear propagation
    # model based with diffusive coupling. Hence, after calling the Laplacian in
    # the next line, the code is the same as for the Leaky Cascade.
    jac = LaplacianMatrix(con, normed=normed)
    jacdiag = np.diagonal(jac)

    # 1) CALCULATE THE RESPONSE MATRICES
    # 2.1) Calculate the extrinsic flow over integration time
    nt = int(tmax / timestep) + 1
    sigma_sqrt = scipy.linalg.sqrtm(sigma)
    resp_matrices = np.zeros((nt,N,N), dtype=float)

    if case == 'regressed':
        for it in range(nt):
            t = it * timestep
            # Calculate the term for jacdiag without using expm(), to speed up
            jacdiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the jaccobian at given time
            jac_t = scipy.linalg.expm(jac * t)
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.dot( sigma_sqrt, jac_t - jacdiag_t )

    elif case == 'intrinsic':
        for it in range(nt):
            t = it * timestep
            # Calculate the term for jacdiag without using expm(), to speed up
            jacdiag_t = np.diag( np.exp(jacdiag * t) )
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.dot( sigma_sqrt, jacdiag_t)

    elif case == 'full':
        for it in range(nt):
            t = it * timestep
            # Calculate the jaccobian at given time
            jac_t = scipy.linalg.expm(jac * t)
            # Calculate the pair-wise responses at time t
            resp_matrices[it] = np.dot( sigma_sqrt, jac_t )

    return resp_matrices



##
