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


## THE MAIN TENSORS ##########################################################
def DynCom(conmatrix, tauconst, tmax=20, timestep=0.1, scalenorm=True,
                                                            eigvalnorm=False):
    """Returns the temporal evolution of a network's dynamic communicability.

    Parameters
    ----------
    conmatrix : ndarray of rank-2
        The adjacency matrix of the network.
    tauconst : real valued number, or ndarray
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning: Not an integration step, just the desired sampling rate.
    scalenorm : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.
    evnormalise : boolean (optional)
        'True' if adjacency matrix shall be normalised by the spectral diameter,
        'False' otherwise. This normalisation modifies the range of 'tauconst'
        for which the system converges.

    Returns
    -------
    dyncomtensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape (tmax*timestep) x N x N, where N is the number of nodes.
    """
    # 0) SECURITY CHECKS
    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep > tmax: raise ValueError("Incompatible values, timestep < tmax given")
    conmatrix = conmatrix.astype(float)

    # 1) NORMALIZE IF REQUESTED
    N = len(conmatrix)
    if eigvalnorm:
        # Find the spectral diameter
        eigenvalues = numpy.linalg.eigvals(conmatrix)
        evnorms = np.zeros(N, np.float)
        for i in range(N):
            evnorms[i] = numpy.linalg.norm(eigenvalues[i])
        evmax = evnorms.max()

        # Normalise the adjacency matrix
        conmatrix = 1./evmax * conmatrix

    # 2) CALCULATE THE DYNAMIC COMMUNICABILITY
    # 2.1) Define the Jacobian matrix
    if np.shape(tauconst):
        # In case tauconst was an array-like data
        assert len(tauconst) == N, "Data not aligned. 'conmatrix and tauconst not of same length"
        if type(tauconst) == numpy.ndarray:
            jac0diag = -1. / tauconst
        else:
            jac0diag = -1. / np.array(tauconst, dtype=float)
        scalingfactor = abs(tauconst).sum()
    else:
        # In case tauconst was just a number
        jac0diag = -1.0 * np.ones(N, dtype=float) / tauconst
        scalingfactor = abs(tauconst) * N

    diagidx = np.diag_indices(N)
    jacobian = conmatrix.copy()
    jacobian[diagidx] = jac0diag

    # 2.2) Dynamic communicability over time
    nsteps = int(tmax / timestep) + 1
    dyncomtensor = np.zeros((nsteps,N,N), np.float)
    for tidx in range(nsteps):
        t = tidx * timestep
        # Calculate the term for J0, without using expm(), which is very slow
        jac0diag_t = np.exp(jac0diag * t)
        jac0t = np.eye(N, dtype=float)
        jac0t[diagidx] = jac0diag_t
        # Calculate the dynamic communicability at time t.
        dyncomtensor[tidx] = (scipy.linalg.expm(jacobian*t) - jac0t)

    # 2.3) Normalise by the scaling factor
    if scalenorm:
        dyncomtensor /= scalingfactor

    return dyncomtensor

def DynFlow(conmatrix, tauconst, sigmamat, tmax=20, timestep=0.1, scalenorm=True, eigvalnorm=False):
    """Returns the extrinsinc flow on a network over time for a given input.

    Parameters
    ----------
    conmatrix : ndarray of rank-2
        The adjacency matrix of the network.
    tauconst : real valued number, or ndarray
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigmamat : ndarray of rank-2
        The matrix of Gaussian noise covariances.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning: Not an integration step, just the desired sampling rate.
    scalenorm : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.
    evnormalise : boolean (optional)
        'True' if adjacency matrix shall be normalised by the spectral diameter,
        'False' otherwise. This normalisation modifies the range of 'tauconst'
        for which the system converges.

    Returns
    -------
    dyncomtensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape (tmax*timestep) x N x N, where N is the number of nodes.
    """
    # 0) SECURITY CHECKS
    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep > tmax: raise ValueError("Incompatible values, timestep < tmax given")
    assert np.shape(conmatrix) == np.shape(sigmamat), "Connectivity and covariance matrices not aligned."
    conmatrix = conmatrix.astype(float)

    # 1) NORMALIZE IF REQUESTED
    N = len(conmatrix)
    if eigvalnorm:
        # Find the spectral diameter
        eigenvalues = numpy.linalg.eigvals(conmatrix)
        evnorms = np.zeros(N, np.float)
        for i in range(N):
            evnorms[i] = numpy.linalg.norm(eigenvalues[i])
        evmax = evnorms.max()

        # Normalise the adjacency matrix
        conmatrix = 1./evmax * conmatrix

    # 2) CALCULATE THE DYNAMIC COMMUNICABILITY
    # 2.1) Define the Jacobian matrix
    if np.shape(tauconst):
        # In case tauconst was an array-like data
        assert len(tauconst) == N, "Data not aligned. 'conmatrix and tauconst not of same length"
        if type(tauconst) == numpy.ndarray:
            jac0diag = -1. / tauconst
        else:
            jac0diag = -1. / np.array(tauconst, dtype=float)
        scalingfactor = abs(tauconst).sum()
    else:
        # In case tauconst was just a number
        jac0diag = -1.0 * np.ones(N, dtype=float) / tauconst
        scalingfactor = abs(tauconst) * N

    diagidx = np.diag_indices(N)
    jacobian = conmatrix.copy()
    jacobian[diagidx] = jac0diag

    # 2.2) Calculate the extrinsic flow over time
    nsteps = int(tmax / timestep) + 1
    sigmamat = np.sqrt(sigmamat)
    flowtensor = np.zeros((nsteps,N,N), np.float)
    for tidx in range(nsteps):
        t = tidx * timestep
        # Calculate the term for J0, without using expm(), which is very slow
        jac0diag_t = np.exp(jac0diag * t)
        jac0t = np.eye(N, dtype=float)
        jac0t[diagidx] = jac0diag_t
        # Calculate the dynamic communicability at time t.
        dynamiccomt = (scipy.linalg.expm(jacobian*t) - jac0t)
        flowtensor[tidx] = np.dot( sigmamat, dynamiccomt )

    # 2.3) Normalise by the scaling factor
    if scalenorm:
        flowtensor /= scalingfactor

    return flowtensor

def FullFlow(conmatrix, tauconst, sigmamat, tmax=20, timestep=0.1,
                                            scalenorm=True, eigvalnorm=False):
    """Returns the complete flow on a network over time for a given input.

    Parameters
    ----------
    conmatrix : ndarray of rank-2
        The adjacency matrix of the network.
    tauconst : real valued number, or ndarray
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigmamat : ndarray of rank-2
        The matrix of Gaussian noise covariances.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step. NOT an integration step, but the sampling step.
    scalenorm : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.
    evnormalise : boolean (optional)
        'True' if adjacency matrix shall be normalised by the spectral diameter,
        'False' otherwise. This normalisation modifies the range of 'tauconst'
        for which the system converges.

    Returns
    -------
    flowtensor : ndarray of rank-3
        Temporal evolution of the network's flow. A tensor of shape
        (tmax*timestep) x N x N, where N is the number of nodes.
    """
    # 0) SECURITY CHECKS
    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep > tmax: raise ValueError("Incompatible values, timestep < tmax given")
    assert np.shape(conmatrix) == np.shape(sigmamat), "Connectivity and covariance matrices not aligned."
    conmatrix = conmatrix.astype(float)

    # 1) NORMALIZE IF REQUESTED
    N = len(conmatrix)
    if eigvalnorm:
        # Find the spectral diameter
        eigenvalues = numpy.linalg.eigvals(conmatrix)
        evnorms = np.zeros(N, np.float)
        for i in range(N):
            evnorms[i] = numpy.linalg.norm(eigenvalues[i])
        evmax = evnorms.max()

        # Normalise the adjacency matrix
        conmatrix = 1./evmax * conmatrix

    # 2) CALCULATE THE TEMPORAL EVOLUTION OF THE FLOW
    # 2.1) Define the Jacobian matrix
    if np.shape(tauconst):
        # In case tauconst was an array-like data
        assert len(tauconst) == N, "Data not aligned. 'conmatrix and tauconst not of same length"
        if type(tauconst) == numpy.ndarray:
            jac0diag = -1. / tauconst
        else:
            jac0diag = -1. / np.array(tauconst, dtype=float)
        scalingfactor = abs(tauconst).sum()
    else:
        # In case tauconst was just a number
        jac0diag = -1.0 * np.ones(N, dtype=float) / tauconst
        scalingfactor = abs(tauconst) * N

    diagidx = np.diag_indices(N)
    jacobian = conmatrix.copy()
    jacobian[diagidx] = jac0diag

    # 2.2) Calculate the flow over time
    nsteps = int(tmax / timestep) + 1
    sigmamat = np.sqrt(sigmamat)
    flowtensor = np.zeros((nsteps,N,N), np.float)
    for tidx in range(nsteps):
        t = tidx * timestep
        # Calculate the non-normalised flow at time t.
        flowtensor[tidx] = np.dot( sigmamat, scipy.linalg.expm(jacobian*t) )

    # 2.3) Normalise by the scaling factor
    if scalenorm:
        flowtensor /= scalingfactor

    return flowtensor

def IntrinsicFlow(conmatrix, tauconst, sigmamat, tmax=20, timestep=0.1, scalenorm=True, eigvalnorm=False):
    """Returns the intrinsic flow on a network over time for a given input.

    Parameters
    ----------
    conmatrix : ndarray of rank-2
        The adjacency matrix of the network.
    tauconst : real valued number, or ndarray
        The decay rate at the nodes. Positive value expected.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    sigmamat : ndarray of rank-2
        The matrix of Gaussian noise covariances.
    tmax : real valued number, positive (optional)
        Final time for integration.
    timestep : real valued number, positive (optional)
        Sampling time-step.
        Warning: Not an integration step, just the desired sampling rate.
    scalenorm : boolean (optional)
        If True, normalises the tensor by the scaling factor, to make networks
        of different size comparable.
    evnormalise : boolean (optional)
        'True' if adjacency matrix shall be normalised by the spectral diameter,
        'False' otherwise. This normalisation modifies the range of 'tauconst'
        for which the system converges.

    Returns
    -------
    dyncomtensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape (tmax*timestep) x N x N, where N is the number of nodes.
    """
    # 0) SECURITY CHECKS
    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep > tmax: raise ValueError("Incompatible values, timestep < tmax given")
    assert np.shape(conmatrix) == np.shape(sigmamat), "Connectivity and covariance matrices not aligned."
    conmatrix = conmatrix.astype(float)

    # 1) NORMALIZE IF REQUESTED
    N = len(conmatrix)
    if eigvalnorm:
        # Find the spectral diameter
        eigenvalues = numpy.linalg.eigvals(conmatrix)
        evnorms = np.zeros(N, np.float)
        for i in range(N):
            evnorms[i] = numpy.linalg.norm(eigenvalues[i])
        evmax = evnorms.max()

        # Normalise the adjacency matrix
        conmatrix = 1./evmax * conmatrix

    # 2) CALCULATE THE DYNAMIC COMMUNICABILITY
    # 2.1) Define the Jacobian matrix
    if np.shape(tauconst):
        # In case tauconst was an array-like data
        assert len(tauconst) == N, "Data not aligned. 'conmatrix and tauconst not of same length"
        if type(tauconst) == numpy.ndarray:
            jac0diag = -1. / tauconst
        else:
            jac0diag = -1. / np.array(tauconst, dtype=float)
        scalingfactor = abs(tauconst).sum()
    else:
        # In case tauconst was just a number
        jac0diag = -1.0 * np.ones(N, dtype=float) / tauconst
        scalingfactor = abs(tauconst) * N

    diagidx = np.diag_indices(N)
    jacobian = conmatrix.copy()
    jacobian[diagidx] = jac0diag

    # 2.2) Calculate the extrinsic flow over time
    nsteps = int(tmax / timestep) + 1
    sigmamat = np.sqrt(sigmamat)
    flowtensor = np.zeros((nsteps,N,N), np.float)
    for tidx in range(nsteps):
        t = tidx * timestep
        # Calculate the term for J0, without using expm(), which is very slow
        jac0diag_t = np.exp(jac0diag * t)
        jac0t = np.eye(N, dtype=float)
        jac0t[diagidx] = jac0diag_t
        # Calculate the dynamic communicability at time t.
        flowtensor[tidx] = np.dot( sigmamat, jac0t )

    # 2.3) Normalise by the scaling factor
    if scalenorm:
        flowtensor /= scalingfactor

    return flowtensor



##
