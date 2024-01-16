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
Simulate the temporal solution x(t) of the nodes
=================================================

This module contains functions to run simulations of the different canonical
models.

Discrete-time models
---------------------
DiscreteCascade
    Simulates temporal evolution of the nodes for the discrete cascade.
RandomWalks
    Simulates temporal evolution of the nodes for the random walks.

Continuous-time models
-----------------------
ContinuousCascade
    Simulates temporal evolution of the nodes for the continuous cascade.
LeakyCascade
    Simulates temporal evolution of the nodes for the leaky-cascade model.
ContinuousDiffusion
    Simulates temporal evolution of the nodes for the simple diffustion model.

"""
# Standard libary imports
import numbers
# Third party packages
import numpy as np



## DISCRETE-TIME CANONICAL MODELS #############################################
def DiscreteCascade(con, X0, tmax=10):
    """Simulates temporal evolution of the nodes for the discrete cascade.

    It returns the time-series of the nodes for the discrete cascade model

            x(t+1) = A x(t),

    If A is a positive definite connectivity  matrix, then the solutions
    xt grow exponentially fast.

    Parameters
    ----------
    con : ndarray of rank-2, and shape (N,N).
        The adjacency matrix of the network.
    X0 : ndarray of rank-1, and length N.
        The initial conditions. Entries X0[i] are the inital values of the nodes.
        They can be either positive or negative, real or integer values.
    tmax : integer. (optional)
        The duration of the simulation in arbitrary time units.

    Returns
    -------
    Xt : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (tmax+1, N).
    """
    # 0) SECURITY CHECKS AND HANDLE THE INPUTS
    # Check the times
    if not isinstance(tmax, numbers.Number):
        raise ValueError( "'tmax' should be a number" )
    # Check the connectivity matrix
    N = len(con)
    # Ensure all arrays are of same dtype (np.float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    N = len(con)
    # Tanspose the connectivity matrix
    conT = np.copy(con.T, order='C')
    # Initialise the output array and enter the initial conditions
    Xt = np.zeros((tmax+1,N), np.float64)
    Xt[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,tmax+1):
        Xt[t] = np.dot(conT, Xt[t-1])

    return Xt

def RandomWalk(con, X0, tmax=10):
    """Simulates temporal evolution of the nodes for the random walks.

    It returns the time-series of the nodes for the discrete cascade model

            x(t+1) = T x(t),

    where T is the transition probability matrix. The solutions are computed
    recursively iterating the equation above. If A is a positive definite
    connectivity matrix the solutions converge to x_i(inf) values proportional
    to the input degree of the nodes.

    Parameters
    ----------
    con : ndarray of rank-2, and shape (N,N).
        The adjacency matrix of the network.
    X0 : ndarray of rank-1, and length N.
        The initial conditions. Entries X0[i] are the number of walkers
        starting at each node.
    tmax : integer. (optional)
        The duration of the simulation in arbitrary time units.

    Returns
    -------
    Xt : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (tmax+1, N).
        Xt[0] corresponds to the initial conditions.
    """
    # 0) SECURITY CHECKS AND HANDLE THE INPUTS
    # Check the times
    if not isinstance(tmax, numbers.Number):
        raise ValueError( "'tmax' should be a number" )
    # Check the connectivity matrix
    N = len(con)
    # Ensure all arrays are of same dtype (np.float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Compute the transition probability matrix
    N = len(con)
    Tmat = con / con.sum(axis=0)    # Assumes Aij = 1 if i -> j
    # Tmat = con / con.sum(axis=1)    # Assumes Aij = 1 if j -> i

    # Initialise the output array and enter the initial conditions
    Xt = np.zeros((tmax+1,N), np.float64)
    Xt[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,tmax+1):
        Xt[t] = np.dot(Tmat, Xt[t-1])

    return Xt


## CONTINUOUS-TIME CANONICAL MODELS ###########################################
def ContinuousCascade(con, X0, noise=None, tmax=10, timestep=0.01):
    """Simulates the temporal evolution of the nodes for the continuous cascade.

    It solves the differential equation for the simplest possible linear
    dynamical (propagation) model of nodes coupled via connectivity matrix A,
    with no local dynamics at the nodes.

            xdot = A x(t).

     If A is a positive definite connectivity  matrix, then the solutions
     xdot(t) grow exponentially fast.

    Parameters
    ----------
    con : ndarray of rank-2, and shape (N,N).
        The adjacency matrix of the network.
    X0 : ndarray of rank-1, and length N.
        The initial conditions. Entries X0[i] are the inital values of the nodes.
        They can be either positive or negative, real or integer values.
    noise : ndarray of rank-2 (optional)
        A precomputed noisy input to all nodes. Optional parameter. Not needed
        for the applications of the model-based network analysis. Left optional
        for general purposes. If given, 'noise' shall be a numpy array of shape
        (nsteps, N), where nsteps = int(tmax*timestep+1) and N is the number of nodes.
    tmax : scalar (optional)
        The duration of the simulation in arbitrary time units.
    timestep : scalar (optional)
        The time-step of the numerical integration.

    Returns
    -------
    Xdot : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (nsteps, N),
        where nsteps = int(tmax*timestep) + 1.
    """
    # 0) SECURITY CHECKS AND HANDLE THE INPUTS
    # Check the times
    if not isinstance(tmax, numbers.Number):
        raise ValueError( "'tmax' should be a number" )
    if not isinstance(timestep, numbers.Number):
        raise ValueError( "'timestep' should be a number" )
    # Calculate the simulation length
    nsteps = int(tmax / timestep) + 1

    # Check the connectivity matrix
    N = len(con)

    # Handle the noise input.
    if noise is None:
        # If nothing given by user, set noise to zeros.
        noise = np.zeros((nsteps,N), dtype=np.float64)
    elif isinstance(noise, np.ndarray): pass
    elif isinstance(noise, (list, tuple)):
        # If array-like, convert to ndarray
        noise = np.array(noise, np.float64)
    else:
        raise TypeError( "'noise' should be either None or a 2D array-like of shape (nsteps,N)." )

    # Ensure all arrays are of same dtype (np.float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)
    if noise.dtype != np.float64:  noise = noise.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Transpose the connectivity matrix
    conT = np.copy(con.T, order='C')

    # Initialise the output array
    Xdot = np.zeros((nsteps, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,nsteps):
        Xpre = Xdot[t-1]
        # Calculate the input to nodes due to couplings
        xcoup = np.dot(conT,Xpre)
        # Integration step
        Xdot[t] = Xpre + timestep * xcoup + noise[t]

    return Xdot

def LeakyCascade(con, X0, tau, noise=None, tmax=10, timestep=0.01):
    """Simulates temporal evolution of the nodes for the leaky-cascade model.

    It solves the differential equation for the linear propagation model of
    nodes coupled via connectivity matrix A, and a dissipative term leaking
    a fraction of the activity.

            xdot(t) = - x(t) / tau + A x(t).

     With λmax being the largest eigenvalue of the (positive definite) matrix A,
     - if tau < tau_max = 1 / λmax, then the leakage term dominates in the long
     time and the solutions for all nodes converge to zero.
     - If tau = tau_max, all nodes converge to x_i(t) = 1.
     - And, if tau < tau_max, then time-courses xdot(t) grow exponentially fast.

    Parameters
    ----------
    con : ndarray of rank-2, and shape (N,N).
        The adjacency matrix of the network.
    X0 : ndarray of rank-1, and length N.
        The initial conditions. Entries X0[i] are the inital values of the nodes.
        They can be either positive or negative, real or integer values.
    tau : real value or ndarray of rank-1 and length N.
        The decay time-constants of the nodes. If a single number is entered
        (e.g., tau = 2), then all nodes will be assigned the same value
        (e.g., tau[i] = 2). If an array is entered, then entries tau[i] will be
        considered as the decay time-constant for each node.
    noise : ndarray of rank-2 (optional)
        A precomputed noisy input to all nodes. Optional parameter. Not needed
        for the applications of the model-based network analysis. Left optional
        for general purposes. If given, 'noise' shall be a numpy array of shape
        (nsteps, N), where nsteps = int(tmax*timestep+1) and N is the number of nodes.
    tmax : scalar (optional)
        The duration of the simulation in arbitrary time units.
    timestep : scalar (optional)
        The time-step of the numerical integration.

    Returns
    -------
    Xdot : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (nsteps, N),
        where nsteps = int(tmax*timestep) + 1.
    """
    # 0) SECURITY CHECKS AND HANDLE THE INPUTS
    # Check the times
    if not isinstance(tmax, numbers.Number):
        raise ValueError( "'tmax' should be a number" )
    if not isinstance(timestep, numbers.Number):
        raise ValueError( "'timestep' should be a number" )
    # Calculate the simulation length
    nsteps = int(tmax / timestep) + 1

    # Check the connectivity matrix
    N = len(con)

    # Check whether tau is given as an scalar or an ndarray
    if isinstance(tau, np.ndarray): pass
    elif isinstance(tau, numbers.Number):
        # If scalar, give same value to all nodes
        tau = tau * np.ones(N, np.float64)
    elif isinstance(tau, (list,tuple)):
        # If array-like, convert to ndarray
        tau = np.array(tau, np.float64)
    else:
        raise TypeError( "'tau' should be either a scalar or 1D array-like of length N." )

    # Handle the noise input.
    if noise is None:
        # If nothing given by user, set noise to zeros.
        noise = np.zeros((nsteps,N), dtype=np.float64)
    elif isinstance(noise, np.ndarray): pass
    elif isinstance(noise, (list, tuple)):
        # If array-like, convert to ndarray
        noise = np.array(noise, np.float64)
    else:
        raise TypeError( "'noise' should be either None or a 2D array-like of shape (nsteps,N)." )

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)
    if tau.dtype != np.float64:     tau = tau.astype(np.float64)
    if noise.dtype != np.float64:  noise = noise.astype(np.float64)


    # 1) PREPARE FOR THE SIMULATION
    # Transpose the connectivity matrix
    conT = np.copy(con.T, order='C')
    # Conver the time-constants into decay rations
    alphas = 1./tau

    # Initialise the output array
    # nsteps = int(tmax / timestep) + 1
    Xdot = np.zeros((nsteps, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,nsteps):
        Xpre = Xdot[t-1]
        # Calculate the input to nodes due to couplings
        xcoup = np.dot(conT,Xpre)
        # Integration step
        Xdot[t] = Xpre + timestep * ( -1.0*alphas * Xpre + xcoup ) + noise[t]

    return Xdot

def ContinuousDiffusion(con, X0, noise=None, tmax=10, timestep=0.01):
    """Simulates the temporal evolution of the nodes for the continuous diffusion.

    It solves the differential equation for the simplest possible linear
    dynamical (propagation) linear model of nodes coupled via diffusive coupling,
    with no local dynamics at the nodes.

        xdot_i = A_ij * (x_i - x_j) .

    In matrix form, this equation is represented as:

        xdot(t) = -D x(t) + A x(t)  =  L x(t),

    where D is the diagonal matrix with the input degrees ink_i in the diagonal
    and L = -D + A is the graph Laplacian matrix.

    Parameters
    ----------
    con : ndarray of rank-2, and shape (N,N).
        The adjacency matrix of the network.
    X0 : ndarray of rank-1, and length N.
        The initial conditions. Entries X0[i] are the inital values of the nodes.
        They can be either positive or negative, real or integer values.
    noise : ndarray of rank-2 (optional)
        A precomputed noisy input to all nodes. Optional parameter. Not needed
        for the applications of the model-based network analysis. Left optional
        for general purposes. If given, 'noise' shall be a numpy array of shape
        (nsteps, N), where nsteps = int(tmax*timestep) + 1 and N is the number of nodes.
    tmax : scalar (optional)
        The duration of the simulation in arbitrary time units.
    timestep : scalar (optional)
        The time-step of the numerical integration.

    Returns
    -------
    Xdot : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (nsteps, N),
        where nsteps = int(tmax*timestep) + 1.
    """
    # 0) SECURITY CHECKS AND HANDLE THE INPUTS
    # Check the times
    if not isinstance(tmax, numbers.Number):
        raise ValueError( "'tmax' should be a number" )
    if not isinstance(timestep, numbers.Number):
        raise ValueError( "'timestep' should be a number" )
    # Calculate the simulation length
    nsteps = int(tmax / timestep) + 1

    # Check the connectivity matrix
    N = len(con)

    # Handle the noise input.
    if noise is None:
        # If nothing given by user, set noise to zeros.
        noise = np.zeros((nsteps,N), dtype=np.float64)
    elif isinstance(noise, np.ndarray): pass
    elif isinstance(noise, (list, tuple)):
        # If array-like, convert to ndarray
        noise = np.array(noise, np.float64)
    else:
        raise TypeError( "'noise' should be either None or a 2D array-like of shape (nsteps,N)." )

    # Ensure all arrays are of same dtype (np.float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)
    if noise.dtype != np.float64:  noise = noise.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Transpose the connectivity matrix
    conT = np.copy(con.T, order='C')
    ink = conT.sum(axis=1)
    # Lmat = - ink * np.identity(N, dtype=np.float64) + conT

    # Initialise the output array
    Xdot = np.zeros((nsteps, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,nsteps):
        Xpre = Xdot[t-1]
        # Calculate the input to nodes due to couplings
        xcoup = np.dot(conT,Xpre) - ink * Xpre
        # xcoup = np.dot(Lmat, Xpre)
        # Integration step
        Xdot[t] = Xpre + timestep * xcoup + noise[t]

    return Xdot





##
