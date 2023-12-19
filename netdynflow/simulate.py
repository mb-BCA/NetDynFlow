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

# Third party packages
import numpy as np



## DISCRETE-TIME CANONICAL MODELS #############################################

def DiscreteCascade(con, X0=None, tmax=10):
    """Simulates temporal evolution of the nodes for the discrete cascade.

    It returns the time-series of the nodes for the discrete cascade model

            x(t+1) = A x(t),

    If A is a positive definite connectivity  matrix, then the solutions
    xt grow exponentially fast.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    X0 : ndarray of rank-1. (optional)
        The initial conditions for the simulation. A vector of length N nodes.
        If none given, simulation will start with unit input to all nodes, X0 = 1.
    tmax : integer. (optional)
        The duration of the simulation in arbitrary time units.

    Returns
    -------
    Xt : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (tmax+1, N).
        Xt[0] corresponds to the initial conditions.
    """

    # 0) SECURITY CHECKS
    # To be done ...
    # Make sure tmax is an integer, or convert to it.
    # Make sure X0 is 1D, if given by the user.

    # 1) PREPARE FOR THE SIMULATION
    # Infos about the network
    N = len(con)
    conT = np.copy(con.T, order='C')

    # Set the initial conditions to default, if not given by the user
    if X0 is None: X0 = np.ones(N, dtype=np.float64)

    # Initialise the output array and enter the initial conditions
    Xt = np.zeros((tmax+1,N), np.float64)
    Xt[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,tmax+1):
        Xt[t] = np.dot(conT, Xt[t-1])

    return Xt

def RandomWalk(con, X0=None, tmax=10):
    """Simulates temporal evolution of the nodes for the random walks.

    It returns the time-series of the nodes for the discrete cascade model

            x(t+1) = T x(t),

    where T is the transition probability matrix. The solutions are computed
    recursively iterating the equation above. If A is a positive definite
    connectivity matrix the solutions converge to x_i(inf) values proportional
    to the input degree of the nodes.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    X0 : ndarray of rank-1. (optional)
        The initial conditions for the simulation. A vector of length N nodes.
        If none given, simulation will start with unit input to all nodes, X0 = 1.
    tmax : integer. (optional)
        The duration of the simulation in arbitrary time units.

    Returns
    -------
    Xt : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (tmax+1, N).
        Xt[0] corresponds to the initial conditions.

    """

    # 0) SECURITY CHECKS
    # To be done ...
    # Make sure tmax is an integer, or convert to it.
    # Make sure X0 is 1D, if given by the user.

    # 1) PREPARE FOR THE SIMULATION
    # Compute the transition probability matrix
    N = len(con)
    Tmat = con / con.sum(axis=0)    # Assumes Aij = 1 if i -> j
    # Tmat = con / con.sum(axis=1)    # Assumes Aij = 1 if j -> i

    # Set the initial conditions to default, if not given by the user
    if X0 is None: X0 = np.ones(N, dtype=np.float64)

    # Initialise the output array and enter the initial conditions
    Xt = np.zeros((tmax+1,N), np.float64)
    Xt[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,tmax+1):
        Xt[t] = np.dot(Tmat, Xt[t-1])

    return Xt


## CONTINUOUS-TIME CANONICAL MODELS ###########################################

def ContinuousCascade(con, X0=None, tmax=10, timestep=0.01, noise=None):
    """Simulates the temporal evolution of the nodes for the continuous cascade.

    It solves the differential equation for the simplest possible linear
    dynamical (propagation) model of nodes coupled via connectivity matrix A,
    with no local dynamics at the nodes.

            xdot = A x(t).

     If A is a positive definite connectivity  matrix, then the solutions
     xdot(t) grow exponentially fast.

    Parameters
    ----------
    con : ndarray of rank-2
        The adjacency matrix of the network.
    X0 : ndarray of rank-1. (optional)
        The initial conditions for the simulation. A vector of length N nodes.
        If none given, simulation will start with unit input to all nodes, X0 = 1.
    tmax : scalar (optional)
        The duration of the simulation in arbitrary time units.
    timestep : scalar (optional)
        The time-step of the numerical integration.
    noise : ndarray of rank-2 (optional)
        A precomputed noisy input to all nodes. Optional parameter. Not needed
        for the applications of the model-based network analysis. Left optional
        for general purposes. If given, 'noise' shall be a numpy array of shape
        (nsteps, N), where nsteps = int(tmax*timestep+1) and N is the number of nodes.

    Returns
    -------
    Xdot : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (nsteps, N),
        where nsteps = int(tmax*timestep) + 1.
    """
    # 0) SECURITY CHECKS
    # To be done ...

    # 1) PREPARE FOR THE SIMULATION
    # Infos about the network
    N = len(con)
    conT = np.copy(con.T, order='C')

    # Set the initial conditions to default, if not given by the user
    if X0 is None: X0 = np.ones(N, dtype=np.float64)

    # Initialise the output array
    nsteps = int(tmax / timestep) + 1
    Xdot = np.zeros((nsteps, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # Set the noise array to default, if not given by the user
    if noise is None: noise = np.zeros((nsteps,N), dtype=np.float64)

    # 2) RUN THE SIMULATION
    for t in range(1,nsteps):
        Xpre = Xdot[t-1]

        # Calculate the input to nodes due to couplings
        xcoup = np.dot(conT,Xpre)

        # Integration step
        # Xdot[t] = Xpre + timestep * ( gcoupling*xcoup ) + noise[t]
        Xdot[t] = Xpre + timestep * xcoup + noise[t]

    return Xdot

def LeakyCascade(con, tau, X0=None, tmax=10, timestep=0.01, noise=None):
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
    con : ndarray of rank-2
        The adjacency matrix of the network.
    tau : ndarray of rank-1
        The decay time-constants of the nodes. A 1D array of length N.
        If a number is given, then the function considers all nodes have same
        decay rate. Alternatively, an array can be inputed with the decay rate
        of each node.
    X0 : ndarray of rank-1. (optional)
        The initial conditions for the simulation. A vector of length N nodes.
        If none given, simulation will start with unit input to all nodes, X0 = 1.
    tmax : scalar (optional)
        The duration of the simulation in arbitrary time units.
    timestep : scalar (optional)
        The time-step of the numerical integration.
    noise : ndarray of rank-2 (optional)
        A precomputed noisy input to all nodes. Optional parameter. Not needed
        for the applications of the model-based network analysis. Left optional
        for general purposes. If given, 'noise' shall be a numpy array of shape
        (nsteps, N), where nsteps = int(tmax*timestep+1) and N is the number of nodes.

    Returns
    -------
    Xdot : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (nsteps, N),
        where nsteps = int(tmax*timestep) + 1.
    """

    # 0) SECURITY CHECKS
    N = len(con)
    # To be done ...

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


    # 1) PREPARE FOR THE SIMULATION
    # Infos about the network
    conT = np.copy(con.T, order='C')

    # Set the leakage time-constants to default, if not given by the user
    if tau is None: tau = np.ones(N, dtype=np.float64)
    alphas = 1./tau

    # Set the initial conditions to default, if not given by the user
    if X0 is None: X0 = np.ones(N, dtype=np.float64)

    # Initialise the output array
    nsteps = int(tmax / timestep) + 1
    Xdot = np.zeros((nsteps, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # Set the noise array to default, if not given by the user
    if noise is None: noise = np.zeros((nsteps,N), dtype=np.float64)

    # 2) RUN THE SIMULATION
    for t in range(1,nsteps):
        Xpre = Xdot[t-1]

        # Calculate the input to nodes due to couplings
        xcoup = np.dot(conT,Xpre)

        # Integration step
        Xdot[t] = Xpre + timestep * ( -1.0*alphas * Xpre + xcoup ) + noise[t]

    return Xdot

def ContinuousDiffusion(con, X0=None, tmax=10, timestep=0.01, noise=None):
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
    con : ndarray of rank-2
        The adjacency matrix of the network.
    X0 : ndarray of rank-1. (optional)
        The initial conditions for the simulation. A vector of length N nodes.
        If none given, simulation will start with unit input to all nodes, X0 = 1.
    tmax : scalar (optional)
        The duration of the simulation in arbitrary time units.
    timestep : scalar (optional)
        The time-step of the numerical integration.
    noise : ndarray of rank-2 (optional)
        A precomputed noisy input to all nodes. Optional parameter. Not needed
        for the applications of the model-based network analysis. Left optional
        for general purposes. If given, 'noise' shall be a numpy array of shape
        (nsteps, N), where nsteps = int(tmax*timestep) + 1 and N is the number of nodes.

    Returns
    -------
    Xdot : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (nsteps, N),
        where nsteps = int(tmax*timestep) + 1.
    """
    # 0) SECURITY CHECKS
    # To be done ...

    # 1) PREPARE FOR THE SIMULATION
    # Infos about the network
    N = len(con)
    conT = np.copy(con.T, order='C')
    ink = conT.sum(axis=1)
    # Lmat = - ink * np.identity(N, dtype=np.float64) + conT

    # Set the initial conditions to default, if not given by the user
    if X0 is None: X0 = np.ones(N, dtype=np.float64)

    # Initialise the output array
    nsteps = int(tmax / timestep) + 1
    Xdot = np.zeros((nsteps, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # Set the noise array to default, if not given by the user
    if noise is None: noise = np.zeros((nsteps,N), dtype=np.float64)

    # 2) RUN THE SIMULATION
    for t in range(1,nsteps):
        Xpre = Xdot[t-1]

        # Calculate the input to nodes due to couplings
        xcoup = np.dot(conT,Xpre) - ink * Xpre
        # xcoup = np.dot(Lmat, Xpre)

        # Integration step
        # Xdot[t] = Xpre + timestep * ( gcoupling*xcoup ) + noise[t]
        Xdot[t] = Xpre + timestep * xcoup + noise[t]

    return Xdot





##
