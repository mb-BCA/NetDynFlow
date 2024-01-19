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
# Local imports from NetDynFlow
from . import io_helpers



## DISCRETE-TIME CANONICAL MODELS #############################################
def DiscreteCascade(con, X0=1, tmax=10):
    """Simulates temporal evolution of the nodes for the discrete cascade.

    It returns the time-series of the nodes for the discrete cascade model

            x(t+1) = A x(t),

    If A is a positive definite connectivity  matrix, then the solutions
    xt grow exponentially fast.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    X0 : scalar or ndarray (1d) of length N, optional
        Initial values of the nodes at time t = 0. If scalar value is entered,
        `X0 = c`, all nodes are initialised as `X0[i] = c` (same initial conditions).
        Default value, `X0 = 1.0`. If a 1d-array is entered, each node i is
        assigned initial value `X0[i]`.
    tmax : integer, optional
        The duration of the simulation, discrete time steps.

    Returns
    -------
    Xt : ndarray (2d)
        Time-courses of the N nodes. A numpy array of shape (tmax+1, N).
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    X0 = io_helpers.validate_X0(X0, N)

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Tanspose the connectivity matrix
    conT = np.copy(con.T, order='C')
    # Initialise the output array and enter the initial conditions
    Xt = np.zeros((tmax+1,N), np.float64)
    Xt[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,tmax+1):
        Xt[t] = np.dot(conT, Xt[t-1])

    return Xt

def RandomWalk(con, X0=1, tmax=10):
    """Simulates temporal evolution of the nodes for the random walks.

    It returns the time-series of the nodes for the discrete cascade model

            x(t+1) = T x(t),

    where T is the transition probability matrix. The solutions are computed
    recursively iterating the equation above. If A is a positive definite
    connectivity matrix the solutions converge to x_i(inf) values proportional
    to the input degree of the nodes.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    X0 : scalar or ndarray (1d) of length N, optional
        Number of walkers (agents) starting at each node. If scalar value is
        entered, 'X0 = c', all nodes are initialised with c walkers (same
        initial conditions). Default value, X0 = 1.0. If a 1d-array is entered,
        X0[i] walkers are initialised at each node.
    tmax : integer, optional
        The duration of the simulation, discrete time steps.

    Returns
    -------
    Xt : ndarray (2d)
        Time-courses of the N nodes. A numpy array of shape (tmax+1, N).
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    X0 = io_helpers.validate_X0(X0, N)

    # Ensure all arrays are of same dtype (float64)
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
def ContinuousCascade(con, X0=1, noise=None, tmax=10, timestep=0.01):
    """Simulates the temporal evolution of the nodes for the continuous cascade.

    It solves the differential equation for the simplest possible linear
    dynamical (propagation) model of nodes coupled via connectivity matrix A,
    with no local dynamics at the nodes.

            xdot = A x(t).

     If A is a positive definite connectivity  matrix, then the solutions
     xdot(t) grow exponentially fast.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    X0 : scalar or ndarray (1d) of length N, optional
        Initial values of the nodes at time t = 0. If scalar value is entered,
        `X0 = c`, all nodes are initialised as `X0[i] = c` (same initial conditions).
        Default value, `X0 = 1.0`. If a 1d-array is entered, each node i is
        assigned initial value `X0[i]`.
    noise : None, scalar or ndarray (2d) of shape (nt,N), optional
        Additive noise. If `noise = None` simulation is run without noise.
        If scalar `noise = c`, a Gaussian white noise, centered at zero and
        variance 'c' is applied to all nodes. Independent for each node.
        Also, `noise = arr` accepts ndarray (arr) of shape (nt, N) with noise
        signals precomputed by the user.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Time-step of the numerical integration.

    Returns
    -------
    Xdot : ndarray (2d) of shape (nt,N)
        Time-courses of the N nodes.

    NOTE
    ----
    Total number of integration steps (samples) is nt = int(tmax*timestep) + 1.
    - Simulation runs from t=0 to t=tmax.
    - Integration goes from it=0 to it=nt, with `Xdot[0] = X0`.
    - The sampled time points are `tpoints = np.arange(0,tmax+timestep,timestep)`
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    X0 = io_helpers.validate_X0(X0, N)
    noise = io_helpers.validate_noise(noise, N,tmax,timestep)

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)
    if noise is not None and noise.dtype != np.float64:
        noise = noise.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Transpose the connectivity matrix
    conT = np.copy(con.T, order='C')
    # Initialise the output array
    nt = int(tmax / timestep) + 1
    Xdot = np.zeros((nt, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    if noise is None:
        for t in range(1,nt):
            Xpre = Xdot[t-1]
            # Calculate the input to nodes due to couplings
            xcoup = np.dot(conT,Xpre)
            # Integration step
            Xdot[t] = Xpre + timestep * xcoup #+ noise[t]
    else:
        for t in range(1,nt):
            Xpre = Xdot[t-1]
            # Calculate the input to nodes due to couplings
            xcoup = np.dot(conT,Xpre)
            # Integration step
            Xdot[t] = Xpre + timestep * xcoup + noise[t]

    return Xdot

def LeakyCascade(con, X0=1, tau, noise=None, tmax=10, timestep=0.01):
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
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    X0 : scalar or ndarray (1d) of length N, optional
        Initial values of the nodes at time t = 0. If scalar value is entered,
        `X0 = c`, all nodes are initialised as `X0[i] = c` (same initial conditions).
        Default value, `X0 = 1.0`. If a 1d-array is entered, each node i is
        assigned initial value `X0[i]`.
    tau : real value or ndarray of rank-1 and length N.
        The decay time-constants of the nodes. If a single number is entered
        (e.g., tau = 2), then all nodes will be assigned the same value
        (e.g., tau[i] = 2). If an array is entered, then entries tau[i] will be
        considered as the decay time-constant for each node.
    noise : None, scalar or ndarray (2d) of shape (nt,N), optional
        Additive noise. If `noise = None` simulation is run without noise.
        If scalar `noise = c`, a Gaussian white noise, centered at zero and
        variance 'c' is applied to all nodes. Independent for each node.
        Also, `noise = arr` accepts ndarray (arr) of shape (nt, N) with noise
        signals precomputed by the user.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Time-step of the numerical integration.

    Returns
    -------
    Xdot : ndarray (2d) of shape (nt,N)
        Time-courses of the N nodes.

    NOTE
    ----
    Total number of integration steps (samples) is nt = int(tmax*timestep) + 1.
    - Simulation runs from t=0 to t=tmax.
    - Integration goes from it=0 to it=nt, with `Xdot[0] = X0`.
    - The sampled time points are `tpoints = np.arange(0,tmax+timestep,timestep)`
    """
    # 0) HANDLE AND CHECK THE INPUTS. Ensure all arrays are of same dtype
    io_helpers.validate_con(con)
    N = len(con)
    X0 = io_helpers.validate_X0(X0, N)
    tau = io_helpers.validate_tau(tau, N)
    noise = io_helpers.validate_noise(noise, N,tmax,timestep)

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)
    if tau.dtype != np.float64:    tau = tau.astype(np.float64)
    if noise is not None and noise.dtype != np.float64:
        noise = noise.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Transpose the connectivity matrix
    conT = np.copy(con.T, order='C')
    # Conver the time-constants into decay rations
    alphas = 1./tau
    # Initialise the output array
    nt = int(tmax / timestep) + 1
    Xdot = np.zeros((nt, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    if noise is None:
        for t in range(1,nt):
            Xpre = Xdot[t-1]
            # Calculate the input to nodes due to couplings
            xcoup = np.dot(conT,Xpre)
            # Integration step
            Xdot[t] = Xpre + timestep * ( -1.0*alphas * Xpre + xcoup ) #+ noise[t]
    else:
        for t in range(1,nt):
            Xpre = Xdot[t-1]
            # Calculate the input to nodes due to couplings
            xcoup = np.dot(conT,Xpre)
            # Integration step
            Xdot[t] = Xpre + timestep * ( -1.0*alphas * Xpre + xcoup ) + noise[t]

    return Xdot

def ContinuousDiffusion(con, X0=1, noise=None, tmax=10, timestep=0.01):
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
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    X0 : scalar or ndarray (1d) of length N, optional
        Initial values of the nodes at time t = 0. If scalar value is entered,
        `X0 = c`, all nodes are initialised as `X0[i] = c` (same initial conditions).
        Default value, `X0 = 1.0`. If a 1d-array is entered, each node i is
        assigned initial value `X0[i]`.
    noise : None, scalar or ndarray (2d) of shape (nt,N), optional
        Additive noise. If `noise = None` simulation is run without noise.
        If scalar `noise = c`, a Gaussian white noise, centered at zero and
        variance 'c' is applied to all nodes. Independent for each node.
        Also, `noise = arr` accepts ndarray (arr) of shape (nt, N) with noise
        signals precomputed by the user.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Time-step of the numerical integration.

    Returns
    -------
    Xdot : ndarray (2d) of shape (nt,N)
        Time-courses of the N nodes.

    NOTE
    ----
    Total number of integration steps (samples) is nt = int(tmax*timestep) + 1.
    - Simulation runs from t=0 to t=tmax.
    - Integration goes from it=0 to it=nt, with `Xdot[0] = X0`.
    - The sampled time points are `tpoints = np.arange(0,tmax+timestep,timestep)`
    """
    # 0) HANDLE AND CHECK THE INPUTS. Ensure all arrays are of same dtype
    io_helpers.validate_con(con)
    N = len(con)
    X0 = io_helpers.validate_X0(X0, N)
    noise = io_helpers.validate_noise(noise, N,tmax,timestep)

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)
    if noise is not None and noise.dtype != np.float64:
        noise = noise.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Transpose the connectivity matrix
    conT = np.copy(con.T, order='C')
    ink = conT.sum(axis=1)
    # Lmat = - ink * np.identity(N, dtype=np.float64) + conT
    # Initialise the output array
    nt = int(tmax / timestep) + 1
    Xdot = np.zeros((nt, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    if noise in None:
        for t in range(1,nt):
            Xpre = Xdot[t-1]
            # Calculate the input to nodes due to couplings
            xcoup = np.dot(conT,Xpre) - ink * Xpre
            # xcoup = np.dot(Lmat, Xpre)
            # Integration step
            Xdot[t] = Xpre + timestep * xcoup #+ noise[t]
    else:
        for t in range(1,nt):
            Xpre = Xdot[t-1]
            # Calculate the input to nodes due to couplings
            xcoup = np.dot(conT,Xpre) - ink * Xpre
            # xcoup = np.dot(Lmat, Xpre)
            # Integration step
            Xdot[t] = Xpre + timestep * xcoup + noise[t]

    return Xdot





##
