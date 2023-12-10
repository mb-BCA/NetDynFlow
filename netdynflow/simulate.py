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
    Simulates the temporal evolution of the nodes for the discrete cascade.
RandomWalks
    Simulates the temporal evolution of the nodes for the random walks.

Continuous-time models
-----------------------
ContinuousCascade
    Simulates the temporal evolution of the nodes for the continuous cascade.
LeakyCascade
    Simulates the temporal evolution of the nodes for the leaky-cascade model.
ContinuousDiffusion
    Simulates the temporal evolution of the nodes for the simple diffustion model.

"""
# Standard libary imports

# Third party packages
import numpy as np
# import numpy.linalg
# import scipy.linalg



## DISCRETE-TIME MODELS #######################################################
# Add the missing models here ...



## CONTINUOUS-TIME MODELS #####################################################

def ContinuousCascade(con, X0=None, tfinal=10, dt=0.01, gcoupling=1.0, noise=None):
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
    tfinal : scalar (optional)
        The duration of the simulation in arbitrary time units.
    dt : scalar (optional)
        The time-step of the numerical integration.
    gcoupling : scalar (optional)
        A global coupling strength scaling the weights of all the links.
    noise : ndarray of rank-2 (optional)
        A precomputed noisy input to all nodes. Optional parameter. Not needed
        for the applications of the model-based network analysis. Left optional
        for general purposes. If given, 'noise' shall be a numpy array of shape
        (nsteps, N), where nsteps = int(tfinal*dt+1) and N is the number of nodes.

    Returns
    -------
    Xdot : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (nsteps, N),
        where nsteps = int(tfinal*dt) + 1.
    """
    # 0) SECURITY CHECKS
    # To be done ...

    # 1) PREPARE FOR THE SIMULATION
    # Infos about the network
    N = len(con)
    conT = np.copy(con.T, order='C')

    # Initialise the output array
    nsteps = int(tfinal / dt) + 1
    Xdot = np.zeros((nsteps, N), np.float, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,nsteps):
        Xpre = Xdot[t-1]

        # Calculate the input to nodes due to couplings
        xcoup = np.dot(conT,Xpre) #- ink * Xpre

        # Integration step
        Xdot[t] = Xpre + dt * ( gcoupling*xcoup ) + noise[t]

    return Xdot

def ContinuousDiffusion(con, X0=None, tfinal=10, dt=0.01, gcoupling=1.0, noise=None):
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
    tfinal : scalar (optional)
        The duration of the simulation in arbitrary time units.
    dt : scalar (optional)
        The time-step of the numerical integration.
    gcoupling : scalar (optional)
        A global coupling strength scaling the weights of all the links.
    noise : ndarray of rank-2 (optional)
        A precomputed noisy input to all nodes. Optional parameter. Not needed
        for the applications of the model-based network analysis. Left optional
        for general purposes. If given, 'noise' shall be a numpy array of shape
        (nsteps, N), where nsteps = int(tfinal*dt) + 1 and N is the number of nodes.

    Returns
    -------
    Xdot : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (nsteps, N),
        where nsteps = int(tfinal*dt) + 1.
    """
    # 0) SECURITY CHECKS
    # To be done ...

    # 1) PREPARE FOR THE SIMULATION
    # Infos about the network
    N = len(con)
    conT = np.copy(con.T, order='C').astype(np.float64)
    ink = conT.sum(axis=1)
    # Lmat = - ink * np.identity(N, dtype=np.float64) + conT

    # Initialise the output array
    nsteps = int(tfinal / dt) + 1
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
        Xdot[t] = Xpre + dt * ( gcoupling*xcoup ) + noise[t]

    return Xdot

def LeakyCascade(con, taus, tfinal=10, dt=0.01, gcoupling=1.0, noise=None):
    """Simulates the temporal evolution of the nodes for the continuous cascade.

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
    taus : ndarray of rank-1
        The decay time-constants of the nodes. A 1D array of length N.
    X0 : ndarray of rank-1. (optional)
        The initial conditions for the simulation. A vector of length N nodes.
        If none given, simulation will start with unit input to all nodes, X0 = 1.
    tfinal : scalar (optional)
        The duration of the simulation in arbitrary time units.
    dt : scalar (optional)
        The time-step of the numerical integration.
    gcoupling : scalar (optional)
        A global coupling strength scaling the weights of all the links.
    noise : ndarray of rank-2 (optional)
        A precomputed noisy input to all nodes. Optional parameter. Not needed
        for the applications of the model-based network analysis. Left optional
        for general purposes. If given, 'noise' shall be a numpy array of shape
        (nsteps, N), where nsteps = int(tfinal*dt+1) and N is the number of nodes.

    Returns
    -------
    Xdot : ndarray of rank-2
        Time-courses of the N nodes. A numpy array of shape (nsteps, N),
        where nsteps = int(tfinal*dt) + 1.
    """

    # 0) SECURITY CHECKS
    # To be done ...

    # 1) PREPARE FOR THE SIMULATION
    # Infos about the network
    N = len(con)
    conT = np.copy(con.T, order='C')
    # ink = conT.sum(axis=1)
    alphas = 1./taus

    # Initialise the output array
    nsteps = int(tfinal / dt) + 1
    Xdot = np.zeros((nsteps, N), np.float, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,nsteps):
        Xpre = Xdot[t-1]

        # Calculate the input to nodes due to couplings
        xcoup = np.dot(conT,Xpre) #- ink * Xpre

        # Integration step
        Xdot[t] = Xpre + dt * ( -1.0*alphas * Xpre + gcoupling*xcoup ) + noise[t]

    return Xdot






##
