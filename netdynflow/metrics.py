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
Analysis of dynamic communicability and flow
============================================

Functions to analyse the spatio-temporal evolution of perturbation-induced
flows in a network. The data shall be given in the form of 3D tensor arrays
previously calculated from a given network. See 'core.py' module.

Metrics derived from the tensors
--------------------------------
TotalEvolution
    Calculates total amount of flow through the network at every time point.
NodeFlows
    Temporal evolution of the input and output flows for each node.
Diversity
    Inhomogeneity of the pair-wise flows in a networks over time.
TimeToPeak
    The time for links, nodes or networks need to reach maximal flow.
TimeToDecay
    The time need for pair-wise, nodes or networks flows to decay.
AreaUnderCurve
    Total amount of flow across time for pair-wise, node or whole-network.


Reference and Citation
----------------------
1. M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).
2. M. Gilson, N. Kouvaris, et al. "Network analysis of whole-brain fMRI
dynamics: A new framework based on dynamic communicability" NeuroImage 201,
116007 (2019).

"""
# Standard library imports

# Third party packages
import numpy as np
import numpy.linalg
import scipy.linalg


## METRICS EXTRACTED FROM THE FLOW AND COMMUNICABILITY TENSORS ################
def TotalEvolution(tensor):
    """
    Calculates total amount of flow through the network at every time point.

    Parameters
    ----------
    tensor : ndarray of rank-3
        Spatio-temporal evolution of the network's flow. A tensor of shape
        (nt,N,N) is expected, where N is the number of nodes and nt is the
        number of time steps.

    Returns
    -------
    totaldyncom : ndarray of rank-1
        Array containing temporal evolution of the total network flow.
    """

    # 0) SECURITY CHECKS
    # Check the input tensor has the correct 3D shape
    tensor_shape = np.shape(tensor)
    if (len(tensor_shape) != 3) or (tensor_shape[1] != tensor_shape[2]):
        raise ValueError("Input array not aligned. A 3D array of shape (N x N x nt) expected.")

    totaldyncom = tensor.sum(axis=(1,2))

    return totaldyncom

def Diversity(tensor):
    """
    Inhomogeneity of the pair-wise flows in a networks over time.

    Parameters
    ----------
    tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability or flow.
        A tensor of shape (nt,N,N) is expected, where N is the number of nodes
        and nt is the number of time steps.

    Returns
    -------
    diversity : ndarray of rank-1
        Array containing temporal evolution of the pair-wise flow inhomogeneities.
    """
    # 0) SECURITY CHECKS
    # Check the input tensor has the correct 3D shape
    tensor_shape = np.shape(tensor)
    if (len(tensor_shape) != 3) or (tensor_shape[1] != tensor_shape[2]):
        raise ValueError("Input array not aligned. A 3D array of shape (nt x N x N) expected.")

    nt = tensor_shape[0]
    diversity = np.zeros(nt, np.float)
    diversity[0] = np.nan
    for i_t in range(1,nt):
        temp = tensor[i_t]
        diversity[i_t] = temp.std() / temp.mean()

    return diversity

def NodeFlows(tensor, selfloops=False):
    """
    Temporal evolution of the input and output flows for each node.

    Parameters
    ----------
    tensor : ndarray of rank-3
        Temporal evolution of the network's dynamic communicability. A tensor
        of shape (nt,N,N) is expected, where N is the number of nodes and nt is
        the number of time steps.
    selfloops : boolean
        If False (default), the function only returns the in-flows into a node
        due to perturbations on other  nodes, and the out-flows that the node
        causes on other nodes, excluding the initial perturbations on itself.
        If True, the function includes the effect of the perturbation on itself
        for the calculation of the input and output flows.

    Returns
    -------
    nodeflows : tuple.
        Temporal evolution of the input and output flows for all nodes.
        The result consists of a tuple containing two ndarrays of shape (nt,N).
        The first is for the input flows into a node over time and the second
        array for the output flows.
    """

    # 0) SECURITY CHECKS
    # Check the input tensor has the correct 3D shape
    arr_shape = np.shape(tensor)
    if (len(arr_shape) != 3) or (arr_shape[1] != arr_shape[2]):
        raise ValueError("Input array not aligned. A 3D array of shape (nt x N x N) expected.")

    # 1) Calculate the input and output node properties
    # When self-loops shall be included to the temporal nodel flows
    if selfloops:
        inflows = tensor.sum(axis=1)
        outflows = tensor.sum(axis=2)

    # Excluding the self-flows a node due to inital perturbation on itself.
    else:
        nt, N,N = arr_shape
        inflows = np.zeros((nt,N), np.float)
        outflows = np.zeros((nt,N), np.float)
        for i in range(N):
            tempdiags = tensor[:,i,i]
            inflows[:,i]  = tensor[:,:,i].sum(axis=1) - tempdiags
            outflows[:,i] = tensor[:,i,:].sum(axis=1) - tempdiags

    nodeflows = ( inflows, outflows )
    return nodeflows

def Time2Peak(arr, timestep):
    """
    The time for links, nodes or networks need to reach maximal flow.

    In terms of binary graphs, time-to-peak is equivalen to the pathlength
    between two nodes.

    The function calculates the time-to-peak for either links, nodes or the
    whole network, depending on the input array given.
    - If 'arr' is a (nt,N,N) flow tensor, the output 'ttp_arr' will be an
    (N,N) matrix with the ttp between every pair of nodes.
    - If 'arr' is a (nt,N) temporal flow of the N nodes, the output 'ttp_arr'
    will be an array of length N, containing the ttp of all N nodes.
    - If 'arr' is an array of length nt (total network flow over time), 'ttp_arr'
    will be a scalar, indicating the time at which the whole-network flow peaks.

    Parameters
    ----------
    arr : ndarray of adaptive shape, according to the case.
        Temporal evolution of the flow. An array of optional shapes. Either
        (nt,N,N) for the pair-wise flows, shape (nt,N,N) for the in- or output
        flows of nodes, or a 1D array of length nt for the network flow.
    timestep : real valued number.
        Sampling time-step. This has to be the time-step employed to simulate
        the temporal evolution encoded in 'arr'.

    Returns
    -------
    ttp_arr : ndarray of variable rank
        The time(s) taken for links, nodes or the network to reach peak flow.
        Output shape depends on input.
    """

    # 0) SECURITY CHECKS
    ## TODO1: Write a check to verify the curve has a real peak and decays after
    ## the peak. Raise a warning that maybe longer simulation is needed.
    ## TODO2: Silent nodes (non-perturbed) should return inf instead of zero.
    # Check correct shape, in case input is the 3D array for the pair-wise flow
    arr_shape = np.shape(arr)
    if arr_shape==3:
        if arr_shape[1] != arr_shape[2]:
            raise ValueError("Input array not aligned. For 3D arrays shape (nt x N x N) is expected.")

    # 1) Get the indices at which every element peaks
    ttp_arr = arr.argmax(axis=0)
    # 2) Convert into simulation time
    ttp_arr = timestep * ttp_arr

    return ttp_arr

def Time2Decay(arr, dt, fraction=0.99):
    """
    The time need for pair-wise, nodes or networks flows to decay.

    Strictly speaking, this function measures the time that the cumulative
    flow (area under the curve) needs to reach x% of the total (cumulative)
    value. Here 'x%' is controled by the optional parameter 'fraction'.
    For example, 'fraction = 0.99' means the time needed to reach 99%
    of the area under the curve, given a response curve.

    The function calculates the time-to-decay either for all pair-wise
    interactions, for the nodes or for the whole network, depending on the
    input array given.
    - If 'arr' is a (nt,N,N) flow tensor, the output 'ttd_arr' will be an
    (N,N) matrix with the ttd between every pair of nodes.
    - If 'arr' is a (nt,N) temporal flow of the N nodes, the output 'ttd_arr'
    will be an array of length N, containing the ttd of all N nodes.
    - If 'arr' is an array of length nt (total network flow over time), 'ttd_arr'
    will be a scalar, indicating the time at which the whole-network flow decays.

    Parameters
    ----------
    arr : ndarray of adaptive shape, according to the case.
        Temporal evolution of the flow. An array of optional shapes. Either
        (nt,N,N) for the pair-wise flows, shape (nt,N,N) for the in- or output
        flows of nodes, or a 1D array of length nt for the network flow.
    timestep : real valued number.
        Sampling time-step. This has to be the time-step employed to simulate
        the temporal evolution encoded in 'arr'.
    fraction : scalar, optional
        The fraction of the total area-under-the-curve to be reached.
        For example, 'fraction = 0.99' means the time the flow needs to
        reach 99% of the area under the curve.

    Returns
    -------
    ttd_arr : ndarray of variable rank
        The time(s) taken for the flows through links, nodes or the network to
        decay. Output shape depends on input.
    """

    # 0) SECURITY CHECKS
    ## TODO: Write a check to verify the curve(s) has (have) really decayed back
    ## to zero. At this moment, it is the user's responsability to guarantee
    ## that all the curves have decayed reasonably well.
    ## The check should rise a warning to simulate for longer time.

    # Check correct shape, in case input is the 3D array for the pair-wise flow
    arr_shape = np.shape(arr)
    if arr_shape==3:
        if arr_shape[1] != arr_shape[2]:
            raise ValueError("Input array not aligned. For 3D arrays shape (nt x N x N) is expected.")

    # 1) Set the level of cummulative flow to be reached over time
    targetcflow = fraction * arr.sum(axis=0)

    # 2) Calculate the time the flow(s) need to decay
    # Initialise the output array, to return the final time-point
    ## NOTE: This version iterates over all the times. This is not necessary.
    ## We could start from the end and save plenty of iterations.
    ttd_shape = arr_shape[1:]
    nsteps = arr_shape[0]
    ttd_arr = nsteps * np.ones(ttd_shape, np.int)

    # Iterate over time, calculating the cumulative flow(s)
    cflow = arr[0].copy()
    for t in range(1,nsteps):
        cflow += arr[t]
        ttd_arr = np.where(cflow < targetcflow, t, ttd_arr)

    # Finally, convert the indices into integration time
    ttd_arr = ttd_arr.astype(np.float) * dt

    return ttd_arr

def AreaUnderCurve(arr, timestep, timespan='alltime'):
    ## TODO: The name of this function needs good thinking. Different options
    ## are possible depending on the interpretation and naming of other
    ## variables or metrics.
    ## The most explicit would be to call it "AreaUnderCurve()" because that is
    ## exactly what it does. But that doesn't sound very sexy nor hints on the
    ## interpretation of what it measures, given that 'arr' will be the
    ## temporal evolution of a flow, response curve or dyncom ... which are
    ## indeed the same !! Maybe.
    """
    Total amount of flow across time for pair-wise, node or whole-network.

    The function calculates the area-under-the-curve for the flow curves over
    time. It does so for all pair-wise interactions, for the nodes or for
    the whole network, depending on the input array given.
    - If 'arr' is a (nt,N,N) flow tensor, the output 'totalflow' will be an
    (N,N) matrix with the accumulated flow passed between every pair of nodes.
    - If 'arr' is a (nt,N) temporal flow of the N nodes, the output 'totalflow'
    will be an array of length N, containing the accumulated flow passed through
    all the nodes.
    - If 'arr' is an array of length nt (total network flow over time), 'totalflow'
    will be a scalar, indicating the total amount of flow that went through the
    whole network.

    Parameters
    ----------
    arr : ndarray of adaptive shape, according to the case.
        Temporal evolution of the flow. An array of shape nt x N x N for the
        flow of the links, an array of shape N X nt for the flow of the nodes,
        or a 1-dimensional array of length nt for the network flow.
    timestep : real valued number.
        Sampling time-step. This has to be the time-step employed to simulate
        the temporal evolution encoded in 'arr'.
    timespan : string, optional
        If timespan = 'alltime', the function calculates the area under the
        curve(s) along the whole time span (nt) that 'arr' contains, from t0 = 0
        to tfinal.
        If timespan = 'raise', the function calculates the area-under-the-
        curve from t0 = 0, to the time the flow(s) reach a peak value.
        If timespan = 'decay', it returns the area-under-the-curve for the
        time spanning from the time the flow peaks, until the end of the signal.

    Returns
    -------
    totalflow : ndarray of variable rank
        The accumulated flow (area-under-the-curve) between pairs of nodes,
        by nodes or by the whole network, over a period of time.
    """

    # 0) SECURITY CHECKS
    ## TODO: Write a check to verify the curve has a real peak and decays after
    ## the peak. Raise a warning that maybe longer simulation is needed.

    # Check correct shape, in case input is the 3D array for the pair-wise flow
    arr_shape = np.shape(arr)
    if arr_shape==3:
        if arr_shape[1] != arr_shape[2]:
            raise ValueError("Input array not aligned. For 3D arrays shape (nt x N x N) is expected.")

    # Validate options for optional variable 'timespan'
    caselist = ['alltime', 'raise', 'decay']
    if timespan not in caselist :
        raise ValueError( "Optional parameter 'timespan' requires one of the following values: %s" %str(caselist) )

    # 1) DO THE CALCULATIONS
    # 1.1) Easy case. Integrate area-under-the-curve along whole time interval
    if timespan == 'alltime':
        totalflow = timestep * arr.sum(axis=0)

    # 1.2) Integrate area-under-the-curve until or from the peak time
    else:
        # Get the temporal indices at which the flow(s) peak
        tpidx = arr.argmax(axis=0)

        # Initialise the final array
        tf_shape = arr_shape[1:]
        totalflow = np.zeros(tf_shape, np.float)

        # Sum the flow(s) over time, only in the desired time interval
        nsteps = arr_shape[0]
        for t in range(1,nsteps):
            # Check if the flow at time t should be accounted for or ignored
            if timespan == 'raise':
                counts = np.where(t < tpidx, True, False)
            elif timespan == 'decay':
                counts = np.where(t < tpidx, False, True)
            # Sum the flow at the given iteration, if accepted
            totalflow += (counts * arr[t])

        # Finally, normalise the integral by the time-step
        totalflow *= timestep

    return totalflow





##
