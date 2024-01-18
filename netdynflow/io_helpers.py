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
Helper functions for handling inputs and checks
===============================================

This module contains functions to help IO operations, specially to carry the
validation checks for the user inputs (parameters to functions) and ensure all
relevant arrays are given in the correct data type.

Input handling
--------------
function_name
    Description here.
function_name
    Description here.
function_name
    Description here.

"""
# Standard libary imports
import numbers
# Third party packages
import numpy as np
import numpy.random



## INPUT HANDLING FUNCTIONS ###################################################

# VALIDATE TIMES ??
# if tmax <= 0.0: raise ValueError("'tmax' must be positive")
# if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
# if timestep > tmax: raise ValueError("Incompatible values, timestep < tmax given")


def validate_con(a):
    """
    """
    # Make sure 'con' is a numpy array, of np.float64 dtype
    if isinstance(a, np.ndarray): pass
    else:
        raise TypeError( "'con' must be numpy array, but %s found" %type(a) )

    # Make sure 'con' is a 2D array
    conshape = np.shape(a)
    if np.ndim(a)==2 and conshape[0]==conshape[1]: pass
    else:
        raise ValueError( "'con' must be a square matrix, but shape %s found" %str(np.shape(a)) )
    # return a

def validate_X0(a, n_nodes):
    """
    """
    # Make sure 'X0' is a numpy array, of np.float64 dtype
    if isinstance(a, numbers.Number) and type(a) != bool:
        a = a * np.ones(n_nodes, np.float64)
    elif isinstance(a, np.ndarray): pass
    else:
        raise TypeError(
        "'X0' must be either scalar or numpy array, but %s found" %type(a) )

    # Make sure 'X0' is a 1D array
    if np.ndim(a) != 1:
        raise ValueError(
        "'X0' must be 1-dimensional of length N, but shape %s found"
        %str(np.shape(a)) )

    return a

def validate_tau(a, n_nodes):
    """
    """
    # Make sure 'tau' is a numpy array, of np.float64 dtype
    if isinstance(a, numbers.Number) and type(a) != bool:
        a = a * np.ones(n_nodes, np.float64)
    elif isinstance(a, np.ndarray): pass
    else:
        raise TypeError(
        "'tau' must be either scalar or numpy array, but %s found" %type(a) )

    # Make sure 'tau' is a 1D array
    if np.ndim(a) != 1:
        raise ValueError(
        "'tau' must be 1-dimensional of length N, but shape %s found"
        %str(np.shape(a)) )

    return a


# def validate_scalar_1darr(a, n_nodes):
#     """
#     """
#     # Get the global name of parameter 'a'
#     ## This doesn't work. I dunno why because it work in other examples :(
#     ## If working, I could use it for different arrays instead of having
#     ## on function per array (e.g., X0 and tau)
#     id_a = id(a)
#     localdict = locals()
#     for name in localdict.keys():
#         if id(localdict[name]) == id_a:
#             lname_a = name
#
#     globaldict = globals()
#     print(len(globaldict))
#     for name in globaldict.keys():
#         if id(globaldict[name]) == id_a:
#             gname_a = name
#
#     print(lname_a, gname_a)
#
#     # Make sure 'a' is a numpy array, of np.float64 dtype
#     if isinstance(a, numbers.Number):
#         a = a * np.ones(n_nodes, np.float64)
#     if isinstance(a, np.ndarray): pass
#     elif isinstance(a, (list,tuple)):
#         a = np.array(a, np.float64)
#     else:
#         raise TypeError( "'%s' must be either scalar or array-like (ndarray, list, tuple)" %gname_a)
#
#     # Make sure 'a' is a 1D array
#     if np.ndim(a) != 1:
#         raise ValueError( "'%s' must be a 1-dimensional array of length N" %gname_a)
#
#     return a


def validate_noise(a, n_nodes, tmax, timestep):
    """
    """
    # When nothing is given by user, skip noise generation
    if a is None:
        pass
    # When 'noise' is a scalar ...
    elif isinstance(a, numbers.Number):
        if not a:
            # If zero or False, do nothing
            a = None
            pass
        elif a < 0:
            # 'noise' must be positive
            raise ValueError( "'noise' amplitude must be positive, %f found" %a )
        else:
            # If positive scalar given, generate the array for the noise
            namp = a
            nnorm = np.sqrt(2.0 * namp * timestep)
            nt = int(tmax / timestep) + 1
            a = nnorm * numpy.random.randn(nt, n_nodes)
    # Make sure 'noise' is a numpy array, of np.float64 dtype
    elif isinstance(a, np.ndarray):
        pass
    else:
        raise TypeError(
        "'noise' must be None, scalar or numpy array, but %s found" %type(a) )

    # Make sure 'noise' is a 2D array
    if a is not None and np.ndim(a) != 2:
        raise ValueError(
        "'noise' must be a 2-dimensional, but shape %s found" %str(np.shape(a)) )

    return a





##
