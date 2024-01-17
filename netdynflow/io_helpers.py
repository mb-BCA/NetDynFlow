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



## INPUT HANDLING FUNCTIONS ###################################################
def validate_con(a):
    """
    """
    # Make sure 'con' is a numpy array, of np.float64 dtype
    if isinstance(a, np.ndarray):
        if a.dtype != np.float64:
            a = a.astype(np.float64)
    elif isinstance(a, (list,tuple)):
        a = np.array(a, np.float64)
    else:
        raise TypeError( "'con' shall be array-like (ndarray, list or tuple)" )

    # Make sure 'con' is a 2D array
    value_err_msg = "'con' shall be a square, 2-dimensional array"
    if np.ndim(a) != 2:
        raise ValueError( value_err_msg )
    else:
        n1,n2 = np.shape(a)
        if n1 != n2:
            raise ValueError( value_err_msg )

    return a

def validate_X0(a, n_nodes):
    """
    """
    # Make sure 'X0' is a numpy array, of np.float64 dtype
    if isinstance(a, np.ndarray):
        if a.dtype != np.float64:
            a = a.astype(np.float64)
    elif isinstance(a, (list,tuple)):
        a = np.array(a, np.float64)
    else:
        raise TypeError( "'X0' shall be array-like (ndarray, list or tuple)" )

    # Make sure 'X0' is a 1D array of length N
    value_err_msg = "'X0' shall be a 1-dimensional array of length N"
    if np.ndim(a) != 1:
        raise ValueError( value_err_msg )
    else:
        if len(a) != n_nodes:
            raise ValueError( "'X0' and 'con' not aligned" )

    return a

def validate_tau(a, n_nodes):
    """
    """
    # Check whether tau is given as an scalar or an ndarray
    if isinstance(a, numbers.Number):
        a = a * np.ones(N, np.float64)
    elif isinstance(a, np.ndarray):
        if a.dtype != np.float64:
            a = a.astype(np.float64)
    elif isinstance(a, (list,tuple)):
        a = np.array(a, dtype=np.float64)
    else:
        raise TypeError( "'tau' shall be array-like (ndarray, list or tuple)" )

    # Make sure 'tau' is a 1D array of length N
    value_err_msg = "'tau' shall be a either a scalar or a 1-dimensional array of length N"
    if np.ndim(a) != 1:
        raise ValueError( value_err_msg )
    else:
        if len(a) != n_nodes:
            raise ValueError( "'tau' and 'con' not aligned" )

    return a

def validate_noise(a, n_steps, n_nodes):
    """
    """
    # If nothing given by user, set noise to zeros
    if a is None:
        a = np.zeros((n_steps,n_nodes), np.float64)
    # Make sure 'noise' is a numpy array, of np.float64 dtype
    elif isinstance(a, np.ndarray):
        if a.dtype != np.float64:
            a = a.astype(np.float64)
    elif isinstance(a, (list,tuple)):
        a = np.array(a, dtype=np.float64)
    else:
        raise TypeError( "'noise' shall be array-like (ndarray, list or tuple)" )

    # Make sure 'noise' is a 2D array of shape (nsteps,N)
    value_err_msg = "'noise' shall be a 2-dimensional array of shape (nsteps,N)"
    if np.ndim(a) != 2:
        raise ValueError(value_err_msg)
    else:
        n1,n2 = np.shape(a)
        if n1 != n_steps:
            raise ValueError( value_err_msg )
        if  n2 != n_nodes:
            raise ValueError( "'noise' and 'con' not aligned" )

    return a








##
