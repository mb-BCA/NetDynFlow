# -*- coding: utf-8 -*-
# Copyright (c) 2021, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris
# <galib@Zamora-Lopez.xyz>
#
# Released under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Network and surrogate generation module
=======================================

Functions to construct synthetic networks and generate surrogates out of given
binary or weighted networks.

Generation and randomization of binary graphs
---------------------------------------------
These functions are all imported from the GAlib library. Please see doctsring
of module 'galib.models' for further documentation and a list of functions.  ::

    >>> import galib
    >>> help(galib.models)

Deterministic network models
----------------------------
Function1
    Short description of the function.
Function2
    Short description of the function.

Random network models
----------------------
Function1
    Short description of the function.
Function2
    Short description of the function.

Weighted surrogate networks
---------------------------
ShuffleLinkWeights
    Randomly re-allocates the weights associated to the links.

"""
# Standard library imports
from __future__ import division, print_function
# Third party packages
import numpy as np
import numpy.linalg
import scipy.linalg
# Import GAlib for graph analysis and graph generation tools
### ACHTUNG!! For now I am importing GAlib but, for the long run, we must
### decide whether we want GAlib as a dependency of NetDynFlow, or we
### prefer to copy/paste the useful functions from GAlib into this module and
### therefore make NetDynFlow independent of GAlib.
import galib
from galib.models import*


## DETERMINISTIC NETWORK MODELS ################################################



## RANDOM NETWORK MODELS #######################################################



## GENERATION OF SURROGATE NETWORKS ############################################
def ShuffleLinkWeights(con):
    """
    Randomly re-allocates the link weights of an input network.

    The function does not alter the position of the links, it only shuffles
    the weights associated to the links. Therefore, the binarised version
    is preserved.

    Parameters
    ----------
    con : ndarray
        Adjacency matrix of the (weighted) network.
    Returns
    -------
    rewmatrix : ndarray
        A connectivity matrix with links between same nodes as 'con' but the
        link weights shuffled.
    """
    # 0) SECURITY CHECKS
    if not type(con) == numpy.ndarray:
        raise TypeError('Please enter the connectivity matrix as a numpy array.')

    # 1) EXTRACT THE CONSTRAINTS FROM THE con MATRIX
    N = len(con)
    nzidx = con.nonzero()
    weights = con[nzidx]

    # 2) GENERATE THE NEW NETWORK WITH THE WEIGHTS SHUFFLED
    numpy.random.shuffle(weights)
    rewmatrix = np.zeros((N,N), dtype=con.dtype)
    rewmatrix[nzidx] = weights

    return rewmatrix





##
