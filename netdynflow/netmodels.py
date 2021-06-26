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

Functions to gnerate different types of networks (deterministic and random),
and to generate surrogates out of given binary or weighted networks.

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

Surrogate network generation
----------------------------
ShuffleLinkWeights
    Randomly re-allocates the weights associated to the links.

"""
from __future__ import division, print_function

import numpy as np
import numpy.linalg
import scipy.linalg
import galib
from galib import (Reciprocity )
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
