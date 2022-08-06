# -*- coding: utf-8 -*-
# Copyright (c) 2022, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris
# <galib@Zamora-Lopez.xyz>
#
# Released under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Miscelaneous tools and helpers
==============================
Several extra functions to help in the analysis of networks using the dynamic
communicability or flow
Randomization of (weighted) networks
-------------------------------------
RewireLinkWeights
    Randomly re-allocates the link weights of an input network.
RandomiseWeightedNetwork:
    Randomises a (weighted) connectivity matrix.
!!!!!!!!!!!!
GORKA: Shall we may name this module as 'benchmarks' or 'nullmodels'?
!!!!!!!!!!!!
GORKA: I am thinking to import pyGAlib for some utilities and network generation
functions. On the one hand, I don't want to enforce NetDynFlow users to use
pyGAlib, just because it is my library. But on the other hand, I don't think
that copy/pasting functions from pyGAlib into this module is robust. Replicating
functions may lead to bugs. Better to have only one function to do a job and,
if that one is broken, it only needs to be fixed once.
What shall we do?
!!!!!!!!!!!!
"""
# Standard library imports

# Third party packages
import numpy as np
import numpy.random
# WARNING! Numba is not listed in 'requirements.txt'
from numba import jit


## MISCELLANEOUS FUNCTIONS #####################################################
def Reciprocity(adjmatrix):
    """Computes the fraction of reciprocal links to total number of links.
    Both weighted and unweighted input matrices are permitted. Weights
    are ignored for the calculation.
    Parameters
    ----------
    adjmatrix : ndarray of rank-2
        The adjacency matrix of the network.
    Returns
    -------
    reciprocity : float
        A scalar value between 0 (for acyclic directed networks) and 1 (for
        fully reciprocal).
    """
    # 0) PREPARE FOR COMPUTATIONS
    adjmatrix = adjmatrix.astype('bool')

    # 1) COMPUTE THE RECIPROCITY
    L = adjmatrix.sum()
    if L == 0:
        reciprocity = 0
    else:
        # Find the assymmetric links
        # Rest = np.abs(adjmatrix - adjmatrix.T)
        Rest = np.abs(adjmatrix ^ adjmatrix.T)
        Lsingle = 0.5*Rest.sum()
        reciprocity = np.float(L-Lsingle) / L

    return reciprocity




##
