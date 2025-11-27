# -*- coding: utf-8 -*-
# Copyright (c) 2019, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris
#
# Released under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Network Analysis Based on Perturbation-Induced Flows
====================================================

A package to study complex networks based on the spatio-temporal propagation of
flows due to external perturbations.
Compatible with Python 3.X.

- 'Dynamic Flow' characterises the transient network response over time, as the
network dynamics relax towards their resting-state after a pulse perturbation
(either independent or correlated Gaussian noise) has been applied to selected
nodes.

NetDynFlow treats networks as connectivity matrices, represented as 2D NumPy
arrays. Given (i) a connectivity matrix and (ii) a an input covariance matrix,
conditional pair-wise network flows return a set of matrices (a 3D NumPy array),
each describing the state of the state of the pairwise node interations at
at consecutive time points.

Reference and Citation
**********************

1. M. Gilson, N. Kouvaris, G. Deco & G.Zamora-Lopez "Framework based on communi-
cability and flow to analyze complex networks" Phys. Rev. E 97, 052301 (2018).

2. M. Gilson, N. Kouvaris, et al. "Network analysis of whole-brain fMRI
dynamics: A new framework based on dynamic communicability" NeuroImage 201,
116007 (2019).

Available functions
*******************

The package is organised into two modules:

core
    Functions to calculate spatio-temporal evolution of conditiona network flows.
metrics
    Descriptors to analyse the spatio-temporal evolution of perturbation-induced
    flows in a network.
netmodels
    Functions to generate benchmark and surrogate networks.
tools
    Data transformations and other helpers.

To see the list of all functions available use the standard help in an
interactive session, for both modules ::

>>> import netdynflow as ndf
>>> ndf.core?
>>> ndf.metrics?

    **NOTE:**
    Importing NetDynflow brings all functions in the two modules into its
    local namespace. Thus, functions in each of the two modules are called as
    `ndf.func()` instead of `ndf.core.func()` or `ndf.metrics.func()`. Details
    of each function is also found using the usual help, e.g.,

	>>> ndf.DynFlow?
	>>> ndf.Diversity?

In an IPython interactive session, or in a Jupyter Notebook, typing ``netdynflow``
and then pressing <tab> will show all the modules and functions available in
the package.

Using NetDynFlow
^^^^^^^^^^^^^^^^
Since NetDynFlow depends on NumPy, it is recommended to import NumPy first,
although this is not necessary for loading the package: ::

>>> import numpy as np
>>> import netdynflow as ndf

    **Note**:
    Importing netdynflow imports also all functions in module *core.py*
    into its namespace. Module *metrics.py* is imported separately. Therefore,
    if the import is relative those functions can be called as, e.g.,  ::

    >>> import netdynflow as ndf
    >>> ...
    >>> sigma = np.identity(N)
    >>> dynflow = ndf.DynFlow(net, tau, sigma)

We did not have to call ``netdynflow.core.DynFlow()``. In the case of an absolute
import (using an asterisk ``*``) all functions in *core.py* are imported to the
base namespace:  ::

    >>> from netdynflow import *
    >>> ...
    >>> sigma = np.identity(N)
    >>> dynflow = DynFlow(net, tau, sigma)

Getting started
***************
Create a simple weighted network of N = 4 nodes (a numpy array) and compute its
dynamic flow over time: ::

>>> net = np.array(((0, 1.2, 0, 0),
                    (0, 0, 1.1, 0),
                    (0, 0, 0, 0.7),
                    (1.0, 0, 0, 0)), float)

>>> tau = 0.8
>>> sigma = np.identity(N)  # Matrix of initial perturbations
>>> dynflow = ndf.DynFlow(net, tau, sigma, tmax=15, timestep=0.01)

The resulting variable ``dynflow`` is an array of rank-3 with dimensions
(nt,N,N) where nt = (tmax / tstep) containing nt = 1500 matrices of size 4 x 4,
each describing the pair-wise flows in the network over time.

    **NOTE**:
    NetDynFlow employs the convention in graph theory that rows of the
    connectivity matrix encode the outputs of the node. That is, `net[i,j] = 1`
    implies that the node in row ``i`` projects over the node in column ``j``.

Now we calculate the *total flow* and the ``diversity`` of the network over
time as:  ::

>>> totalflow = ndf.TotalEvolution(dynflow)
>>> divers = ndf.Diversity(dynflow)

``totalflow`` and ``divers`` are two arrays of length (tmax / tsteps) = 1500.


License
-------
Copyright (c) 2019, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris

Released under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

or see the LICENSE.txt file.

"""

# Branch to test different options to organise the core.py module
from . import core
from .core import *
from . import metrics
from .metrics import *
from . import tools
from . import netmodels


# Some metada of the package
__version__ = "1.1"



#
