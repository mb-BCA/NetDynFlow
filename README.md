# NetDynFlow

A package to study complex networks based on the temporal evolution of their Dynamic Communicability and Flow.

Graph theory constitutes a widely used and established field providing powerful tools for the characterization of complex networks. However, the diversity of complex networks studied nowadays overcomes the capabilities of classical graph metrics (originally developed for binary adjacency matrices) to provide with information to understand networks and their function. Also, in several domains, networks are often inferred from real-data-forming dynamic systems and thus, their analysis requires a different angle. The tools given in this package allow to overcome those limitations for a variety of complex networks, specially those that are weighted and whose structure is associated with dynamical phenomena.

*Dynamic Flow* characterises the transient network response over time, as the network dynamics relax towards their resting-state after a pulse perturbation (either independent or correlated Gaussian noise) has been applied to selected nodes. On the other hand, *Dynamic Communicability* corresponds to the special case where uncorrelated Gaussian noise has initially been applied to all nodes. The behaviour of the interactions during this transition allows to uncover properties of networks and their function. From a computational point of view dynamic communicability and flow are characterised by a series of matrices, encoding the temporal evolution of the pair-wise interactions between nodes.


#### Reference and Citation

* M. Gilson, N. Kouvaris, G. Deco and G. Zamora-Lopez "*[Framework based on communicability and flow to analyze complex networks](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.97.052301)*" Phys. Rev. E **97**, 052301 (2018).
* M. Gilson, N. Kouvaris, et al. "*[Network analysis of whole-brain fMRI
dynamics: A new framework based on dynamic communicability](https://doi.org/10.1016/j.neuroimage.2019.116007)*" NeuroImage **201**, 116007 (2019).



### INSTALLATION

Installation of *NetDynFlow* is simple. An existing python distribution and the [pip](https://github.com/pypa/pip) package manager need to be installed. If Python was installed via the [Canopy](https://www.enthought.com/product/canopy/) or the [Anaconda](https://www.anaconda.com) distributions, then pip is surely installed. To check, open a terminal and type:

	$ pip --help

*NetDynFlow* is still not registered in PyPI (the Python Packages Index) and installation follows directly from GitHub. However, pip will automatically take care of the  dependencies (see the *requirements.txt* file). There are two alternative manners to install: the easy and the lazy. 

**- The easy installation**: Visit the GitHub repository [https://github.com/gorkazl/NetDynFlow/](https://github.com/gorkazl/NetDynFlow/) and click on the "Clone or download" button at the right hand side (the green button). Select "Download ZIP". Unzip the file, open a terminal and move to the folder, e.g.,

	$ cd ~/Downloads/NetDynFlow-master/

Once on the folder that contains the *setup.py* file, type the following

	$ pip install .

Do not forget the "." at the end which means "*look for the setup.py file in the current directory*." This will check for the dependencies and install *NetDynFlow*. To confirm the installation open an interactive session and try to import the library by typing `import netdynflow`.

> **NOTE**: After installation the current folder "*~/Downloads/NetDynFlow-master/*" can be safely deleted, or moved somewhere else if you want to conserve the examples and the tests.

**- The lazy installation**: If [git](https://git-scm.com) is also installed in your computer, then open a terminal and type:

	$ pip install git+https://github.com/gorkazl/NetDynFlow.git@master

This will install the package, that is, the content in the folder *netdynflow/*. Other files (Examples/, README.md, LICENSE.txt, etc.) need to be downloaded manually, if wanted.


> **NOTE:** If you are using Python 2 and Python 3 environments, *NetDynFlow* needs to be installed in each of the environments.



### HOW TO USE *NetDynFlow*

The package is organised into two modules:

- *core.py*: Functions to obtain the temporal evolution of dynamic communicability and flow.
- *metrics.py*: Network descriptors to analyse the temporal evolution of the dynamic communicability and flow.

To see the list of all functions available use the standard help in an interactive session, e.g.,

	>>> import netdynflow as ndf
	>>> ndf.core?
	>>> ndf.metrics?

>**NOTE:** Importing *NetDynflow* brings all functions in the two modules into its local namespace. Thus, functions in each of the two modules are called as `ndf.func()` instead of `ndf.core.func()` or `ndf.metrics.func()`. Details of each function is also found using the usual help, e.g.,

	>>> ndf.DynCom?
	>>> ndf.Diversity?


#### Getting started 
Create a simple weighted network of N = 4 nodes (a numpy array) and compute its dynamic communicability over time:

	>>> net = np.array((	(0, 1.2, 0, 0),
							(0, 0, 1.1, 0),
							(0, 0, 0, 0.7),
							(1.0, 0, 0, 0)), float)
	>>> tau = 0.8
	>>> dyncom = ndf.DynCom(net, tau, tmax=15, timestep=0.01)

The resulting variable `dyncom` is an array of rank-3 with dimensions ((tmax x tstep) x N x N) containing tmax / tstep = 1500 matrices of size 4 x 4, each describing the state of the network at a given time step. 

> **NOTE**: *NetDynFlow* employs the convention in graph theory that rows of the connectivity matrix encode the outputs of the node. That is, `net[i,j] = 1` implies that the node in row `i` projects over the node in column `j`.

Now we calculate the *total communicability* and *diversity* of the network over time as:

	>>> totalcom = ndf.TotalEvolution(dyncom)
	>>> divers = ndf.Diversity(dyncom)

`totalcom` and `divers` are two numpy arrays of length (tmax / tsteps) = 1500.


### LICENSE

Copyright 2019, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris. E-mail: <gorka@Zamora-Lopez.xyz>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


-------------------------------------------------------------------------------
### VERSION HISTORY

##### November 20, 2019
Official **version 1.0.0** has been uploaded. 
The package went through an in-depth internal revision but implies minor changes from the user point of view.

1. Function `CalcTensor()` was created, which makes most of the dirty job while the rest of core functions, e.g., `DynCom()` or `DynFlow()` became wrappers calling `CalcTensor()`. This internal redesign was made to avoid reproducing code in several parts and thus simplify maintenance.
2. Some internal variables were remaned for uniformity with the [pyMOU](https://github.com/mb-BCA/pyMOU) package.
3. An examples folder was added to host tutorials (Jupyter Notebooks). These examples will be further updated but version of the package will remain v1.0.0 until changes are done at the core files.

##### July 10, 2019
First release of *NetDynFlow* (Beta).


