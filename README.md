> DEPRECATED PACKAGE. NetDynFlow is being maintained but no longer developed. The functionalities of NetDynFlow are superseded by [SiReNetA](https://github.com/mb-BCA/SiReNetA), which generalises the same type of network analyses but accounts for different underlying propagation models.

# NetDynFlow

*A package to study complex networks based on the temporal evolution of their Dynamic Communicability and Flow.*

Graph theory constitutes a widely used and established field providing powerful tools for the characterization of complex networks. However, the diversity of complex networks studied nowadays overcomes the capabilities of classical graph metrics (originally developed for binary adjacency matrices) to provide with information to understand networks and their function. Also, in several domains, networks are often inferred from real-data-forming dynamic systems and thus, their analysis requires a different angle. The tools given in this package allow to overcome those limitations for a variety of complex networks, specially those that are weighted and whose structure is associated with dynamical phenomena.

*Dynamic Flow* characterises the transient network response over time, as the network dynamics relax towards their resting-state after a pulse perturbation (either independent or correlated Gaussian noise) has been applied to selected nodes. On the other hand, *Dynamic Communicability* corresponds to the special case where uncorrelated Gaussian noise has initially been applied to all nodes. The behaviour of the interactions during this transition allows to uncover properties of networks and their function. From a computational point of view dynamic communicability and flow are characterised by a series of matrices, encoding the temporal evolution of the pair-wise interactions between nodes.


#### Reference and Citation

* M. Gilson, N. Kouvaris, G. Deco and G. Zamora-Lopez "*[Framework based on communicability and flow to analyze complex networks](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.97.052301)*" Phys. Rev. E **97**, 052301 (2018).
* M. Gilson, N. Kouvaris, et al. "*[Network analysis of whole-brain fMRI
dynamics: A new framework based on dynamic communicability](https://doi.org/10.1016/j.neuroimage.2019.116007)*" NeuroImage **201**, 116007 (2019).



### INSTALLATION

#### Direct installation from GitHub 

If you have [git](https://git-scm.com) installed, you can install NetDynFlow directly from its GitHub repository. Open a terminal window and type:

	python3 -m pip install git+https://github.com/mb-BCA/netdynflow@master

This procedure will only download and install the package (files in the "*netdynflow/*" folder) into your current environment. 

#### Installing NetDynFlow in editable mode

If you want to install NetDynFlow such that you can make changes to it "*on the fly*" then, visit its GitHub repository [https://github.com/mb-BCA/NetDynFlow/tree/master](https://github.com/mb-BCA/netdynflow/tree/master), and click on the green "*<> Code*" button on the top right and select "Download ZIP" from the pop-up menu. Once downloaded, move the *zip* file to a target folder (e.g., "*~/Documents/myLibraries/*") and unzip the file. Open a terminal and `cd` to the resulting folder, e.g.,

	cd ~/Documents/myLibraries/NetDynFlow-master/

Once on the path (make sure it contains the *pyproject.toml* file), type:

	python3 -m pip install -e .

Do not forget the "." at the end which means "*look for the pyproject.toml file in the current directory*." This will install NetDynFlow such that every time changes are made to the package (located in the path chosen), these will be inmediately available. You may need to restart the IPython or Jupyter notebook session, though.



### HOW TO USE NetDynFlow

The package is organised into four modules:

- *core.py*: Functions to obtain the temporal evolution of dynamic communicability and flow.
- *metrics.py*: Network descriptors to analyse the temporal evolution of the dynamic communicability and flow.
- *netmodels.py*: Functions to generate benchmark and surrogate networks.
- *tools.py*: Data transformations and other helpers.


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

Copyright 2019, Gorka Zamora-López, Matthieu Gilson and Nikos E. Kouvaris.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


-------------------------------------------------------------------------------
### VERSION HISTORY

##### November 27, 2025 (Version 1.1)

* The library has been reshaped to be compliant with the modern [PyPA specifications](https://packaging.python.org/en/latest/specifications/).
* [Hatch](https://hatch.pypa.io/latest/) was chosen as the tool to build and publish the package. See the *pyproject.toml* file. 
* Minor bug and documentation fixes.


##### March 14, 2024
Small bugs fixed:

- Remaining *Numba* dependency removed.
- Fixed the new  aliases for `int` and `float` in *Numpy*. All arrays are now declared as `np.int64` or `np.float64`, and individual numbers as standard Python `int` or `float`. 

##### December 14, 2023
*core.py* module has been simplified to three practical functions `DynFlow()`, `IntrinsicFlow()` and `ExtrinsicFlow()`. 

New measures added to *metrics.py* module: `Time2Peak()`, `Time2Decay` and `AreaUnderCurve()`. These functions accept either the temporal response matrices of shape (nt,N,N) (e.g., the output of DynFlow), the temporal responses of nodes of shape (nt,N) or the global network response of shape (nt,1). The functions will return a matrix, a vector or an scalar accordingly for each case.

##### November 20, 2019
Official **version 1.0.0** has been uploaded. 
The package went through an in-depth internal revision but implies minor changes from the user point of view.

1. Function `CalcTensor()` was created, which makes most of the dirty job while the rest of core functions, e.g., `DynCom()` or `DynFlow()` became wrappers calling `CalcTensor()`. This internal redesign was made to avoid reproducing code in several parts and thus simplify maintenance.
2. Some internal variables were remaned for uniformity with the [pyMOU](https://github.com/mb-BCA/pyMOU) package.
3. An examples folder was added to host tutorials (Jupyter Notebooks). These examples will be further updated but version of the package will remain v1.0.0 until changes are done at the core files.

##### July 10, 2019
First release of *NetDynFlow* (Beta).


