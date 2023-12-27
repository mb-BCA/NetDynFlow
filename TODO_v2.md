# Roadmap for version v2 of NetDynFlow

Given that in the new Perspective Article in which we call for a model-based network analyses using different canonical models, we need a complete new roadmap for v2 of the library. We need to provide two new things:

1. The functions to calculate the response matrices (tensors) for the different canonical models, regardless of whether they are time-discrete or time-continuous.
2. A new module with the code to simulate and get the temporal solutions **x**(t) for the nodes for each canonical model.

I guess that version v2 of *NetDynFlow* is going to be transparent from the point of view of the code but will have several redundant functions with similar names, specially those to generate the response matrices under the different models. I would like to make v3 an object-oriented library which would reduce the number of "functions" and names the users need to remember. That is, only one function (with different options, of course) to generate the tensor of the response matrices, and another function to compute the solutions **x**(t). Those two functions will then be called specifying the canonical model and the particular parameters if the model needs that. E.g., the decay time-constant for the leaky-cascade (MOU).

In any case, v2 has to be a clean and coherent library such that the transition to an object-oriented version should be as smooth as possible.


### Pending

- Add functions to *core.py* module to compute R(t) for the different models:
	- (DONE) Unify the functions for the MOU case into one function.
	- (DONE) R(t) for the constinuous cascade.
	- (DONE) R(t) for the discrete cascade.
	- R(t) for the random walks.
	- R(t) for the continuous diffusion.
- Add functions to the *metrics.py* module:
	- (DONE) To return the peak flows. I know, it is really trivial to compute but… we need to give these things in functions for beginer users. 
	- We need a function to verify the response curve has reached "zero". Not sure of the criteria that should be applied to this, specially considering the small numbers that flows tend to have. At this moment, it is the user's responsability to guarantee that all the curves have decayed reasonably well. If the responses haven't properly decay, the function should return a warning, recommending to run longer simulations.
	- We need a function to extract and study the evolution of the self-interactions. That is, the temporal response of a node to a perturbation on itself at time t = 0. This is in a way what Ernesto called "returnability" but we have that over time. Remind that in graphs the clustering coefficient is indeed calculated in this manner, for loops of lenght = 3, but longer loops could be included.

- What to do about the '**sigma**' parameter that we only have for the MOU? It expects a matrix, not a vector of input amplitudes to the nodes. 
	- A vector is what the readers of the Perspective paper will expect that is needed, but …
	- The good thing about '**sigma**' is that it allows for correlated inputs between nodes, it is more general than the individual 'kicks' to the nodes.
	- But … if '**sigma**' stays for the MOU canonical version, then, the R(t) generating functions should also allow the same … but … does a Gaussian noise make sense for, e.g., the discrete cascade and the random walker canonical models? I don't think so. Does it make sense for the continuous cascade and the continuous diffusion models? Probably yes. DOUBLE CHECK WITH MATT.
	- Possible solution: make `sigma` available for the R(t) generators of the continuous canonical models. BUT, make it an optional parameter with default being `sigma=None`. This default will use the identity matrix = input of unit 1.0 to all nodes. Then, allow `sigma` to be either a vector of size N for the amplitudes of the initial inputs, OR a NxN matrix with the possible correlated noise inputs as well.
	- If `sigma` will be the matrix of Gaussian noise input amplitudes, Matt said that the norm of the matrix should fulfil some condition. DOUBLE CHECK WITH MATT and add the subsequent security check to the function(s).

- Include a *netmodels.py* module for generating networks and surrogates. Include the followong functions:
	- A function to generate random weighted networks of different distributions.
	- In spatially embedded networks, a function to assign the stronger links to the closest nodes.
	- Weighted ring lattice, with stronger weights between neighbouring nodes (model by Muldoon et al.)

- Think very carefully the **naming of all the metrics**. Both for the existing metrics and the new ones. Stablish a coherent naming system that is general enough, precise and will survive over time to avoid renaming things in the future again. See the *NamingConventions.md* file for proposals. 
- Think very carefully the **naming of the canonical models**. There are historical implications here but … One should be pragmatical and besides, those names should really be informative for the user. I would prefer that than using names only because in one field or in another, the models are called in some way. See the *NamingConventions.md* file for proposals.
- Function `Time2Peak()` should return `np.inf` for those pair-wise elements when there is no input in a node. Now, it returns zeros in those cases.
- Same for function `Time2Decay()`. Not it returns the duration of the simulation in those cases. 
- Add security checks at the beginning of all functions.



### Finished

- Include a new module named `simulations.py` containing the code to simulate the network under the different canonical models and return the temporal solutions **x**(t) for the nodes.
- Add functions to the *metrics.py* module:
	- Dynamic distance (time-to-peak). For the links, for the nodes and for the whole network. It should be best to write a single function that can return the adequate results either if the NxNxnt tensor with the evolution of the links is given as input, or the Nxnt matrix for the nodes, or the array for the network flow. It requires knowlegde of the temporal span of the simmulation (t0, ffinal) and the time-step dt. That would not be the case if we end transforming everything into objects.
	- Dynamic distance (time-to-relaxation). This one will be the time it takes to reach 95% or 99% of the area under the curve. Try to write a single function that can handle the array for the link, node and network evolutions at once. It requires knowlegde of the temporal span of the simmulation (t0, ffinal) and the time-step dt.
	- Total flow over time. That is, the integral of the area under the curve. ACHTUNG! we need to take the simulation time-step into account. It is an integral, not just the sum over all the values in the time-series. Include optional time span, such that, for example, we can calculate the integral only until the peak (rise phase), or from the peak until the end (decay phase).
	- Function `NodeEvolution()`has been renamed as `NodeFlows()`. The optional parameter `directed` has been removed. Now the function always returns both the input and the output flows regardless of whether the underlying connectivity is directed or not. If it is symmetric, then the in- and out-flows will simply be the same. Although redundant, it is less confusing if the function always returns the structure of the function's output is always the same.
	- A new parameter `selfloops` has been added to function `NodeFlows()`. If `selfloops = True` the output will include the consequence of the initial perturbation applied to a node on itself. If `selfloops = False` (default) then the function only returns the in-flows into a node due to the perturbations on other nodes, and account only for the flow that a nodel provokes on other nodes, not itself.
	- Revisit the function to get the network and node flows over timein *metrics.py*: `TotalEvolution()` and  `NodeEvolution()`. We should have an optional parameter named `selfloops=False/True` to exclude or include the self-interactions from the cross-nodal interactions. At this moment, the calculations include the self-interactions but I am not sure we should do that. The evolution and strength of the self-interactions (response of a node to a perturbation on itself at t = 0) carry their own meaning and should be characterised separately. Re-think how to do this in the algorithm.


- Include a *netmodels.py* module for generating networks and surrogates. Include the followong functions:
	- I have imported *pyGAlib* in module *netmodels.py* such that we can use all the (di) graph generation and randomization functions. In the future we could think whether we want GAlib to be a dependence of netdynflow, or we prefere to duplicate those functions here.
	- Random weighted surrogate networks from an input connectivity matrix: `RandomiseWeightedNetwork()`.
	- A function to shuffle only the weights of the links in a network, conserving the location of the links (same binary net, randomised weights): `ShuffleLinkWeights()`.


- Add functions `NNt2tNN()` and `tNN2NNt()` for transposing the flow tensors in the *tools.py* module.




