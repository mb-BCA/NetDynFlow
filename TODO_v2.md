# TO-DO list for version v2 of NetDynFlow


### Pending

- Add functions to the *metrics.py* module:
	- Dynamic distance (time-to-peak). For the links, for the nodes and for the whole network. It should be best to write a single function that can return the adequate results either if the NxNxnt tensor with the evolution of the links is given as input, or the Nxnt matrix for the nodes, or the array for the network flow. It requires knowlegde of the temporal span of the simmulation (t0, ffinal) and the time-step dt. That would not be the case if we end transforming everything into objects.
	- Dynamic distance (time-to-relaxation). This one will be the time it takes to reach 95% or 99% of the area under the curve. Try to write a single function that can handle the array for the link, node and network evolutions at once. It requires knowlegde of the temporal span of the simmulation (t0, ffinal) and the time-step dt.
	- Total flow over time. That is, the integral of the area under the curve. ACHTUNG! we need to take the simulation time-step into account. It is an integral, not just the sum over all the values in the time-series. Include optional time span, such that, for example, we can calculate the integral only until the peak (rise phase), or from the peak until the end (decay phase).
	- The peak flow. 
	- We need a function to verify the response curve has reached "zero". Not sure of the criteria that should be applied to this, specially considering the small numbers that flows tend to have. At this moment, it is the user's responsability to guarantee that all the curves have decayed reasonably well. 
	- Revisit the functions to get the flows of the nodes. We should have an option to exclude the self-interactions from those. The interations of a node with the rest, and with itself (inputs given by itself) should be separated. re-think how to do this, but probably we will need an optional parameter `selfinteractions=False/True`.

- Include a *netmodels.py* module for generating networks and surrogates. Include the followong functions:
	- Generate graph of different kinds (deterministic, random graphs, modular, etc.) All these already exist in *pyGAlib*. The question is whether we want to import GAlib and use them from there (this adds one more dependency to the library) or we want to just copy/paste them here.
	- Graph surrogates are also part of GAlib. Same question.
	- A function to generate random weighted networks of different distributions.
	- Random weighted networks from the weight distribution of a given network (conserve exactly the same weights.
	- A function to shuffle only the weights of the links in a network, conserving the location of the links (same binary net, randomised weights).

- Add functions `NNt2tNN()` and `tNN2NNt()` for transposing the flow tensors in the *tools.py* module.

- Think very carefully the **naming of all the metrics**. Both for the existing metrics and the new ones. Stablish a coherent naming system that is general enough, precise and will survive over time to avoid renaming things in the future again. See the *NamingConventions.md* file for proposals. 



### Finished