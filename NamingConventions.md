## Naming conventions for the metrics and their corresponding functions

Naming of objects and metrics needs to be corrected in the DynCom / DynFlow formalism. We should give a detailed though of what each object and metric truly measure from the point of view of the dynamic propagation system that represents. But, we should neither forget the practicality, that many graph/network concepts are well stablished and are familiar to the community. So, as an extension of graph/network analysis tools, we should take this on account and provide names that are familiar to the end users.

#### Communicability, response, flows, … ??

A very important question we must address is to give proper names for what we are measuring. Following the inertia in the literature we started calling this "*dynamic communicability*" but this was the moment we (and nobody) trully understood what is the physical interpretation of communicability. Which now we know, is not communication. Internally, we are calling some measures as the "dynamic flow" or "intrinsic flow." But, are we 100% sure those are trully flows, as properly defined in physics? Or are they "accumulated responses" or something like that? We must specify and be precise.

We need to clarify this for good because otherwise we will create more confusion when our goal is to create a framework that clarifies things. One issue is that extrinsic flow and dynamic communicability are the same. We can't have two names for the same. To be honest, I never liked the word communicability. It means nothing. It is pretty much a generic name to refer to something unspecific that is not well explained. So, **we have to remove the word "communicability" from our formalism and code**.

When thinking how to name the measures, and how to explain the formalism, it should be important to realise the distinction between concepts and actual physical quantities. I mean:

- Diffusion, response, perturbation, etc. are concepts, they are not physical measures. That is, they do not have a physical unit per se. So, we can use those terms with care for descriptive purposes but we should not name actual metrics using those concepts.
- For example, *flow* **is** a physical quantity. So, we should use "flow" to name the measures and the tensors. BUT we must make sure that those actually reflect flows, as properly defined by physics. 

Another possible source of confusion is that some of the metrics are characterised at three different levels: network, node and link. So, this needs to be explicit in the naming. For example, we could say (network flow, node flow and link flow).


##### List of names to take care of, and decide upon


- model-based network analysis --> flow-based, diffusion-based, response-based, impulse-based, perturbation-based, ...
- communicability --> activity, influence, flow, **response**
- dynamic communicability --> temporal communicability.
- total communicability --> network communicability, network response, network flow, … ("total" is not a good term because it could mean the whole-network, or the total communicability over time.)
- in-/out-communicability --> node communicability.
- in-communicability --> node sensitivity/reactivity?
- out-communicability --> node influence? Centrality?
- If network/node/link communicability are the temporal evolution of the network/node/link, then how to call the sum over time (integral) of those quantities?


##### Internal naming for the canonical models, to decide upon

These names are relevant because they are the "short names" we should use to "tag" the functions that are specific for each model. And, for simplicity and coherence, these should be the same names we will use in v3 for the class attribute that specifies the canonical model.

- Discrete cascade
- Random walks
- (Random walks with teleportation)
- Continuous cascade
- Leaky cascade
- Continuous diffusion (simple diffusion)


####  A new name for the library, to replace 'NetDynFlow'

In the new paper, where we generalise the ideas of the "dynamic communicability" for plugin different canonical models, we converged into calling Rij(t) as the responses from one node to another. Therefore, we are no longer talking of flows or communicability and we should have a different name for the library. Also, we do not want to use the word "pertubartion" because some people think it refers to a lession of the network nodes or links. So, it seems that "Response Network Analysis" could be a reasonable naming.
 
- Response Network Analysis. 
	- ReNetA `rna.function()`
	- Renata
	- RespNet `rn.function()`
- Network Response Analysis. NetReA, `nra.function()`
- Perturbation-Response Network analysis
- Stimulus-Response Network Analysis
	- SRNetA `sra.function()` 
	- SiReNA `sirena.function()`  `sna.function()`
	- SiReNetA `sra.function()` `srn.function()`

**ACHTUNG !!** We must look in PyPI whether libraries with these names already exist :(



## Internal variable naming conventions


<br/>
Here the list of proposed variable and function changes. Add alternative names for each case if dissagree, or add other proposals as well.

- con_matrix --> con
- tau_const --> tau
- sigma_mat --> sigma
- n_nodes --> N
- GenerateTensors() --> CalcTensor()



<br/>

List here the agreed name changes:

- jac 	<-- J (in pyMOU.mou_model), jacobian (in NetDynFlow.core)
- jacdiag <-- jacobian_diag (in NetDynFlow.core)
- tau 	<-- tau_x (in pyMOU.mou_model), tau_const (in NetDynFlow.core)
- con 	<- C (in pyMOU.mou_model), con_matrix (in NetDynFlow.core)
- tensor	<-- dyn_tensor (in NetDynFlow.metrics)
- sigma or incov <-- Sigma (in pyMOU.mou_model)
- cov0, covlag <-- Q0, Qtau (in pyMOU.mou_model)
- lag 	<- tau (in pyMOU.mou_model.Mou.fit_LO)


