## Naming conventions for the metrics and their corresponding functions

Naming of objects and metrics needs to be corrected in the DynCom / DynFlow formalism. We should give a detailed though of what each object and metric truly measure from the point of view of the dynamic propagation system that represents. But, we should neither forget the practicality, that many graph/network concepts are well stablished and are familiar to the community. So, as an extension of graph/network analysis tools, we should take this on account and provide names that are familiar to the end users.

On the other hand, there are some potential sources for confusion in this formalism when it comes to naming things. One such reason is the fact that we have both flow, intrinsic flow, extrinsic flow and dynamic communicability. With the two latter being the same, indeed. Another source is the fact that some of the metrics are characterised at three different levels: network, node and link. So, this needs to be explicit in the naming. For example, we could say (network communicability, node communicability and link communicability).

GORKA: To be honest, I don't even like the word communicability. What the flows quantify is the level of response of a node j to a perturbation on node i, or seen the other way around, the influence of node i over j. So, would it make more sense to call it the activity of j? The causal activity of j?


- model-based network analysis --> flow-based, diffusion-based, response-based, impulse-based, perturbation-based, ...
- communicability --> activity, influence, flow, **response**
- dynamic communicability --> temporal communicability.
- total communicability --> network communicability, network response, network flow, … ("total" is not a good term because it could mean the whole-network, or the total communicability over time.)
- in-/out-communicability --> node communicability.
- in-communicability --> node sensitivity/reactivity?
- out-communicability --> node influence? Centrality?
- If network/node/link communicability are the temporal evolution of the network/node/link, then how to call the sum over time (integral) of those quantities?





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


