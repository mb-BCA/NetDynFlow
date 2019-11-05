## Internal variable naming conventions


<br/>
Here the list of proposed variable and function changes. Add alternative names for each case if dissagree, or add other proposals as well.

- con_matrix --> C, c_mat
- tau_const --> tau, tau_vec
- sigma_mat --> sigma
- n_nodes --> N
- GenerateTensors() --> ??



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


