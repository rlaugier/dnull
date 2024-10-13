# Introduction

dNull is a flexible package for simulation of interferometric observations and implementing automatic differentiation (autodiff). It is built as a **backend of NIFITS** powered by [Zodiax](https://github.com/LouisDesdoigts/zodiax) which is a superset of Equinox, powered by jax.

Automatic differentiation allows very fast and efficient computation of the partial derivatives of functions and methods with respect to all arguments. The computation of Jacobians and Hessians open new possibilities:
* Faster convergence on large number of adjusted parameters
* Evaluation of posterior covariance
* Numerical inversion of the relationshib between instrument parameters and scientific performance.

Contrary to the [nifits library](https://github.com/rlaugier/nifits) dNull does not pretend at full compatibility with the whole of NIFITS standard. It targets a specific set of experimental proof of concept concerning the establishment of instrument requirements and development of a 3rd generation LIFE pipeline.


## Acknowledgements
NIFITS is a development carried out in the context of the [SCIFY project](http://denis-defrere.com/scify.php). [SCIFY](http://denis-defrere.com/scify.php) has received funding from the **European Research Council (ERC)** under the European Union's Horizon 2020 research and innovation program (*grant agreement No 866070*).  Part of this work has been carried out within the framework of the NCCR PlanetS supported by the Swiss National Science Foundation under grants 51NF40_18290 and 51NF40_205606. We wish to thank Louis Desdoigts for his support in implementing the Zodiax library.

