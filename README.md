# DiscreteSamplingFramework
Bayesian sampling over distributions of discrete variables.

This software is licensed under Eclipse Public License 2.0. See [LICENSE](LICENSE) for more details.

This software is property of University of Liverpool and any requests for the use of the software for commercial use or other use outside of the Eclipse Public License should be made to University of Liverpool.

O(logN) Parallel Redistribution (submodule in `discretesampling/base/executor/MPI/distributed_fixed_size_redistribution`) is covered by a patent - A. Varsi & S. Maskell, Method of Parallel Implementation in Distributed Memory Architectures, University of Liverpool, Patent Request GB2101274.5, 29 Jan 2021 - (filed [here](https://patents.google.com/patent/AU2022212776A1/)).

Copyright (c) 2023, University of Liverpool.


## Installation 

### Requirements
 - Python 3.x
 - numpy
 - sympy
 - pandas
 - scipy
 - scikit-learn (for examples)

### Cloning and installing from github

Latest source code can be cloned with:
```bash
git clone https://github.com/Juan3214/ACS-SMC.git --recursive
cd ACS-SMC
```
Package requirements can be installed with:
```bash
pip install -r requirements.txt
```

And the development version of the package cna be installed with:
```bash
pip install -e .
```

## Variables and Distributions
### Discrete Variables
Each example of these should, at minimum implement functions:
 - `__init__` - a constructor with some arguments setting up the basic attributes of the variable
 - `getProposalType` - a class method returning the relevant proposal distribution type for this variable type
 - `getInitialProposalType` - a class method returning the relevant initial proposal distribution type for this variable type
- `getTargetType` - a class method returning the relevant target distribution type for this variable type

We may at some point in the future need to also include:
 - `getLKernelType` - a class method returning the relevant LKernel distribution type for this variable type
 - `getOptimalLKernelType` - a class method returning the relevant Optimal LKernel distribution type for this variable type (a special case of the above probably)
 
 as well as implementing these distribution classes for each variable type.


### Discrete Variable Proposal Distributions
Proposal distributions, q(x'|x), for each variable type should be described.
Each example of these should, at minimum implement functions:
 - `__init__` - constructor with a single argument (a DiscreteVariable, x) which will act as the "starting point" for this proposal
 - `eval` - function taking a single argument (a DiscreteVariable, x') that returns the log-probability of sampling that DiscreteVariable from this proposal (P(x|x'))
 - `sample` - function with no arguments which returns a sample of a DiscreteVariable, x', from this proposal, q (x' ~ q(x))

For more efficient evaluation, optionally implement class methods:
 - `norm` - function that takes a single additional argument (a DiscreteVariable x) and returns some position-like value for that proposal/variable type. This value should impose some sort of ordering on DiscreteVariables.
 - `heuristic` - function that takes two arguments, which will be returns from the above `norm` function and, given
 these values, either return true if the proposal could potentially make transitions between two DiscreteVariables with these norm values or otherwise return false. Note that `heuristic(norm(x), norm(y)) == true` need not guarantee that the transition probability between x and y is non-zero. The inverse is however true: `heuristic(norm(x), norm(y)) == false` guarantees that the transition probability between x and y is zero.

 These two functions will allows for efficient culling of proposal evaluations when implementing e.g. optimal L-kernels


### Discrete Variable Initial Proposal Distributions
Similarly, distributions of initial proposals. q0(x) should be described.
Each example of these should, at minimum implement functions:
 - `__init__` - constructor, possibly with some extra parameters
 - `eval` - function taking a single argument (a DiscreteVariable, x) that returns the log-probability of sampling that DiscreteVariable from this proposal (q0(x))
 - `sample` - function with no arguments which returns a sample of a DiscreteVariable, x', from this initial proposal (x ~ q0())

### Discrete Variable Target Distributions
Each example of these should, at minimum implement functions:
 - `__init__` - constructor, possibly with some parameters theta and data D
 - `eval` - function taking a single argument (a DiscreteVariable, x) that returns the log-posterior density logP(D|x,theta) + logP(x|theta)


