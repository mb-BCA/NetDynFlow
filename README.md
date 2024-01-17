**Staled branch.** Only purpose was to try to use Numba to accelerate some calculations but it was pretty useless. 

Only the functions to simulate the canonical models (*simulate.py* module) were possible to "numbify" but the gain is quite small compared to the original code, provided all arrays are enforced to have the same `np.dtype`. The numbified functions are all in the module *simulate_numba.py*.

Regarding the calculation of the pair-wise response matrices over time, which are the heaviest calculations of the library, it turned either impossible or useless to mumbify those (funcitons in *core.py*). The reason is that the main performance bottleneck in *core.py* are the calculations of `scipy.linalg.expm()` in several functions and scipy functions are not supported by Numba !! Well, there are a few functions that are supported through the [numba-scipy](https://github.com/numba/numba-scipy) project (see [docs here](https://numba-scipy.readthedocs.io/en/latest/index.html#)) but not the ones we needed here. Functions `RespMatrices_DiscreteCascade()`and `RespMatrices_RandomWalk()` could be numbified, in principle. But the complication is not worth since their calculations are pretty straight forward and usually converge after a few iterations, as as the are time-discrete processes.



On the positive side, these efforts served to learn how handle better the user inputs (parameters to the functions) and to enforce that all array-like inputs are given the same `np.dtype`.