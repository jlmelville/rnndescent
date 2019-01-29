# rnndescent

An R package implementing the Nearest Neighbor Descent method 
([Dong et al., 2011](https://doi.org/10.1145/1963405.1963487)) for finding
approximate nearest neighbors, based on the Python library 
[PyNNDescent](https://github.com/lmcinnes/pynndescent).

## Current Status

NOTHING TO SEE HERE, MOVE ALONG

## Installation

```r
remotes::install_github("rnndescent")
```

## Example

Optimizing an initial set of approximate nearest neighbors:

```r
irism <- as.matrix(iris[, -5])
iris_rand_nn <- random_nbrs(irism, 15)

res <- nn_descent_opt(irism, iris_rand_nn$idx, iris_rand_nn$dist, verbose = TRUE)
```

## Citation

Dong, W., Moses, C., & Li, K. (2011, March). 
Efficient k-nearest neighbor graph construction for generic similarity measures. 
In *Proceedings of the 20th international conference on World wide web* (pp. 577-586). ACM.
[doi.org/10.1145/1963405.1963487](https://doi.org/10.1145/1963405.1963487).

## License

[GPLv3 or later](https://www.gnu.org/licenses/gpl-3.0.txt).

## See Also

* [PyNNDescent](https://github.com/lmcinnes/pynndescent), the Python implementation.
* [nndescent](https://github.com/TatsuyaShirakawa/nndescent), a C++ implementation.
* [NearestNeighborDescent.jl](https://github.com/dillondaudert/NearestNeighborDescent.jl), a Julia implementation.

