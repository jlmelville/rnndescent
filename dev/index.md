# rnndescent

An R package for approximate nearest neighbor search on dense and sparse
data. Use it to build a neighbor graph, refine an existing graph, or
build an index to query with new data.

The package adapts the core search methods from Leland McInnes’s
[PyNNDescent](https://github.com/lmcinnes/pynndescent) to R. It uses
[nearest neighbor descent](https://doi.org/10.1145/1963405.1963487) and
[random partition trees](https://doi.org/10.1145/1374376.1374452). It
also draws on [FANNG](https://doi.org/10.1109/CVPR.2016.616) and
[NGT](https://github.com/yahoojapan/NGT).

## Current Status

**26 May 2026.** Version 0.2.0 adds the Haversine metric and improves
input validation. [CRAN
release](https://cran.r-project.org/package=rnndescent)

## Installation

### CRAN

``` r

install.packages("rnndescent")
```

### Development version

``` r

# install.packages("pak")
pak::pak("jlmelville/rnndescent")
```

Building from source requires a C++17 compiler. See [Rtools for
Windows](https://cran.r-project.org/bin/windows/Rtools/) and the [R
build tools for macOS](https://mac.r-project.org/tools/).

## Example

``` r

library(rnndescent)

set.seed(42)
x <- iris[1:4]

# Find the 15 nearest neighbors of each observation
iris_knn <- rnnd_knn(x, k = 15)

# Build an index on reference rows and query it with the remaining rows
iris_index <- rnnd_build(x[1:100, ], k = 15)
iris_query_nn <- rnnd_query(iris_index, x[101:150, ], k = 15)
```

## Documentation

- [Get
  started](https://jlmelville.github.io/rnndescent/articles/rnndescent.html)
- [Articles](https://jlmelville.github.io/rnndescent/articles/)
- [Changelog](https://jlmelville.github.io/rnndescent/news/index.html)

## Missing Metrics

The `metric` argument takes the name of a built-in distance. This covers
many PyNNDescent metrics, with gaps such as Mahalanobis and weighted
Minkowski. See the [metrics
article](https://jlmelville.github.io/rnndescent/articles/metrics.html)
for the full list.

## Citation

Dong, W., Moses, C., & Li, K. (2011, March). Efficient k-nearest
neighbor graph construction for generic similarity measures. In
*Proceedings of the 20th international conference on World wide web*
(pp. 577-586). ACM.
[doi.org/10.1145/1963405.1963487](https://doi.org/10.1145/1963405.1963487).

## License

The R package is licensed under [GPLv3 or
later](https://www.gnu.org/licenses/gpl-3.0.txt).

- `inst/include/dqsample.h` is adapted from
  [dqrng](https://github.com/daqana/dqrng) and licensed under [AGPLv3 or
  later](https://www.gnu.org/licenses/agpl-3.0.en.html).
- `inst/include/pforr.h` is adapted from
  [RcppParallel](https://github.com/RcppCore/RcppParallel) and licensed
  under [GPLv2 or
  later](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html).
- The C++ library under `inst/include/tdoann` uses the [BSD 2-Clause
  license](https://opensource.org/license/bsd-2-clause/).

## See Also

- [nndescent](https://github.com/TatsuyaShirakawa/nndescent), a C++
  implementation.
- [NearestNeighborDescent.jl](https://github.com/dillondaudert/NearestNeighborDescent.jl),
  a Julia implementation.
- [NNDescent.cpp](https://github.com/AnabelSMRuggiero/NNDescent.cpp),
  another C++ implementation.
- [nndescent](https://github.com/brj0/nndescent), a C++ implementation
  with Python bindings.
