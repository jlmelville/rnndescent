# Find nearest neighbors using nearest neighbor descent

Uses the Nearest Neighbor Descent method due to Dong and co-workers
(2011) to optimize an approximate nearest neighbor graph.

## Usage

``` r
nnd_knn(
  data,
  k = NULL,
  metric = "euclidean",
  init = "rand",
  init_args = NULL,
  n_iters = NULL,
  max_candidates = NULL,
  delta = 0.001,
  low_memory = TRUE,
  weight_by_degree = FALSE,
  use_alt_metric = TRUE,
  n_threads = 0,
  verbose = FALSE,
  progress = "bar",
  obs = "R",
  ret_forest = FALSE
)
```

## Arguments

- data:

  Matrix of `n` items to generate neighbors for, with observations in
  the rows and features in the columns. Optionally, input can be passed
  with observations in the columns, by setting `obs = "C"`, which should
  be more efficient. Possible formats are
  [`base::data.frame()`](https://rdrr.io/r/base/data.frame.html),
  [`base::matrix()`](https://rdrr.io/r/base/matrix.html) or
  [`Matrix::sparseMatrix()`](https://rdrr.io/pkg/Matrix/man/sparseMatrix.html).
  Sparse matrices should be in `dgCMatrix` format. Dataframes will be
  converted to `numerical` matrix format internally, so if your data
  columns are `logical` and intended to be used with the specialized
  binary `metric`s, you should convert it to a logical matrix first
  (otherwise you will get the slower dense numerical version).

- k:

  Number of nearest neighbors to return. Optional if `init` is
  specified.

- metric:

  Type of distance calculation to use. One of:

  - `"braycurtis"`

  - `"canberra"`

  - `"chebyshev"`

  - `"correlation"` (1 minus the Pearson correlation)

  - `"cosine"`

  - `"dice"`

  - `"euclidean"`

  - `"hamming"`

  - `"haversine"` (great-circle distance for 2D latitude/longitude in
    *radians*; an error will be raised if values that appear to be
    supplied in degrees are encountered)

  - `"hellinger"`

  - `"jaccard"`

  - `"jensenshannon"`

  - `"kulsinski"`

  - `"sqeuclidean"` (squared Euclidean)

  - `"manhattan"`

  - `"rogerstanimoto"`

  - `"russellrao"`

  - `"sokalmichener"`

  - `"sokalsneath"`

  - `"spearmanr"` (1 minus the Spearman rank correlation)

  - `"symmetrickl"` (symmetric Kullback-Leibler divergence)

  - `"tsss"` (Triangle Area Similarity-Sector Area Similarity or TS-SS
    metric)

  - `"yule"`

  For non-sparse data, the following variants are available with
  preprocessing: this trades memory for a potential speed up during the
  distance calculation. Some minor numerical differences should be
  expected compared to the non-preprocessed versions:

  - `"cosine-preprocess"`: `cosine` with preprocessing.

  - `"correlation-preprocess"`: `correlation` with preprocessing.

  For non-sparse binary data passed as a `logical` matrix, the following
  metrics have specialized variants which should be substantially faster
  than the non-binary variants (in other cases the logical data will be
  treated as a dense numeric vector of 0s and 1s):

  - `"dice"`

  - `"hamming"`

  - `"jaccard"`

  - `"kulsinski"`

  - `"matching"`

  - `"rogerstanimoto"`

  - `"russellrao"`

  - `"sokalmichener"`

  - `"sokalsneath"`

  - `"yule"`

- init:

  Name of the initialization strategy or initial `data` neighbor graph
  to optimize. One of:

  - `"rand"` random initialization (the default).

  - `"tree"` use the random projection tree method of Dasgupta and
    Freund (2008).

  - a pre-calculated neighbor graph. A list containing:

    - `idx` an `n` by `k` matrix containing the nearest neighbor
      indices.

    - `dist` (optional) an `n` by `k` matrix containing the nearest
      neighbor distances. If the input distances are omitted, they will
      be calculated for you.'

  If `k` and `init` are specified as arguments to this function, and the
  number of neighbors provided in `init` is not equal to `k` then:

  - if `k` is smaller, only the `k` closest values in `init` are
    retained.

  - if `k` is larger, then random neighbors will be chosen to fill
    `init` to the size of `k`. Note that there is no checking if any of
    the random neighbors are duplicates of what is already in `init` so
    effectively fewer than `k` neighbors may be chosen for some
    observations under these circumstances.

- init_args:

  a list containing arguments to pass to the random partition forest
  initialization. See
  [`rpf_knn()`](https://jlmelville.github.io/rnndescent/reference/rpf_knn.md)
  for possible arguments. To avoid inconsistences with the tree
  calculation and subsequent nearest neighbor descent optimization, if
  you attempt to provide a `metric` or `use_alt_metric` option in this
  list it will be ignored.

- n_iters:

  Number of iterations of nearest neighbor descent to carry out. By
  default, this will be chosen based on the number of observations in
  `data`.

- max_candidates:

  Maximum number of candidate neighbors to try for each item in each
  iteration. Use relative to `k` to emulate the "rho" sampling parameter
  in the nearest neighbor descent paper. By default, this is set to `k`
  or `60`, whichever is smaller.

- delta:

  The minimum relative change in the neighbor graph allowed before early
  stopping. Should be a value between 0 and 1. The smaller the value,
  the smaller the amount of progress between iterations is allowed.
  Default value of `0.001` means that at least 0.1% of the neighbor
  graph must be updated at each iteration.

- low_memory:

  If `TRUE`, use a lower memory, but more computationally expensive
  approach to index construction. If set to `FALSE`, you should see a
  noticeable speed improvement, especially when using a smaller number
  of threads, so this is worth trying if you have the memory to spare.

- weight_by_degree:

  If `TRUE`, then candidates for the local join are weighted according
  to their in-degree, so that if there are more than `max_candidates` in
  a candidate list, candidates with a smaller degree are favored for
  retention. This prevents items with large numbers of edges crowding
  out other items and for high-dimensional data is likely to provide a
  small improvement in accuracy. Because this incurs a small extra cost
  of counting the degree of each node, and because it tends to delay
  early convergence, by default this is `FALSE`.

- use_alt_metric:

  If `TRUE`, use faster metrics that maintain the ordering of distances
  internally (e.g. squared Euclidean distances if using
  `metric = "euclidean"`), then apply a correction at the end. Probably
  the only reason to set this to `FALSE` is if you suspect that some
  sort of numeric issue is occurring with your data in the alternative
  code path.

- n_threads:

  Number of threads to use.

- verbose:

  If `TRUE`, log information to the console.

- progress:

  Determines the type of progress information logged if
  `verbose = TRUE`. Options are:

  - `"bar"`: a simple text progress bar.

  - `"dist"`: the sum of the distances in the approximate knn graph at
    the end of each iteration.

- obs:

  set to `"C"` to indicate that the input `data` orientation stores each
  observation as a column. The default `"R"` means that observations are
  stored in each row. Storing the data by row is usually more
  convenient, but internally your data will be converted to column
  storage. Passing it already column-oriented will save some memory and
  (a small amount of) CPU usage.

- ret_forest:

  If `TRUE` and `init = "tree"` then the RP forest used to initialize
  the nearest neighbors will be returned with the nearest neighbor data.
  See the `Value` section for details. The returned forest can be used
  as part of initializing the search for new data: see
  [`rpf_knn_query()`](https://jlmelville.github.io/rnndescent/reference/rpf_knn_query.md)
  and
  [`rpf_filter()`](https://jlmelville.github.io/rnndescent/reference/rpf_filter.md)
  for more details.

## Value

the approximate nearest neighbor graph as a list containing:

- `idx` an n by k matrix containing the nearest neighbor indices.

- `dist` an n by k matrix containing the nearest neighbor distances.

- `forest` (if `init = "tree"` and `ret_forest = TRUE` only): the RP
  forest used to initialize the neighbor data.

## Details

If no initial graph is provided, a random graph is generated, or you may
also specify the use of a graph generated from a forest of random
projection trees, using the method of Dasgupta and Freund (2008).

## References

Dasgupta, S., & Freund, Y. (2008, May). Random projection trees and low
dimensional manifolds. In *Proceedings of the fortieth annual ACM
symposium on Theory of computing* (pp. 537-546).
[doi:10.1145/1374376.1374452](https://doi.org/10.1145/1374376.1374452) .

Dong, W., Moses, C., & Li, K. (2011, March). Efficient k-nearest
neighbor graph construction for generic similarity measures. In
*Proceedings of the 20th international conference on World Wide Web*
(pp. 577-586). ACM.
[doi:10.1145/1963405.1963487](https://doi.org/10.1145/1963405.1963487) .

## Examples

``` r
# Find 4 (approximate) nearest neighbors using Euclidean distance
# If you pass a data frame, non-numeric columns are removed
iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean")

# Manhattan (l1) distance
iris_nn <- nnd_knn(iris, k = 4, metric = "manhattan")

# Multi-threading: you can choose the number of threads to use: in real
# usage, you will want to set n_threads to at least 2
iris_nn <- nnd_knn(iris, k = 4, metric = "manhattan", n_threads = 1)

# Use verbose flag to see information about progress
iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", verbose = TRUE)
#> 00:38:32 Using alt metric 'sqeuclidean' for 'euclidean'
#> 00:38:32 Initializing neighbors using 'rand' method
#> 00:38:32 Generating random k-nearest neighbor graph with k = 4
#> 00:38:32 Running nearest neighbor descent for 7 iterations
#> 00:38:32 Finished

# Nearest neighbor descent uses random initialization, but you can pass any
# approximation using the init argument (as long as the metrics used to
# calculate the initialization are compatible with the metric options used
# by nnd_knn).
iris_nn <- random_knn(iris, k = 4, metric = "euclidean")
iris_nn <- nnd_knn(iris, init = iris_nn, metric = "euclidean", verbose = TRUE)
#> 00:38:32 Using alt metric 'sqeuclidean' for 'euclidean'
#> 00:38:32 Initializing from user-supplied graph
#> 00:38:32 Applying metric correction to initial distances from 'euclidean' to 'sqeuclidean'
#> 00:38:32 Running nearest neighbor descent for 7 iterations
#> 00:38:32 Finished

# Number of iterations controls how much optimization is attempted. A smaller
# value will run faster but give poorer results
iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", n_iters = 2)

# You can also control the amount of work done within an iteration by
# setting max_candidates
iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", max_candidates = 50)

# Optimization may also stop early if not much progress is being made. This
# convergence criterion can be controlled via delta. A larger value will
# stop progress earlier. The verbose flag will provide some information if
# convergence is occurring before all iterations are carried out.
set.seed(1337)
iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", n_iters = 5, delta = 0.5)

# To ensure that descent only stops if no improvements are made, set delta = 0
set.seed(1337)
iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", n_iters = 5, delta = 0)

# A faster version of the algorithm is available that avoids repeated
# distance calculations at the cost of using more RAM. Set low_memory to
# FALSE to try it.
set.seed(1337)
iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", low_memory = FALSE)

# Using init = "tree" is usually more efficient than random initialization.
# arguments to the tree initialization method can be passed via the init_args
# list
set.seed(1337)
iris_nn <- nnd_knn(iris, k = 4, init = "tree", init_args = list(n_trees = 5))
```
