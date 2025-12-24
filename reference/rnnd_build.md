# Build approximate nearest neighbors index and neighbor graph

This function builds an approximate nearest neighbors graph with
convenient defaults, then prepares the index for querying new data, for
later use with
[`rnnd_query()`](https://jlmelville.github.io/rnndescent/reference/rnnd_query.md).
For more control over the process, please see the other functions in the
package.

## Usage

``` r
rnnd_build(
  data,
  k = 30,
  metric = "euclidean",
  use_alt_metric = TRUE,
  init = "tree",
  n_trees = NULL,
  leaf_size = NULL,
  max_tree_depth = 200,
  margin = "auto",
  n_iters = NULL,
  delta = 0.001,
  max_candidates = NULL,
  low_memory = TRUE,
  weight_by_degree = FALSE,
  n_search_trees = 1,
  pruning_degree_multiplier = 1.5,
  diversify_prob = 1,
  prune_reverse = FALSE,
  n_threads = 0,
  verbose = FALSE,
  progress = "bar",
  obs = "R"
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

  Number of nearest neighbors to build the index for. You can specify a
  different number when running `rnnd_query`, but the index is
  calibrated using this value so it's recommended to set `k` according
  to the likely number of neighbors you will want to retrieve. Optional
  if `init` is specified, in which case `k` can be inferred from the
  `init` data. If you do both, then the specified version of `k` will
  take precedence.

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

- use_alt_metric:

  If `TRUE`, use faster metrics that maintain the ordering of distances
  internally (e.g. squared Euclidean distances if using
  `metric = "euclidean"`), then apply a correction at the end. Probably
  the only reason to set this to `FALSE` is if you suspect that some
  sort of numeric issue is occurring with your data in the alternative
  code path.

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

- n_trees:

  The number of trees to use in the RP forest. A larger number will give
  more accurate results at the cost of a longer computation time. The
  default of `NULL` means that the number is chosen based on the number
  of observations in `data`. Only used if `init = "tree"`.

- leaf_size:

  The maximum number of items that can appear in a leaf. This value
  should be chosen to match the expected number of neighbors you will
  want to retrieve when running queries (e.g. if you want find 50
  nearest neighbors set `leaf_size = 50`) and should not be set to a
  value smaller than `10`. Only used if `init = "tree"`.

- max_tree_depth:

  The maximum depth of the tree to build (default = 200). If the maximum
  tree depth is exceeded then the leaf size of a tree may exceed
  `leaf_size` which can result in a large number of neighbor distances
  being calculated. If `verbose = TRUE` a message will be logged to
  indicate that the leaf size is large. However, increasing the
  `max_tree_depth` may not help: it may be that there is something
  unusual about the distribution of your data set under your chose
  `metric` that makes a tree-based initialization inappropriate. Only
  used if `init = "tree"`.

- margin:

  A character string specifying the method used to assign points to one
  side of the hyperplane or the other. Possible values are:

  - `"explicit"` categorizes all distance metrics as either Euclidean or
    Angular (Euclidean after normalization), explicitly calculates a
    hyperplane and offset, and then calculates the margin based on the
    dot product with the hyperplane.

  - `"implicit"` calculates the distance from a point to each of the
    points defining the normal vector. The margin is calculated by
    comparing the two distances: the point is assigned to the side of
    the hyperplane that the normal vector point with the closest
    distance belongs to.

  - `"auto"` (the default) picks the margin method depending on whether
    a binary-specific `metric` such as `"bhammming"` is chosen, in which
    case `"implicit"` is used, and `"explicit"` otherwise:
    binary-specific metrics involve storing the data in a way that isn't
    very efficient for the `"explicit"` method and the binary-specific
    metric is usually a lot faster than the generic equivalent such that
    the cost of two distance calculations for the margin method is still
    faster.

  Only used if `init = "tree"`.

- n_iters:

  Number of iterations of nearest neighbor descent to carry out. By
  default, this will be chosen based on the number of observations in
  `data`.

- delta:

  The minimum relative change in the neighbor graph allowed before early
  stopping. Should be a value between 0 and 1. The smaller the value,
  the smaller the amount of progress between iterations is allowed.
  Default value of `0.001` means that at least 0.1% of the neighbor
  graph must be updated at each iteration.

- max_candidates:

  Maximum number of candidate neighbors to try for each item in each
  iteration. Use relative to `k` to emulate the "rho" sampling parameter
  in the nearest neighbor descent paper. By default, this is set to `k`
  or `60`, whichever is smaller.

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

- n_search_trees, :

  the number of trees to keep in the search forest as part of index
  preparation. The default is `1`.

- pruning_degree_multiplier:

  How strongly to truncate the final neighbor list for each item. The
  neighbor list of each item will be truncated to retain only the
  closest `d` neighbors, where `d = k * pruning_degree_multiplier`, and
  `k` is the original number of neighbors per item in `graph`. Roughly,
  values larger than `1` will keep all the nearest neighbors of an item,
  plus the given fraction of reverse neighbors (if they exist). For
  example, setting this to `1.5` will keep all the forward neighbors and
  then half as many of the reverse neighbors, although exactly which
  neighbors are retained is also dependent on any occlusion pruning that
  occurs. Set this to `NULL` to skip this step.

- diversify_prob:

  the degree of diversification of the search graph by removing
  unnecessary edges through occlusion pruning. This should take a value
  between `0` (no diversification) and `1` (remove as many edges as
  possible) and is treated as the probability of a neighbor being
  removed if it is found to be an "occlusion". If item `p` and `q`, two
  members of the neighbor list of item `i`, are closer to each other
  than they are to `i`, then the nearer neighbor `p` is said to
  "occlude" `q`. It is likely that `q` will be in the neighbor list of
  `p` so there is no need to retain it in the neighbor list of `i`. You
  may also set this to `NULL` to skip any occlusion pruning. Note that
  occlusion pruning is carried out twice, once to the forward neighbors,
  and once to the reverse neighbors.

- prune_reverse:

  If `TRUE`, prune the reverse neighbors of each item before the reverse
  graph diversification step using `pruning_degree_multiplier`. Because
  the number of reverse neighbors can be much larger than the number of
  forward neighbors, this can help to avoid excessive computation during
  the diversification step, with little overall effect on the final
  search graph. Default is `FALSE`.

- n_threads:

  Number of threads to use.

- verbose:

  If `TRUE`, log information to the console.

- progress:

  Determines the type of progress information logged during the nearest
  neighbor descent stage when `verbose = TRUE`. Options are:

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

## Value

the approximate nearest neighbor index, a list containing:

- `graph` the k-nearest neighbor graph, a list containing:

  - `idx` an n by k matrix containing the nearest neighbor indices.

  - `dist` an n by k matrix containing the nearest neighbor distances.

- Other list items are intended only for internal use by other functions
  such as
  [`rnnd_query()`](https://jlmelville.github.io/rnndescent/reference/rnnd_query.md).

## Details

The process of k-nearest neighbor graph construction using Random
Projection Forests (Dasgupta and Freund, 2008) for initialization and
Nearest Neighbor Descent (Dong and co-workers, 2011) for refinement.
Index preparation, uses the graph diversification method of Harwood and
Drummond (2016).

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

Harwood, B., & Drummond, T. (2016). Fanng: Fast approximate nearest
neighbour graphs. In *Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition* (pp. 5713-5722).

## See also

[`rnnd_query()`](https://jlmelville.github.io/rnndescent/reference/rnnd_query.md)

## Examples

``` r
iris_even <- iris[seq_len(nrow(iris)) %% 2 == 0, ]
iris_odd <- iris[seq_len(nrow(iris)) %% 2 == 1, ]

# Find 4 (approximate) nearest neighbors using Euclidean distance
iris_even_index <- rnnd_build(iris_even, k = 4)
iris_odd_nbrs <- rnnd_query(index = iris_even_index, query = iris_odd, k = 4)
```
