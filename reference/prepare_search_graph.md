# Convert a nearest neighbor graph into a search graph

Create a graph using existing nearest neighbor data to balance search
speed and accuracy using the occlusion pruning and truncation strategies
of Harwood and Drummond (2016). The resulting search graph should be
more efficient for querying new data than the original nearest neighbor
graph.

## Usage

``` r
prepare_search_graph(
  data,
  graph,
  metric = "euclidean",
  use_alt_metric = TRUE,
  diversify_prob = 1,
  pruning_degree_multiplier = 1.5,
  prune_reverse = FALSE,
  n_threads = 0,
  verbose = FALSE,
  obs = "R"
)
```

## Arguments

- data:

  Matrix of `n` items, with observations in the rows and features in the
  columns. Optionally, input can be passed with observations in the
  columns, by setting `obs = "C"`, which should be more efficient.
  Possible formats are
  [`base::data.frame()`](https://rdrr.io/r/base/data.frame.html),
  [`base::matrix()`](https://rdrr.io/r/base/matrix.html) or
  [`Matrix::sparseMatrix()`](https://rdrr.io/pkg/Matrix/man/sparseMatrix.html).
  Sparse matrices should be in `dgCMatrix` format. Dataframes will be
  converted to `numerical` matrix format internally, so if your data
  columns are `logical` and intended to be used with the specialized
  binary `metric`s, you should convert it to a logical matrix first
  (otherwise you will get the slower dense numerical version).

- graph:

  neighbor graph for `data`, a list containing:

  - `idx` an `n` by `k` matrix containing the nearest neighbor indices
    of the data in `data`.

  - `dist` an `n` by `k` matrix containing the nearest neighbor
    distances.

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
  and once to the reverse neighbors. Reducing this value will result in
  a more dense graph. This is similar to increasing the "alpha"
  parameter used by in the DiskAnn pruning method of Subramanya and
  co-workers (2014).

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

- obs:

  set to `"C"` to indicate that the input `data` orientation stores each
  observation as a column. The default `"R"` means that observations are
  stored in each row. Storing the data by row is usually more
  convenient, but internally your data will be converted to column
  storage. Passing it already column-oriented will save some memory and
  (a small amount of) CPU usage.

## Value

a search graph for `data` based on `graph`, represented as a sparse
matrix, suitable for use with
[`graph_knn_query()`](https://jlmelville.github.io/rnndescent/reference/graph_knn_query.md).

## Details

An approximate nearest neighbor graph is not very useful for querying
via
[`graph_knn_query()`](https://jlmelville.github.io/rnndescent/reference/graph_knn_query.md),
especially if the query data is initialized randomly: some items in the
data set may not be in the nearest neighbor list of any other item and
can therefore never be returned as a neighbor, no matter how close they
are to the query. Even those which do appear in at least one neighbor
list may not be reachable by expanding an arbitrary starting list if the
neighbor graph contains disconnected components.

Converting the directed graph represented by the neighbor graph to an
undirected graph by adding an edge from item `j` to `i` if an edge
exists from `i` to `j` (i.e. creating the mutual neighbor graph) solves
the problems above, but can result in inefficient searches. Although the
out-degree of each item is restricted to the number of neighbors the
in-degree has no such restrictions: a given item could be very "popular"
and in a large number of neighbors lists. Therefore mutualizing the
neighbor graph can result in some items with a large number of neighbors
to search. These usually have very similar neighborhoods so there is
nothing to be gained from searching all of them.

To balance accuracy and search time, the following procedure is carried
out:

1.  The graph is "diversified" by occlusion pruning.

2.  The reverse graph is formed by reversing the direction of all edges
    in the pruned graph.

3.  The reverse graph is diversified by occlusion pruning.

4.  The pruned forward and pruned reverse graph are merged.

5.  The outdegree of each node in the merged graph is truncated.

6.  The truncated merged graph is returned as the prepared search graph.

Explicit zero distances in the `graph` will be converted to a small
positive number to avoid being dropped in the sparse representation. The
one exception is the "self" distance, i.e. any edge in the `graph` which
links a node to itself (the diagonal of the sparse distance matrix).
These trivial edges aren't useful for search purposes and are always
dropped.

## References

Harwood, B., & Drummond, T. (2016). Fanng: Fast approximate nearest
neighbour graphs. In *Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition* (pp. 5713-5722).

Jayaram Subramanya, S., Devvrit, F., Simhadri, H. V., Krishnawamy, R., &
Kadekodi, R. (2019). Diskann: Fast accurate billion-point nearest
neighbor search on a single node. *Advances in Neural Information
Processing Systems*, *32*.

## See also

[`graph_knn_query()`](https://jlmelville.github.io/rnndescent/reference/graph_knn_query.md)

## Examples

``` r
# 100 reference iris items
iris_ref <- iris[iris$Species %in% c("setosa", "versicolor"), ]

# 50 query items
iris_query <- iris[iris$Species == "versicolor", ]

# First, find the approximate 4-nearest neighbor graph for the references:
ref_ann_graph <- nnd_knn(iris_ref, k = 4)

# Create a graph for querying with
ref_search_graph <- prepare_search_graph(iris_ref, ref_ann_graph)

# Using the search graph rather than the ref_ann_graph directly may give
# more accurate or faster results
iris_query_nn <- graph_knn_query(
  query = iris_query, reference = iris_ref,
  reference_graph = ref_search_graph, k = 4, metric = "euclidean",
  verbose = TRUE
)
#> 13:30:05 Using alt metric 'sqeuclidean' for 'euclidean'
#> 13:30:05 Initializing from random neighbors
#> 13:30:05 Generating random k-nearest neighbor graph from reference with k = 4
#> 13:30:05 Searching nearest neighbor graph with epsilon = 0.1 and max_search_fraction = 1
#> 13:30:05 Finished
```
