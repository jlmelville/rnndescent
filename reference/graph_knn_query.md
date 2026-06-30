# Query a search graph for nearest neighbors

Run queries against a search graph, to return nearest neighbors taken
from the reference data used to build that graph.

## Usage

``` r
graph_knn_query(
  query,
  reference,
  reference_graph,
  k = NULL,
  metric = "euclidean",
  init = NULL,
  epsilon = 0.1,
  max_search_fraction = 1,
  use_alt_metric = TRUE,
  n_threads = 0,
  verbose = FALSE,
  obs = "R"
)
```

## Arguments

- query:

  Matrix of `n` query items, with observations in the rows and features
  in the columns. Optionally, the data may be passed with the
  observations in the columns, by setting `obs = "C"`, which should be
  more efficient. The `reference` data must be passed in the same
  orientation as `query`. Possible formats are
  [`base::data.frame()`](https://rdrr.io/r/base/data.frame.html),
  [`base::matrix()`](https://rdrr.io/r/base/matrix.html) or
  [`Matrix::sparseMatrix()`](https://rdrr.io/pkg/Matrix/man/sparseMatrix.html).
  Sparse matrices should be in `dgCMatrix` format. Dataframes will be
  converted to `numerical` matrix format internally, so if your data
  columns are `logical` and intended to be used with the specialized
  binary `metric`s, you should convert it to a logical matrix first
  (otherwise you will get the slower dense numerical version).

- reference:

  Matrix of `m` reference items, with observations in the rows and
  features in the columns. The nearest neighbors to the queries are
  calculated from this data. Optionally, the data may be passed with the
  observations in the columns, by setting `obs = "C"`, which should be
  more efficient. The `query` data must be passed in the same format and
  orientation as `reference`. Possible formats are
  [`base::data.frame()`](https://rdrr.io/r/base/data.frame.html),
  [`base::matrix()`](https://rdrr.io/r/base/matrix.html) or
  [`Matrix::sparseMatrix()`](https://rdrr.io/pkg/Matrix/man/sparseMatrix.html).
  Sparse matrices should be in `dgCMatrix` format.

- reference_graph:

  Search graph of the `reference` data. A neighbor graph, such as that
  output from
  [`nnd_knn()`](https://jlmelville.github.io/rnndescent/reference/nnd_knn.md)
  can be used, but preferably a suitably prepared sparse search graph
  should be used, such as that output by
  [`prepare_search_graph()`](https://jlmelville.github.io/rnndescent/reference/prepare_search_graph.md).

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

  Initial `query` neighbor graph to optimize. If not provided, `k`
  random neighbors are created. If provided, the input format must be
  one of:

  1.  A list containing:

      - `idx` an `n` by `k` matrix containing the nearest neighbor
        indices.

      - `dist` (optional) an `n` by `k` matrix containing the nearest
        neighbor distances.

      If `k` and `init` are specified as arguments to this function, and
      the number of neighbors provided in `init` is not equal to `k`
      then:

      - if `k` is smaller, only the `k` closest values in `init` are
        retained.

      - if `k` is larger, then random neighbors will be chosen to fill
        `init` to the size of `k`. Note that there is no checking if any
        of the random neighbors are duplicates of what is already in
        `init` so effectively fewer than `k` neighbors may be chosen for
        some observations under these circumstances.

      If the input distances are omitted, they will be calculated for
      you.

  2.  A random projection forest, such as that returned from
      [`rpf_build()`](https://jlmelville.github.io/rnndescent/reference/rpf_build.md)
      or
      [`rpf_knn()`](https://jlmelville.github.io/rnndescent/reference/rpf_knn.md)
      with `ret_forest = TRUE`.

- epsilon:

  Controls trade-off between accuracy and search cost, as described by
  Iwasaki and Miyazaki (2018), by specifying a distance tolerance on
  whether to explore the neighbors of candidate points. The larger the
  value, the more neighbors will be searched. A value of 0.1 allows
  query-candidate distances to be 10% larger than the current
  most-distant neighbor of the query point, 0.2 means 20%, and so on.
  Suggested values are between 0-0.5, although this value is highly
  dependent on the distribution of distances in the dataset (higher
  dimensional data should choose a smaller cutoff). Too large a value of
  `epsilon` will result in the query search approaching brute force
  comparison. Use this parameter in conjunction with
  `max_search_fraction` and
  [`prepare_search_graph()`](https://jlmelville.github.io/rnndescent/reference/prepare_search_graph.md)
  to prevent excessive run time. Default is 0.1. If you set
  `verbose = TRUE`, statistics of the number of distance calculations
  will be logged which can help you tune `epsilon`.

- max_search_fraction:

  Maximum fraction of the reference data to search. This is a value
  between 0 (search none of the reference data) and 1 (search all of the
  data if necessary). This works in conjunction with `epsilon` and will
  terminate the search early if the specified fraction of the reference
  data has been searched. Default is 1.

- use_alt_metric:

  If `TRUE`, use faster metrics that maintain the ordering of distances
  internally (e.g. squared Euclidean distances if using
  `metric = "euclidean"`), then apply a correction at the end. Probably
  the only reason to set this to `FALSE` is if you suspect that some
  sort of numeric issue is occurring with your data in the alternative
  code path. If a search forest is used for initialization via the
  `init` parameter, then the metric is fetched from there and this
  setting is ignored.

- n_threads:

  Number of threads to use.

- verbose:

  If `TRUE`, log information to the console.

- obs:

  set to `"C"` to indicate that the input `query` and `reference`
  orientation stores each observation as a column (the orientation must
  be consistent). The default `"R"` means that observations are stored
  in each row. Storing the data by row is usually more convenient, but
  internally your data will be converted to column storage. Passing it
  already column-oriented will save some memory and (a small amount of)
  CPU usage.

## Value

the approximate nearest neighbor graph as a list containing:

- `idx` a `n` by `k` matrix containing the nearest neighbor indices
  specifying the row of the neighbor in `reference`.

- `dist` a `n` by `k` matrix containing the nearest neighbor distances.

## Details

A greedy beam search is used to query the graph, combining two search
pruning strategies. The first, due to Iwasaki and Miyazaki (2018), only
considers new candidates within a relative distance of the current
furthest neighbor in the query's graph. The second, due to Harwood and
Drummond (2016), puts a limit on the absolute number of distance
calculations to carry out. See the `epsilon` and `max_search_fraction`
parameters respectively.

## References

Harwood, B., & Drummond, T. (2016). Fanng: Fast approximate nearest
neighbour graphs. In *Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition* (pp. 5713-5722).

Iwasaki, M., & Miyazaki, D. (2018). Optimization of indexing based on
k-nearest neighbor graph for proximity search in high-dimensional data.
*arXiv preprint arXiv:1810.07355*.

## Examples

``` r
# 100 reference iris items
iris_ref <- iris[iris$Species %in% c("setosa", "versicolor"), ]

# 50 query items
iris_query <- iris[iris$Species == "versicolor", ]

# First, find the approximate 4-nearest neighbor graph for the references:
iris_ref_graph <- nnd_knn(iris_ref, k = 4)

# For each item in iris_query find the 4 nearest neighbors in iris_ref.
# You need to pass both the reference data and the reference graph.
# If you pass a data frame, non-numeric columns are removed.
# set verbose = TRUE to get details on the progress being made
iris_query_nn <- graph_knn_query(iris_query, iris_ref, iris_ref_graph,
  k = 4, metric = "euclidean", verbose = TRUE
)
#> 01:57:16 Using alt metric 'sqeuclidean' for 'euclidean'
#> 01:57:16 Initializing from random neighbors
#> 01:57:16 Generating random k-nearest neighbor graph from reference with k = 4
#> 01:57:16 Searching nearest neighbor graph with epsilon = 0.1 and max_search_fraction = 1
#> 01:57:16 Finished

# A more complete example, converting the initial knn into a search graph
# and using a filtered random projection forest to initialize the search
# create initial knn and forest
iris_ref_graph <- nnd_knn(iris_ref, k = 4, init = "tree", ret_forest = TRUE)
# keep the best tree in the forest
forest <- rpf_filter(iris_ref_graph, n_trees = 1)
# expand the knn into a search graph
iris_ref_search_graph <- prepare_search_graph(iris_ref, iris_ref_graph)
# run the query with the improved graph and initialization
iris_query_nn <- graph_knn_query(iris_query, iris_ref, iris_ref_search_graph,
  init = forest, k = 4
)
```
