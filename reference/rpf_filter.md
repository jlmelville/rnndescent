# Keep the best trees in a random projection forest

Reduce the size of a random projection forest, by scoring each tree
against a k-nearest neighbors graph. Only the top N trees will be
retained which allows for a faster querying.

## Usage

``` r
rpf_filter(nn, forest = NULL, n_trees = 1, n_threads = 0, verbose = FALSE)
```

## Arguments

- nn:

  Nearest neighbor data in the dense list format. This should be derived
  from the same data that was used to build the `forest`.

- forest:

  A random partition forest, e.g. created by
  [`rpf_build()`](https://jlmelville.github.io/rnndescent/reference/rpf_build.md),
  representing partitions of the same underlying data reflected in `nn`.
  As a convenient, this parameter is ignored if the `nn` list contains a
  `forest` entry, e.g. from running
  [`rpf_knn()`](https://jlmelville.github.io/rnndescent/reference/rpf_knn.md)
  or
  [`nnd_knn()`](https://jlmelville.github.io/rnndescent/reference/nnd_knn.md)
  with `ret_forest = TRUE`, and the forest value will be extracted from
  `nn`.

- n_trees:

  The number of trees to retain. By default only the best-scoring tree
  is retained.

- n_threads:

  Number of threads to use.

- verbose:

  If `TRUE`, log information to the console.

## Value

A forest with the best scoring `n_trees` trees.

## Details

Trees are scored based on how well each leaf reflects the neighbors as
specified in the nearest neighbor data. It's best to use as accurate
nearest neighbor data as you can and it does not need to come directly
from searching the `forest`: for example, the nearest neighbor data from
running
[`nnd_knn()`](https://jlmelville.github.io/rnndescent/reference/nnd_knn.md)
to optimize the neighbor data output from an RP Forest is a good choice.

Rather than rely on an RP Forest solely for approximate nearest neighbor
querying, it is probably more cost-effective to use a small number of
trees to initialize the neighbor list for use in a graph search via
[`graph_knn_query()`](https://jlmelville.github.io/rnndescent/reference/graph_knn_query.md).

## See also

[`rpf_build()`](https://jlmelville.github.io/rnndescent/reference/rpf_build.md)

## Examples

``` r
# Build a knn with a forest of 10 trees using the odd rows
iris_odd <- iris[seq_len(nrow(iris)) %% 2 == 1, ]
# also return the forest with the knn
rfknn <- rpf_knn(iris_odd, k = 15, n_trees = 10, ret_forest = TRUE)

# keep the best 2 trees:
iris_odd_filtered_forest <- rpf_filter(rfknn)

# get some new data to search
iris_even <- iris[seq_len(nrow(iris)) %% 2 == 0, ]

# search with the filtered forest
iris_even_nn <- rpf_knn_query(
  query = iris_even, reference = iris_odd,
  forest = iris_odd_filtered_forest, k = 15
)
```
