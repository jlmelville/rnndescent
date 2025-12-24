# Query a random projection forest index for nearest neighbors

Run queries against a "forest" of Random Projection Trees (Dasgupta and
Freund, 2008), to return nearest neighbors taken from the reference data
used to build the forest.

## Usage

``` r
rpf_knn_query(
  query,
  reference,
  forest,
  k,
  cache = TRUE,
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
  calculated from this data and should be the same data used to build
  the `forest`. Optionally, the data may be passed with the observations
  in the columns, by setting `obs = "C"`, which should be more
  efficient. The `query` data must be passed in the same format and
  orientation as `reference`. Possible formats are
  [`base::data.frame()`](https://rdrr.io/r/base/data.frame.html),
  [`base::matrix()`](https://rdrr.io/r/base/matrix.html) or
  [`Matrix::sparseMatrix()`](https://rdrr.io/pkg/Matrix/man/sparseMatrix.html).
  Sparse matrices should be in `dgCMatrix` format.

- forest:

  A random partition forest, created by
  [`rpf_build()`](https://jlmelville.github.io/rnndescent/reference/rpf_build.md),
  representing partitions of the data in `reference`.

- k:

  Number of nearest neighbors to return. You are unlikely to get good
  results if you choose a value substantially larger than the value of
  `leaf_size` used to build the `forest`.

- cache:

  if `TRUE` (the default) then candidate indices found in the leaves of
  the forest are cached to avoid recalculating the same distance
  repeatedly. This incurs an extra memory cost which scales with
  `n_threads`. Set this to `FALSE` to disable distance caching.

- n_threads:

  Number of threads to use. Note that the parallelism in the search is
  done over the observations in `query` not the trees in the `forest`.
  Thus a single observation will not see any speed-up from increasing
  `n_threads`.

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

the approximate nearest neighbor graph as a list containing:

- `idx` an n by k matrix containing the nearest neighbor indices.

- `dist` an n by k matrix containing the nearest neighbor distances.

`k` neighbors per observation are not guaranteed to be found. Missing
data is represented with an index of `0` and a distance of `NA`.

## References

Dasgupta, S., & Freund, Y. (2008, May). Random projection trees and low
dimensional manifolds. In *Proceedings of the fortieth annual ACM
symposium on Theory of computing* (pp. 537-546).
[doi:10.1145/1374376.1374452](https://doi.org/10.1145/1374376.1374452) .

## See also

[`rpf_build()`](https://jlmelville.github.io/rnndescent/reference/rpf_build.md)

## Examples

``` r
# Build a forest of 10 trees from the odd rows
iris_odd <- iris[seq_len(nrow(iris)) %% 2 == 1, ]
iris_odd_forest <- rpf_build(iris_odd, n_trees = 10)

iris_even <- iris[seq_len(nrow(iris)) %% 2 == 0, ]
iris_even_nn <- rpf_knn_query(
  query = iris_even, reference = iris_odd,
  forest = iris_odd_forest, k = 15
)
```
