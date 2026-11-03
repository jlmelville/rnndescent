# Query an index for approximate nearest neighbors

Takes a nearest neighbor index produced by
[`rnnd_build()`](https://jlmelville.github.io/rnndescent/reference/rnnd_build.md)
and uses it to find the nearest neighbors of a query set of
observations, using a back-tracking search with the search size
determined by the method of Iwasaki and Miyazaki (2018). For further
control over the search effort, the total number of distance
calculations can also be bounded, similar to the method of Harwood and
Drummond (2016).

## Usage

``` r
rnnd_query(
  index,
  query,
  k = 30,
  epsilon = 0.1,
  max_search_fraction = 1,
  init = NULL,
  n_threads = 0,
  verbose = FALSE,
  obs = "R"
)
```

## Arguments

- index:

  A nearest neighbor index produced by
  [`rnnd_build()`](https://jlmelville.github.io/rnndescent/reference/rnnd_build.md).

- query:

  Matrix of `n` query items, with observations in the rows and features
  in the columns. Optionally, the data may be passed with the
  observations in the columns, by setting `obs = "C"`, which should be
  more efficient. Possible formats are
  [`base::data.frame()`](https://rdrr.io/r/base/data.frame.html),
  [`base::matrix()`](https://rdrr.io/r/base/matrix.html) or
  [`Matrix::sparseMatrix()`](https://rdrr.io/pkg/Matrix/man/sparseMatrix.html).
  Sparse matrices should be in `dgCMatrix` format. Dataframes will be
  converted to `numerical` matrix format internally, so if your data
  columns are `logical` and intended to be used with the specialized
  binary `metric`s, you should convert it to a logical matrix first
  (otherwise you will get the slower dense numerical version). Sparse
  and non-sparse data cannot be mixed, so if the data used to build
  index was sparse, the `query` data must also be sparse. and vice
  versa.

- k:

  Number of nearest neighbors to return.

- epsilon:

  Controls trade-off between accuracy and search cost, as described by
  Iwasaki and Miyazaki (2018). Setting `epsilon` to a positive value
  specifies a distance tolerance on whether to explore the neighbors of
  candidate points. The larger the value, the more neighbors will be
  searched. A value of 0.1 allows query-candidate distances to be 10%
  larger than the current most-distant neighbor of the query point, 0.2
  means 20%, and so on. Suggested values are between 0-0.5, although
  this value is highly dependent on the distribution of distances in the
  dataset (higher dimensional data should choose a smaller cutoff). Too
  large a value of `epsilon` will result in the query search approaching
  brute force comparison. Use this parameter in conjunction with
  `max_search_fraction` to prevent excessive run time. Default is 0.1.
  If you set `verbose = TRUE`, statistics of the number of distance
  calculations will be logged which can help you tune `epsilon`.

- max_search_fraction:

  Maximum fraction of the reference data to search. This is a value
  between 0 (search none of the reference data) and 1 (search all of the
  data if necessary). This works in conjunction with `epsilon` and will
  terminate the search early if the specified fraction of the reference
  data has been searched. Default is 1.

- init:

  An optional matrix of `k` initial nearest neighbors for each query
  point.

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

the approximate nearest neighbor index, a list containing:

- `idx` an n by k matrix containing the nearest neighbor indices.

- `dist` an n by k matrix containing the nearest neighbor distances.

## References

Harwood, B., & Drummond, T. (2016). Fanng: Fast approximate nearest
neighbour graphs. In *Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition* (pp. 5713-5722).

Iwasaki, M., & Miyazaki, D. (2018). Optimization of indexing based on
k-nearest neighbor graph for proximity search in high-dimensional data.
*arXiv preprint* *arXiv:1810.07355*. <https://arxiv.org/abs/1810.07355>

## See also

`rnnd_query()`

## Examples

``` r
iris_even <- iris[seq_len(nrow(iris)) %% 2 == 0, ]
iris_odd <- iris[seq_len(nrow(iris)) %% 2 == 1, ]

iris_even_index <- rnnd_build(iris_even, k = 4)
iris_odd_nbrs <- rnnd_query(index = iris_even_index, query = iris_odd, k = 4)
```
