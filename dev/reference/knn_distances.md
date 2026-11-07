# Calculate distances for fixed neighbor indices

Calculate distances for the supplied indices in their existing order.
Rows of `idx` correspond to observations, or to queries when `query` is
supplied. Indices always refer to observations in `data`. Missing
indices (`0`) remain paired with `NA` distances.

## Usage

``` r
knn_distances(
  data,
  idx,
  metric = "euclidean",
  query = NULL,
  n_threads = 0,
  verbose = FALSE,
  obs = "R"
)
```

## Arguments

- data:

  Reference observations, in a numeric matrix or data frame, a logical
  matrix, or a numeric sparse matrix.

- idx:

  Numeric matrix of whole-number neighbor indices. For a graph, pass
  `graph$idx`.

- metric:

  Distance metric, evaluated directly. See
  [`brute_force_knn()`](https://jlmelville.github.io/rnndescent/dev/reference/brute_force_knn.md).

- query:

  Optional query observations with the same feature layout as `data`.
  Query and reference must both be sparse or both dense, and their
  logical or numeric storage must match. Logical features cannot be
  `NA`.

- n_threads:

  Number of threads to use. Zero runs serially.

- verbose:

  Whether to report progress.

- obs:

  Whether observations are in rows (`"R"`) or columns (`"C"`).

## Value

A list containing the supplied `idx` and calculated `dist` matrices.

## Examples

``` r
graph <- brute_force_knn(iris[1:8, 1:4], k = 5)
knn_distances(iris[1:8, 1:4], graph$idx, metric = "manhattan")
#> $idx
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1    5    8    3    7
#> [2,]    2    3    4    8    7
#> [3,]    3    4    7    2    8
#> [4,]    4    3    2    7    8
#> [5,]    5    1    8    7    3
#> [6,]    6    5    1    8    7
#> [7,]    7    3    4    8    5
#> [8,]    8    1    5    3    2
#> 
#> $dist
#>      [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,]    0 0.1999998 0.2999998 0.8000001 0.6999999
#> [2,]    0 0.5000004 0.5000001 0.6000000 0.8000003
#> [3,]    0 0.4000001 0.5000000 0.5000004 0.7000003
#> [4,]    0 0.4000001 0.5000001 0.5000002 0.7000003
#> [5,]    0 0.1999998 0.2999998 0.6999999 0.8000001
#> [6,]    0 1.2000004 1.2000004 1.3000002 1.7000003
#> [7,]    0 0.5000000 0.5000002 0.6000001 0.6999999
#> [8,]    0 0.2999998 0.2999998 0.7000003 0.6000000
#> 
```
