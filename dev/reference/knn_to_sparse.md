# Convert a neighbor graph to a sparse matrix

Rows represent observations or queries and columns represent reference
observations. Set `transpose = TRUE` to reverse the axes. Missing
neighbors (`idx = 0`, `dist = NA`) are omitted. Stored zero distances
are retained. If a row contains the same neighbor more than once, the
last stored distance is used.

## Usage

``` r
knn_to_sparse(graph, n_reference = nrow(graph$idx), transpose = FALSE)
```

## Arguments

- graph:

  A neighbor graph containing `idx` and `dist` matrices.

- n_reference:

  Number of reference observations. Supply this explicitly for query
  results when the reference size differs from the number of rows.

- transpose:

  Whether to put reference observations in rows.

## Value

A numeric `dgCMatrix` of distances.

## Examples

``` r
graph <- brute_force_knn(iris[1:8, 1:4], k = 5)
knn_to_sparse(graph)
#> 8 x 8 sparse Matrix of class "dgCMatrix"
#>                                                                             
#> [1,] 0.0000000 .         0.5099020 .         0.1414212 . 0.5196152 0.1732050
#> [2,] .         0.0000000 0.3000002 0.3316626 .         . 0.5099021 0.4242641
#> [3,] .         0.3000002 0.0000000 0.2449490 .         . 0.2645751 0.4123107
#> [4,] .         0.3316626 0.2449490 0.0000000 .         . 0.3316627 0.5000002
#> [5,] 0.1414212 .         0.5099019 .         0.0000000 . 0.4582576 0.2236066
#> [6,] 0.6164416 .         .         .         0.6164416 0 0.9949876 0.7000001
#> [7,] .         .         0.2645751 0.3316627 0.4582576 . 0.0000000 0.4242642
#> [8,] 0.1732050 0.4242641 0.4123107 .         0.2236066 . .         0.0000000
```
