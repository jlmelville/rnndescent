# Brute Force Search

``` r

library(rnndescent)
```

If your dataset is sufficiently “small” (either in terms of number of
points or number of dimensions) you can use brute force search to find
the nearest neighbors: just calculate the distances between all pairs of
points.

This has the advantages of simplicity and accuracy. But it is of course
very slow. That said as it can be easily parallelized, you can get a
surprisingly long way with it, especially if you only need the k-nearest
neighbors and won’t be running a neighbor search multiple times.

Let’s calculate the exact 15 nearest neighbors for the `iris` dataset:

``` r

iris_nbrs <- brute_force_knn(iris, k = 15)
lapply(iris_nbrs, function(x) {
  head(x[, 1:5], 3)
})
#> $idx
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1   18   29    5   40
#> [2,]    2   13   46   35   10
#> [3,]    3   48    4    7   13
#> 
#> $dist
#>      [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,]    0 0.1000000 0.1414212 0.1414212 0.1414213
#> [2,]    0 0.1414213 0.1414213 0.1414213 0.1732050
#> [3,]    0 0.1414213 0.2449490 0.2645751 0.2645753
```

And here’s an example of querying the odd items in the `iris` data
against the even items (i.e. the indices in `iris_nbrs` will refer to
the odd items):

``` r

iris_even <- iris[seq_len(nrow(iris)) %% 2 == 0, ]
iris_odd <- iris[seq_len(nrow(iris)) %% 2 == 1, ]
iris_query_nbrs <-
  brute_force_knn_query(
    query = iris_even,
    reference = iris_odd,
    k = 15
  )
lapply(iris_query_nbrs, function(x) {
  head(x[, 1:5], 3)
})
#> $idx
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    7   18   16    2   15
#> [2,]   16    2    7    5   20
#> [3,]   10    6   25   23   24
#> 
#> $dist
#>           [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,] 0.1414213 0.1414213 0.2449489 0.3000002 0.4999999
#> [2,] 0.2236071 0.2449490 0.2645753 0.2999998 0.2999999
#> [3,] 0.3316623 0.3464102 0.3605552 0.3741659 0.3872985
```
