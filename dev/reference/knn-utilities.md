# Manipulate nearest neighbor graphs

Truncate or reorder the neighbors in a graph, or adjust self neighbors
in a graph of a dataset against itself. Each function returns a list
containing only `idx` and `dist` matrices.

## Usage

``` r
truncate_knn(graph, k)

sort_knn(graph, by = NULL)

remove_self_knn(graph, k = max(0L, ncol(graph$idx) - 1L))

self_first_knn(graph, k = ncol(graph$idx), self_dist = NULL)
```

## Arguments

- graph:

  A nearest neighbor graph with `idx` and `dist` matrices, such as the
  result of
  [`brute_force_knn()`](https://jlmelville.github.io/rnndescent/dev/reference/brute_force_knn.md).

- k:

  Number of columns to retain. For `truncate_knn()`, from zero to the
  stored width. For `remove_self_knn()`, it is the number of nonself
  neighbors. For `self_first_knn()`, it is the total width including
  self, at least one. Supplying `self_dist` permits a width one greater
  than the stored width when every row has enough nonself neighbors.

- by:

  Optional numeric matrix of scores with the same dimensions as the
  graph matrices. Defaults to the stored distances.

- self_dist:

  Finite numeric scalar used when a row has no self entry. The default
  `NULL` errors if self is absent.

## Value

A list containing the resulting `idx` and `dist` matrices.

## Details

`truncate_knn()` keeps the first `k` columns. `sort_knn()` orders each
row by ascending distance, or by the supplied score matrix. Ties retain
their original order. Missing scores follow other scores, and missing
neighbors (`idx = 0`) always come last. Distances remain paired with
their indices.

`remove_self_knn()` keeps the first `k` present, nonself neighbors in
each row (by default, one fewer than the stored width, with a minimum of
zero). `self_first_knn()` puts self first, retaining the distance of the
first existing self entry, and follows it with the first `k - 1`
present, nonself neighbors. If self is absent, supply `self_dist` to
insert it. Both functions identify self by the row number, skip missing
entries and other self entries, and error if a row has too few nonself
neighbors. They are for self graphs. Do not use them on query results,
where row numbers identify queries rather than reference observations.

## Examples

``` r
graph <- brute_force_knn(iris[1:8, 1:4], k = 6)
truncate_knn(graph, k = 3)
#> $idx
#>      [,1] [,2] [,3]
#> [1,]    1    5    8
#> [2,]    2    3    4
#> [3,]    3    4    7
#> [4,]    4    3    2
#> [5,]    5    1    8
#> [6,]    6    5    1
#> [7,]    7    3    4
#> [8,]    8    1    5
#> 
#> $dist
#>      [,1]      [,2]      [,3]
#> [1,]    0 0.1414212 0.1732050
#> [2,]    0 0.3000002 0.3316626
#> [3,]    0 0.2449490 0.2645751
#> [4,]    0 0.2449490 0.3316626
#> [5,]    0 0.1414212 0.2236066
#> [6,]    0 0.6164416 0.6164416
#> [7,]    0 0.2645751 0.3316627
#> [8,]    0 0.1732050 0.2236066
#> 
remove_self_knn(graph, k = 5)
#> $idx
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    5    8    3    7    2
#> [2,]    3    4    8    7    1
#> [3,]    4    7    2    8    5
#> [4,]    3    2    7    8    5
#> [5,]    1    8    7    3    2
#> [6,]    5    1    8    7    3
#> [7,]    3    4    8    5    2
#> [8,]    1    5    3    2    7
#> 
#> $dist
#>           [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,] 0.1414212 0.1732050 0.5099020 0.5196152 0.5385164
#> [2,] 0.3000002 0.3316626 0.4242641 0.5099021 0.5385164
#> [3,] 0.2449490 0.2645751 0.3000002 0.4123107 0.5099019
#> [4,] 0.2449490 0.3316626 0.3316627 0.5000002 0.6480741
#> [5,] 0.1414212 0.2236066 0.4582576 0.5099019 0.6082762
#> [6,] 0.6164416 0.6164416 0.7000001 0.9949876 1.0862783
#> [7,] 0.2645751 0.3316627 0.4242642 0.4582576 0.5099021
#> [8,] 0.1732050 0.2236066 0.4123107 0.4242641 0.4242642
#> 
sort_knn(graph)
#> $idx
#>      [,1] [,2] [,3] [,4] [,5] [,6]
#> [1,]    1    5    8    3    7    2
#> [2,]    2    3    4    8    7    1
#> [3,]    3    4    7    2    8    5
#> [4,]    4    3    2    7    8    5
#> [5,]    5    1    8    7    3    2
#> [6,]    6    5    1    8    7    3
#> [7,]    7    3    4    8    5    2
#> [8,]    8    1    5    3    2    7
#> 
#> $dist
#>      [,1]      [,2]      [,3]      [,4]      [,5]      [,6]
#> [1,]    0 0.1414212 0.1732050 0.5099020 0.5196152 0.5385164
#> [2,]    0 0.3000002 0.3316626 0.4242641 0.5099021 0.5385164
#> [3,]    0 0.2449490 0.2645751 0.3000002 0.4123107 0.5099019
#> [4,]    0 0.2449490 0.3316626 0.3316627 0.5000002 0.6480741
#> [5,]    0 0.1414212 0.2236066 0.4582576 0.5099019 0.6082762
#> [6,]    0 0.6164416 0.6164416 0.7000001 0.9949876 1.0862783
#> [7,]    0 0.2645751 0.3316627 0.4242642 0.4582576 0.5099021
#> [8,]    0 0.1732050 0.2236066 0.4123107 0.4242641 0.4242642
#> 
self_first_knn(graph)
#> $idx
#>      [,1] [,2] [,3] [,4] [,5] [,6]
#> [1,]    1    5    8    3    7    2
#> [2,]    2    3    4    8    7    1
#> [3,]    3    4    7    2    8    5
#> [4,]    4    3    2    7    8    5
#> [5,]    5    1    8    7    3    2
#> [6,]    6    5    1    8    7    3
#> [7,]    7    3    4    8    5    2
#> [8,]    8    1    5    3    2    7
#> 
#> $dist
#>      [,1]      [,2]      [,3]      [,4]      [,5]      [,6]
#> [1,]    0 0.1414212 0.1732050 0.5099020 0.5196152 0.5385164
#> [2,]    0 0.3000002 0.3316626 0.4242641 0.5099021 0.5385164
#> [3,]    0 0.2449490 0.2645751 0.3000002 0.4123107 0.5099019
#> [4,]    0 0.2449490 0.3316626 0.3316627 0.5000002 0.6480741
#> [5,]    0 0.1414212 0.2236066 0.4582576 0.5099019 0.6082762
#> [6,]    0 0.6164416 0.6164416 0.7000001 0.9949876 1.0862783
#> [7,]    0 0.2645751 0.3316627 0.4242642 0.4582576 0.5099021
#> [8,]    0 0.1732050 0.2236066 0.4123107 0.4242641 0.4242642
#> 
```
