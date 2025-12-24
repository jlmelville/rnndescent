# Merge multiple approximate nearest neighbors graphs

`merge_knn` takes a list of nearest neighbor graphs and merges them into
a single graph, with the same number of neighbors as the first graph.
This is useful to combine the results of multiple different nearest
neighbor searches: the output will be at least as accurate as the most
accurate of the two input graphs, and ideally will be more accurate than
either.

## Usage

``` r
merge_knn(graphs, is_query = FALSE, n_threads = 0, verbose = FALSE)
```

## Arguments

- graphs:

  A list of nearest neighbor graphs to merge. Each item in the list
  should consist of a sub-list containing:

  - `idx` an n by k matrix containing the k nearest neighbor indices.

  - `dist` an n by k matrix containing k nearest neighbor distances. The
    number of neighbors can differ between graphs, but the merged result
    will have the same number of neighbors as the first graph in the
    list.

- is_query:

  If `TRUE` then the graphs are treated as the result of a knn query,
  not a knn building process. Or: is the graph bipartite? This should be
  set to `TRUE` if `nn_graphs` are the results of using e.g.
  [`graph_knn_query()`](https://jlmelville.github.io/rnndescent/reference/graph_knn_query.md)
  or
  [`random_knn_query()`](https://jlmelville.github.io/rnndescent/reference/random_knn_query.md),
  and set to `FALSE` if these are the results of
  [`nnd_knn()`](https://jlmelville.github.io/rnndescent/reference/nnd_knn.md)
  or
  [`random_knn()`](https://jlmelville.github.io/rnndescent/reference/random_knn.md).
  The difference is that if `is_query = FALSE`, if an index `p` is found
  in `nn_graph1[i, ]`, i.e. `p` is a neighbor of `i` with distance `d`,
  then it is assumed that `i` is a neighbor of `p` with the same
  distance. If `is_query = TRUE`, then `i` and `p` are indexes into two
  different datasets and the symmetry does not hold. If you aren't sure
  what case applies to you, it's safe (but potentially inefficient) to
  set `is_query = TRUE`.

- n_threads:

  Number of threads to use.

- verbose:

  If `TRUE`, log information to the console.

## Value

a list containing:

- `idx` an n by k matrix containing the merged nearest neighbor indices.

- `dist` an n by k matrix containing the merged nearest neighbor
  distances.

The size of `k` in the output graph is the same as that of the first
item in `nn_graphs`.

## Examples

``` r
set.seed(1337)
# Nearest neighbor descent with 15 neighbors for iris three times,
# starting from a different random initialization each time
iris_rnn1 <- nnd_knn(iris, k = 15, n_iters = 1)
iris_rnn2 <- nnd_knn(iris, k = 15, n_iters = 1)
iris_rnn3 <- nnd_knn(iris, k = 15, n_iters = 1)

# Merged results should be an improvement over individual results
iris_mnn <- merge_knn(list(iris_rnn1, iris_rnn2, iris_rnn3))
sum(iris_mnn$dist) < sum(iris_rnn1$dist)
#> [1] TRUE
sum(iris_mnn$dist) < sum(iris_rnn2$dist)
#> [1] TRUE
sum(iris_mnn$dist) < sum(iris_rnn3$dist)
#> [1] TRUE
```
