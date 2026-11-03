# Overlap between the indices of two nearest neighbor graphs

Calculates the mean average number of neighbors in common between the
two graphs. The per-item overlap can also be returned. This function can
be useful as a measure of accuracy of approximation algorithms, if the
exact nearest neighbors are known, or as a measure of diversity of two
different approximate graphs.

## Usage

``` r
neighbor_overlap(idx1, idx2, k = NULL, ret_vec = FALSE)
```

## Arguments

- idx1:

  Indices of a nearest neighbor graph, i.e. a matrix of nearest neighbor
  indices. Can also be a list containing an `idx` element.

- idx2:

  Indices of a nearest neighbor graph, i.e. a matrix of nearest neighbor
  indices. Can also be a list containing an `idx` element. This is
  considered to be the ground truth.

- k:

  Number of neighbors to consider. If `NULL`, then the minimum of the
  number of neighbors in `idx1` and `idx2` is used.

- ret_vec:

  If `TRUE`, also return a vector containing the per-item overlap.

## Value

The mean overlap between `idx1` and `idx2`. If `ret_vec = TRUE`, then a
list containing the mean overlap and the overlap of each item in is
returned with names `mean` and `overlaps`, respectively.

## Details

The graph format is the same as that returned by e.g.
[`nnd_knn()`](https://jlmelville.github.io/rnndescent/reference/nnd_knn.md)
and should be of dimensions n by k, where n is the number of points and
k is the number of neighbors. If you pass a neighbor graph directly, the
index matrix will be extracted if present. If the two graphs have
different numbers of neighbors, then the smaller number of neighbors is
used.

## Examples

``` r
set.seed(1337)
# Generate two random neighbor graphs for iris
iris_rnn1 <- random_knn(iris, k = 15)
iris_rnn2 <- random_knn(iris, k = 15)

# Overlap between the two graphs
mean_overlap <- neighbor_overlap(iris_rnn1, iris_rnn2)

# Also get a vector of per-item overlap
overlap_res <- neighbor_overlap(iris_rnn1, iris_rnn2, ret_vec = TRUE)
summary(overlap_res$overlaps)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#>  0.0000  0.1333  0.2000  0.1871  0.2667  0.4000 
```
