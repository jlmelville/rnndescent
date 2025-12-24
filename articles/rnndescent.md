# rnndescent

`rnndescent` is an R package for finding approximate nearest neighbors,
based heavily on the Python package
[PyNNDescent](https://github.com/lmcinnes/pynndescent) by [Leland
McInnes](https://github.com/lmcinnes), but is a fully independent
reimplementation written in C++. It uses the following techniques:

1.  Initialization by creating a forest of random project trees
    (Dasgupta and Freund 2008).
2.  Optimization by using nearest neighbor descent (Dong, Moses, and Li
    2011).
3.  For building a search graph, graph diversification techniques from
    FANNG (Harwood and Drummond 2016).
4.  For querying new data, the back-tracking search from NGT (Iwasaki
    and Miyazaki 2018) (without dynamic degree-adjustment).

The easiest way to find k-nearest neighbors and query new data is to use
the `rnnd_knn` function, which combine several of the available
techniques into sensible defaults use the `rnnd_build` and `rnnd_query`
functions. For greater flexibility, the underlying functions used by
`rnnd_build` and `rnnd_query` can be used directly. The other vignettes
in this package describe their use and go into more detail about the how
the methods work.

``` r
library(rnndescent)
```

## Find the k-nearest neighbors

If you just want the k-nearest neighbors of some data, use `rnnd_knn`:

``` r
iris_knn <- rnnd_knn(data = iris, k = 5)
```

### The Neighbor Graph Format

The nearest neighbor graph format returned by all functions in this
package is a list of two matrices:

- `idx` – a matrix of indices of the nearest neighbors. As usual in R,
  these are 1-indexed.
- `dist` – the equivalent distances.

``` r
lapply(iris_knn, function(x) {
  head(x, 3)
})
#> $idx
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1   18   29    5   28
#> [2,]    2   13   46   35   10
#> [3,]    3   48    4    7   13
#> 
#> $dist
#>      [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,]    0 0.1000000 0.1414212 0.1414212 0.1414213
#> [2,]    0 0.1414213 0.1414213 0.1414213 0.1732050
#> [3,]    0 0.1414213 0.2449490 0.2645751 0.2645753
```

## Build an Index

`rnnd_knn` returns the k-nearest neighbors, but does not return any
“index” that you can use to query new data. To do that, use
`rnnd_build`. Normally you would query the index with different from
that which you used to build the index, so let’s split `iris` up:

``` r
iris_even <- iris[seq_len(nrow(iris)) %% 2 == 0, ]
iris_odd <- iris[seq_len(nrow(iris)) %% 2 == 1, ]
```

``` r
iris_index <- rnnd_build(iris_even, k = 5)
```

The index is also a list but with a lot more components (none of which
are intended for manual examination), apart from the the neighbor graph
which can be found under the `graph` component in the same format as the
return value of `rnnd_knn`:

``` r
lapply(iris_index$graph, function(x) {
  head(x, 3)
})
#> $idx
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1   23    5   13   18
#> [2,]    2   24   15   23    5
#> [3,]    3   10   11   17   14
#> 
#> $dist
#>      [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,]    0 0.1414213 0.1732050 0.2236068 0.3000000
#> [2,]    0 0.1414215 0.1732051 0.2645753 0.3162279
#> [3,]    0 0.3872986 0.4123107 0.4795830 0.5291505
```

Be aware that for large and high-dimensional data, the returned index
can get **very** large, especially if you set `n_search_trees` to a
large value.

## Querying Data

To query new data, use `rnnd_query`:

``` r
iris_odd_nn <- rnnd_query(
  index = iris_index,
  query = iris_odd,
  k = 5
)
lapply(iris_odd_nn, function(x) {
  head(x, 3)
})
#> $idx
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    9   14   20    4   25
#> [2,]   24    2   23   15    1
#> [3,]   19    9    4   20   14
#> 
#> $dist
#>           [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,] 0.1000000 0.1414213 0.1414213 0.1732050 0.2236068
#> [2,] 0.1414213 0.2449490 0.2645753 0.3000001 0.3000002
#> [3,] 0.1414213 0.1732050 0.2236066 0.2449488 0.2449488
```

You don’t need to keep the data that was used to build the index around,
because internally, the index stores that (that’s one of the reasons the
index can get large).

Another use for `rnnd_query` is to improve the quality of a k-nearest
neighbor graph. We are using for a `query` the same data we used to
build `iris_index` and specifying via the `init` parameter the knn graph
we already generated:

``` r
iris_knn_improved <- rnnd_query(
  index = iris_index,
  query = iris_even,
  init = iris_index$graph,
  k = 5
)
```

If the k-nearest neighbor graph in `index$graph` isn’t sufficiently high
quality, then result of running `rnnd_query` on the same data should be
an improvement. Exactly how much better is hard to say, but you can
always compare the sum of the distances:

``` r
c(
  sum(iris_index$graph$dist),
  sum(iris_knn_improved$dist)
)
#> [1] 124.3317 124.3317
```

In this case, the initial knn has not been improved, which is hardly
surprising due to the size of the dataset. Another function that might
be of use is the `neighbor_overlap` function to see how many neighbors
are shared between the two graphs:

``` r
neighbor_overlap(iris_index$graph, iris_knn_improved)
#> [1] 1
```

As there was no change to the graph, the overlap is 100%. More details
on this can be found in the
[hubness](https://jlmelville.github.io/rnndescent/articles/hubness.md)
vignette and a more ambitious dataset is covered in the [FMNIST
article](https://jlmelville.github.io/rnndescent/articles/fmnist-example.html).

## Parallelism

`rnndescent` is multi-threaded, but by default is single-threaded. Set
`n_threads` to set the number of threads you want to use:

``` r
iris_index <- rnnd_build(data = iris_even, k = 5, n_threads = 2)
```

## Available Metrics

Several different distances are available in `rnndescent` beyond the
typically-supported Euclidean and Cosine-based distances in other
nearest neighbor packages. See the
[metrics](https://jlmelville.github.io/rnndescent/articles/metrics.md)
vignette for more details.

## Supported Data Types

- Dense matrices and data frames.
- Sparse matrices, in the `dgCMatrix`. All the same distances are
  supported as for dense matrices.
- Additionally, for dense binary data, if you supply it as a `logical`
  matrix, then for certain distances intended for binary data,
  specialized functions will be used to speed up the computation.

## Parameters

There are several options that `rnnd_build` and `rnnd_query` expose that
can be modified to change the behavior of the different stages of the
algorithm. See the documentation for those functions
(e.g. [`?rnnd_build`](https://jlmelville.github.io/rnndescent/reference/rnnd_build.md))
or the [Random Partition
Forests](https://jlmelville.github.io/rnndescent/articles/random-partition-forests.md),
[Nearest Neighbor
Descent](https://jlmelville.github.io/rnndescent/articles/nearest-neighbor-descent.md)
and [Querying
Data](https://jlmelville.github.io/rnndescent/articles/querying-data.md)
vignettes for more details.

## References

Dasgupta, Sanjoy, and Yoav Freund. 2008. “Random Projection Trees and
Low Dimensional Manifolds.” In *Proceedings of the Fortieth Annual ACM
Symposium on Theory of Computing*, 537–46.

Dong, Wei, Charikar Moses, and Kai Li. 2011. “Efficient k-Nearest
Neighbor Graph Construction for Generic Similarity Measures.” In
*Proceedings of the 20th International Conference on World Wide Web*,
577–86.

Harwood, Ben, and Tom Drummond. 2016. “Fanng: Fast Approximate Nearest
Neighbour Graphs.” In *Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition*, 5713–22.

Iwasaki, Masajiro, and Daisuke Miyazaki. 2018. “Optimization of Indexing
Based on k-Nearest Neighbor Graph for Proximity Search in
High-Dimensional Data.” *arXiv Preprint arXiv:1810.07355*.
