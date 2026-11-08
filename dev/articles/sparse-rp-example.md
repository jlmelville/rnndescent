# A sparse random projection for forest initialization

Fair warning, this article ends in defeat. But I had the idea, and I
tried the idea, and I am now writing this down so I don’t forget and try
it again another day.

The [tree-count
article](https://jlmelville.github.io/rnndescent/dev/articles/tree-count.md)
explores the trade-off between forest time and recall. High-dimensional
data takes a noticeably longer amount of time to build and at least at
low `k`, recall from a forest can be underwhelming, although it *is*
better than initializing nearest neighbor descent with the projected
data than using random initialization.

So could we make the forest cheaper by pre-processing the input data to
a lower dimensional space? PCA is the obvious choice, but even
approaches based on truncated SVD (like irlba) can be slower than
building the forest in the original space. A much faster approach is to
use random projections.

The intriguing idea originating with [Johnson and Lindenstrauss
(1984)](#references) is that a random projection into a lower
dimensional space approximately preserves pairwise distances, depending
on the number of dimensions. This projection can be done very quickly
and further work by [Achlioptas (2003)](#references) shows you can do
the projection with a matrix containing only +1, -1 and 0 as entries.
Even better, [Li, Hastie and Church (2006)](#references) show you can
make this matrix very sparse, so storing the projection matrix can be
very cheap.

In this article, we use a very sparse random projection (SRP) to map the
whole dataset down to 128 dimensions, then build an ordinary random
projection forest on the projected data. This global projection is
separate from the randomized splits made at individual tree nodes. As
you will see the results are probably not worth the extra effort of
implementation, mainly that it’s an extra complication in the projection
forest storage. And technically this only makes sense for Euclidean
distances, so this would be a lot of extra machinery that only works for
one metric (albeit a very common one).

## Data and setup

The comparisons use
[macosko2015](https://jlmelville.github.io/uwot/articles/umap-examples.html#macosko2015)
mouse retina single-cell RNA sequencing data (3,000 gene expression
features), [MNIST](https://github.com/fgnt/mnist) handwritten digits,
and [COIL20](https://www.cs.columbia.edu/CAVE/software/coil-20-proc)
object images.

| Dataset     | Observations | Features |
|:------------|-------------:|---------:|
| macosko2015 |       44,808 |    3,000 |
| MNIST       |       70,000 |      784 |
| COIL20      |        1,440 |   16,384 |
| Gaussian    |       10,000 |    1,000 |

For the recipe, use the 1,000-dimensional independent standard normal
Gaussian data from the [tree-count
article](https://jlmelville.github.io/rnndescent/dev/articles/tree-count.md),
a difficult neighbor-search example; `X` contains all reference rows,
and separately generated query rows `Q` represent future observations
with the same features in the same order.

``` r

library(rnndescent)
library(Matrix)
set.seed(20261105)
X <- matrix(rnorm(10000 * 1000), nrow = 10000)
set.seed(907)
Q <- matrix(rnorm(256 * 1000), nrow = 256)
k <- 15L
build_k <- k + 1L # Allow for self in the reference graph.
threads <- 8L
```

## Project and build

Pass candidate indices alone so that
[`rnnd_build()`](https://jlmelville.github.io/rnndescent/dev/reference/rnnd_build.md)
calculates Euclidean distances using the original features.

Following [Li, Hastie and Church (2006)](#references), the map uses
nonzero probability \rho = 1/\sqrt{d}, equally likely positive and
negative (Rademacher) signs, and scale 1/\sqrt{m\rho}, where d is the
original feature count and m the projected feature count.

``` r

sparse_sign_map <- function(d, m) {
  rho <- 1 / sqrt(d)
  Matrix::rsparsematrix(
    d,
    m,
    nnz = rbinom(1, d * m, rho),
    rand.x = function(n) sample(c(-1, 1), n, replace = TRUE) / sqrt(m * rho)
  )
}
set.seed(42)
P <- sparse_sign_map(ncol(X), 128L)
Z <- as.matrix(X %*% P)
rp <- rpf_knn(
  Z,
  k = build_k,
  include_self = FALSE,
  ret_forest = TRUE,
  n_threads = threads
)
index <- rnnd_build(
  X,
  k = build_k,
  metric = "euclidean",
  init = rp$idx,
  n_threads = threads
)
forest <- rpf_filter(
  nn = index$graph,
  forest = rp$forest,
  n_trees = 1L,
  n_threads = threads
)
model <- list(
  index = index,
  map = P,
  forest = forest
)
saveRDS(model, "sparse-rp-index.rds")
```

## Query

Because querying only uses one tree, the effect of the sparse projection
is minimized, so I won’t present any results for query performance in
this article, but the code is here for completeness.

``` r

model <- readRDS("sparse-rp-index.rds")
Z <- as.matrix(crossprod(model$index$data, model$map))
Zq <- as.matrix(Q %*% model$map)
starts <- rpf_knn_query(
  Zq,
  Z,
  model$forest,
  k = k,
  n_threads = threads
)
neighbors <- rnnd_query(
  model$index,
  Q,
  k = k,
  init = starts$idx,
  n_threads = threads
)
head(cbind(row_in_X = neighbors$idx[1, ], distance = neighbors$dist[1, ]))
```

## A small comparison

These results use Euclidean distance and eight threads. Build times are
medians over five seeds and include index preparation, but exclude
serialization. The recall column compares nonself neighbors with saved
reference neighbors calculated from brute force search.

At k = 15:

| Dataset     | Forest              | Build median (s) | Graph recall (%) |
|:------------|:--------------------|-----------------:|-----------------:|
| macosko2015 | Ordinary, 20 trees  |            63.05 |            46.50 |
| macosko2015 | Ordinary, 5 trees   |            47.64 |            43.10 |
| macosko2015 | Projected, 20 trees |            38.51 |            43.46 |
| MNIST       | Ordinary, 21 trees  |            12.42 |            98.86 |
| MNIST       | Ordinary, 5 trees   |            11.76 |            98.57 |
| MNIST       | Projected, 21 trees |             7.50 |            98.62 |
| Gaussian    | Ordinary, 15 trees  |             2.66 |            38.01 |
| Gaussian    | Ordinary, 5 trees   |             2.33 |            36.71 |
| Gaussian    | Projected, 15 trees |             2.27 |            37.23 |
| COIL20      | Ordinary, 11 trees  |             2.61 |            99.95 |
| COIL20      | Ordinary, 5 trees   |             2.09 |            99.90 |
| COIL20      | Projected, 11 trees |             1.64 |            99.94 |

Using the SRP data, the build time is even faster than a 5-tree forest
in the original space. But recall is only slightly higher than that
5-tree forest. So you could argue that instead of using SRP, you should
just use a 5-tree forest in the original space and save yourself the
hassle of managing the sparse projection matrix.

## Saved index size and projection regeneration

Is there any advantage in index size at least? We are only having to
save 128 dimensions to define the split at each node of the trees,
compared to 3,000 for e.g. macosko2015, and the only extra cost is the
storage of the sparse signed projection matrix. Its nonzero entries have
equal magnitude, so it should be very efficient to store.

At k = 15, these are the median gzip-compressed RDS sizes in MiB for
complete models:

| Dataset | Ordinary, default trees | Ordinary, 5 trees | Projected, default trees | Regenerate Z (s) |
|:---|---:|---:|---:|---:|
| macosko2015 | 38.74 | 38.65 | 31.38 | 0.253 |
| MNIST | 39.65 | 39.64 | 39.14 | 0.143 |
| Gaussian | 80.87 | 80.76 | 75.90 | 0.014 |
| COIL20 | 24.88 | 24.89 | 20.62 | 0.019 |

The SRP models are smaller, but the original reference data and the
neighbor and search graphs still need to be saved. Both methods retain
only one query tree, so reducing the number of trees used during
construction barely changes the saved size.

Because we don’t save the `Z` matrix, it needs to be regenerated. The
regeneration column shows how much time that takes (again a median of 5
runs just to be consistent). It is not a major bottleneck.

## Building 50 neighbors

| Dataset     | Forest              | Build median (s) | Graph recall (%) |
|:------------|:--------------------|-----------------:|-----------------:|
| macosko2015 | Ordinary, 20 trees  |           173.90 |           77.972 |
| macosko2015 | Ordinary, 5 trees   |           177.85 |           76.798 |
| macosko2015 | Projected, 20 trees |           164.28 |           77.202 |
| MNIST       | Ordinary, 21 trees  |            42.29 |           99.933 |
| MNIST       | Ordinary, 5 trees   |            48.83 |           99.921 |
| MNIST       | Projected, 21 trees |            40.92 |           99.924 |
| Gaussian    | Ordinary, 15 trees  |            10.83 |           83.387 |
| Gaussian    | Ordinary, 5 trees   |            11.01 |           83.165 |
| Gaussian    | Projected, 15 trees |            11.08 |           84.656 |
| COIL20      | Ordinary, 11 trees  |            12.48 |           99.994 |
| COIL20      | Ordinary, 5 trees   |            13.10 |           99.983 |
| COIL20      | Projected, 11 trees |            12.43 |           99.996 |

If we go up to 50 nearest neighbors, then using `n_trees = 5` is slower
than using the default larger number of trees. This is because the NND
from a worse initialization spends more time optimizing. For a fixed
number of trees, the time taken is not hugely affected and neither is
recall: sometimes a bit better, sometimes a bit worse.

## What if descent also uses the projection?

One last idea: it would at least be a bit less awkward to use SRP if it
was a preprocessing step: do the projection once and then do the nearest
neighbor search on that projected data, i.e. the RP forest and then
nearest neighbor descent. If you wanted the distances in the original
space, you can just calculate that at the end using the returned
indices. The recall goes down a bit in the tree-building step, but maybe
NND can recover it, and the distance calculation will be faster, so it
might be worth a try?

For example, build the graph on `Z`, then rescore its nonself neighbors
on `X` and order them by their original-space distances.

``` r

set.seed(42)
projected_index <- rnnd_build(
  Z,
  k = build_k,
  metric = "euclidean",
  init = "tree",
  n_trees = 15L,
  leaf_size = build_k,
  n_search_trees = 1L,
  n_threads = threads
)
ids <- remove_self_knn(projected_index$graph, k = k)$idx
original_neighbors <- sort_knn(
  knn_distances(X, ids, metric = "euclidean", n_threads = threads)
)
```

| Dataset | k | Projected forest only: s / recall (%) | Projected forest and descent: s / recall (%) |
|:---|---:|---:|---:|
| macosko2015 | 15 | 38.51 / 43.456 | 4.38 / 7.791 |
| macosko2015 | 50 | 164.28 / 77.202 | 11.98 / 11.050 |
| MNIST | 15 | 7.50 / 98.618 | 4.22 / 65.194 |
| MNIST | 50 | 40.92 / 99.924 | 13.37 / 67.690 |
| Gaussian | 15 | 2.27 / 37.229 | 0.56 / 1.774 |
| Gaussian | 50 | 11.08 / 84.656 | 1.82 / 4.030 |
| COIL20 | 15 | 1.64 / 99.940 | 0.28 / 86.528 |
| COIL20 | 50 | 12.43 / 99.996 | 0.59 / 83.332 |

Oh dear. It’s certainly much faster (and that includes recalculating the
distances in the original space and sorting them), but the recall is a
bit of a disaster in all cases.

So for high-dimensional data and a small number of neighbors, there
might be an interesting idea lurking in here somewhere. Maybe using more
than 128 dimensions would help? But I am not inclined to explore that
further.

## References

- Johnson, W. B. and Lindenstrauss, J. (1984). [Extensions of Lipschitz
  mappings into a Hilbert
  space](https://doi.org/10.1090/conm/026/737400). *Contemporary
  Mathematics*, 26, 189–206.
  [PDF](https://stanford.edu/class/cs114/readings/JL-Johnson.pdf)
- Achlioptas, D. (2003). [Database-friendly random projections:
  Johnson-Lindenstrauss with binary
  coins](https://doi.org/10.1016/S0022-0000(03)00025-4). *Journal of
  Computer and System Sciences*, 66(4), 671–687.
  [PDF](https://cgi.di.uoa.gr/~optas/papers/jl.pdf)
- Li, P., Hastie, T. J. and Church, K. W. (2006). [Very sparse random
  projections](https://doi.org/10.1145/1150402.1150436). *Proceedings of
  KDD ’06*, 287–296.
  [PDF](https://hastie.su.domains/Papers/Ping/KDD06_rp.pdf)
