# A sparse random projection for forest initialization

Fair warning, this article ends in defeat. But I had the idea, and I
tried the idea, and I am now writing this down so I don’t forget and try
it again another day.

The [tree-count
article](https://jlmelville.github.io/rnndescent/articles/tree-count.md)
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
article](https://jlmelville.github.io/rnndescent/articles/tree-count.md),
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
[`rnnd_build()`](https://jlmelville.github.io/rnndescent/reference/rnnd_build.md)
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

The comparisons use Euclidean distance and eight threads on an Apple M2.
The tables report medians over five seeds (42–46). Build timing runs
through preparation of the search index and stops before serialization.
Graph recall measures the overlap with exact neighbors from brute force
search, counting 15 nonself neighbors per point in this first
comparison.

| Dataset     | Forest              | Build median (s) | Graph recall (%) |
|:------------|:--------------------|-----------------:|-----------------:|
| macosko2015 | Ordinary, 20 trees  |            27.21 |            46.44 |
| macosko2015 | Ordinary, 5 trees   |            20.82 |            43.20 |
| macosko2015 | Projected, 20 trees |            16.78 |            43.38 |
| MNIST       | Ordinary, 21 trees  |             5.87 |            98.87 |
| MNIST       | Ordinary, 5 trees   |             4.73 |            98.59 |
| MNIST       | Projected, 21 trees |             4.85 |            98.61 |
| Gaussian    | Ordinary, 15 trees  |             1.16 |            38.03 |
| Gaussian    | Ordinary, 5 trees   |             1.09 |            36.73 |
| Gaussian    | Projected, 15 trees |             1.07 |            37.27 |
| COIL20      | Ordinary, 11 trees  |             1.31 |            99.99 |
| COIL20      | Ordinary, 5 trees   |             1.25 |            99.89 |
| COIL20      | Projected, 11 trees |             0.95 |            99.94 |

SRP saves the most time compared with a 5-tree forest on macosko2015 and
COIL20. On MNIST and Gaussian the two build times are close. The 5-tree
forest gives much the same recall on all four datasets and saves you the
hassle of managing the sparse projection matrix.

## Saved index size and projection regeneration

Is there any advantage in index size at least? The projected query tree
splits in 128 dimensions, compared with 3,000 for macosko2015, for
example. The model also needs the sparse projection matrix to regenerate
the projected data.

At k = 15, these are the median gzip-compressed RDS sizes in MiB for
complete models:

| Dataset | Ordinary, default trees | Ordinary, 5 trees | Projected, default trees | Regenerate Z (s) |
|:---|---:|---:|---:|---:|
| macosko2015 | 33.45 | 33.46 | 31.38 | 0.258 |
| MNIST | 38.55 | 38.54 | 39.14 | 0.127 |
| Gaussian | 80.07 | 80.04 | 75.90 | 0.013 |
| COIL20 | 24.07 | 24.04 | 20.62 | 0.019 |

The projected model saves about 2–4 MiB on macosko2015, Gaussian, and
COIL20. On MNIST it takes about 0.6 MiB more. Every model stores the
original data, the neighbor and search graphs, and one query tree. That
last point is why reducing the number of trees used during construction
barely changes the saved size of the ordinary models.

The regeneration column times rebuilding `Z` from the loaded model. The
median is at most about a quarter of a second on these datasets.

## Building 50 neighbors

| Dataset     | Forest              | Build median (s) | Graph recall (%) |
|:------------|:--------------------|-----------------:|-----------------:|
| macosko2015 | Ordinary, 20 trees  |            66.76 |           77.952 |
| macosko2015 | Ordinary, 5 trees   |            66.92 |           76.736 |
| macosko2015 | Projected, 20 trees |            60.24 |           77.235 |
| MNIST       | Ordinary, 21 trees  |            18.11 |           99.934 |
| MNIST       | Ordinary, 5 trees   |            18.81 |           99.920 |
| MNIST       | Projected, 21 trees |            18.77 |           99.924 |
| Gaussian    | Ordinary, 15 trees  |             4.19 |           83.451 |
| Gaussian    | Ordinary, 5 trees   |             4.31 |           83.163 |
| Gaussian    | Projected, 15 trees |             4.36 |           84.625 |
| COIL20      | Ordinary, 11 trees  |             4.63 |           99.996 |
| COIL20      | Ordinary, 5 trees   |             4.98 |           99.992 |
| COIL20      | Projected, 11 trees |             4.51 |           99.997 |

At 50 neighbors the larger ordinary forest gives higher recall than the
5-tree forest on all four datasets, with similar or shorter median build
times. At the same tree count, projection saves about 6.5 seconds on
macosko2015. The other three datasets are within a second either way.
Projection changes recall the most on the Gaussian data, where it gains
about 1.2 percentage points.

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

Both columns use a forest built on `Z`. The timings for descent on `Z`
include making the projection and rescoring in the original space.

| Dataset     |   k | Descent on X (s / recall %) | Descent on Z (s / recall %) |
|:------------|----:|----------------------------:|----------------------------:|
| macosko2015 |  15 |              16.78 / 43.377 |                3.58 / 7.792 |
| macosko2015 |  50 |              60.24 / 77.235 |               8.45 / 11.049 |
| MNIST       |  15 |               4.85 / 98.612 |               3.10 / 65.213 |
| MNIST       |  50 |              18.77 / 99.924 |               8.12 / 67.690 |
| Gaussian    |  15 |               1.07 / 37.267 |                0.45 / 1.788 |
| Gaussian    |  50 |               4.36 / 84.625 |                1.19 / 4.038 |
| COIL20      |  15 |               0.95 / 99.940 |               0.30 / 86.519 |
| COIL20      |  50 |               4.51 / 99.997 |               0.47 / 83.331 |

Oh dear. Descent on `Z` makes the build faster in every case. Rescoring
can only reorder the candidates that descent found in `Z`, so the recall
loss remains.

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
