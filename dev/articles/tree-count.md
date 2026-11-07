# How many trees should I use?

The RP forest initialization is intended to give NN-descent a starting
set of neighbors that’s better than random.

By default,
[`rnnd_build()`](https://jlmelville.github.io/rnndescent/dev/reference/rnnd_build.md)
chooses the number of initialization trees using the same formula as
PyNNDescent, based on the number of observations, n:

\text{trees} = \min\\\left(32,\\ 5 +
\operatorname{round}\\\left(n^{1/4}\right)\right).

To give a sense of the range of `n_trees`, with 1,000 observations, the
formula gives 11 trees, 10,000 gives 15 trees, and 100,000 gives 23
trees.

In this article, we will look to see how varying the number of trees in
the forest affects the performance of the nearest neighbor search.
Presumably, more trees is better in terms of finding the k-nearest
neighbors, but you’d think there must be a trade-off between when
diminishing returns kick in between spending time building more trees,
versus the nearest neighbor search starting from a less optimal
initialization. Does building more trees save enough work during descent
to be worth the time?

## The comparison

Random initialization and forests of 2, 5, 10, 20 and 40 trees, followed
by NN-descent, using Euclidean distance. Times and final recall are
medians of three seeds on an Apple M2 with 8 GiB memory and eight
threads. Recall measures reference neighbors found. Neighbor counts and
recall exclude each observation itself. The benchmark requested one
extra position and assumed self was the first neighbor in each graph.
The example below removes self by index. Each timing column has its own
median, so the components need not add to the displayed total.

Times cover initialization and descent, excluding searchable index
preparation and queries. Nearby runtime ranges often overlap.

## macosko2015: 15 neighbors

The
[macosko2015](https://jlmelville.github.io/uwot/articles/umap-examples.html#macosko2015)
data set contains single cell RNA sequencing data with 44,808
observations (cells) and 3,000 features (gene expression values).
Keeping 3,000 genes in these bioinformatics workflows is a reasonable
choice, but usually they are further reduced by PCA to 50-100 dimensions
before a nearest neighbor search. I like keeping all the features as an
example of a non-synthetic high-dimensional dataset that is challenging
to get good results from using approximate nearest neighbor methods.

Here are the results for 15 neighbors (the default `n_trees` would be
20):

| Trees  | Initialization (s) | NND (s) | Total (s) | Final recall (%) |
|:-------|-------------------:|--------:|----------:|-----------------:|
| Random |               2.08 |    22.7 |      24.7 |            39.76 |
| 2      |               5.78 |    25.2 |      30.9 |            40.87 |
| 5      |               6.42 |    21.4 |      27.8 |            43.19 |
| 10     |               11.9 |    19.3 |      30.4 |            44.70 |
| 20     |               18.6 |    16.9 |      35.5 |            46.51 |
| 40     |               32.8 |    15.4 |      48.2 |            48.97 |

Random initialization is fast, but results in the lowest recall, 40%.
Not great! Adding trees helps, but maybe not as much as you’d hope. Note
that the NND time does go down due to the better starting point, but the
time taken for the entire search begins to be dominated by the forest
construction.

## macosko2015: 50 neighbors

What happens if we need a larger neighbor graph?

| Trees  | Initialization (s) | NND (s) | Total (s) | Final recall (%) |
|:-------|-------------------:|--------:|----------:|-----------------:|
| Random |               2.99 |   168.1 |     171.1 |            76.05 |
| 2      |               5.91 |   177.9 |     183.4 |            75.84 |
| 5      |               7.47 |   155.5 |     162.9 |            76.74 |
| 10     |               13.5 |   146.2 |     159.6 |            77.27 |
| 20     |               21.9 |   137.2 |     158.2 |            77.98 |
| 40     |               38.6 |   124.9 |     164.1 |            79.05 |

The recall itself is improved at least, but now the improvement from
initializing from the forest is even less impressive. That said, at
least on my machine, even going up to 40 trees gives a faster run time
than using random initialization, although the sweet spot is around 20
trees.

## Other data

At 15 neighbors, compare five and 20 trees on several datasets: the
[MNIST digits](https://github.com/fgnt/mnist),
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist),
[COIL20](https://www.cs.columbia.edu/CAVE/software/coil-20-proc),
[Mammoth](https://github.com/PAIR-code/understanding-umap), and a
synthetic Gaussian dataset with 10,000 observations and either 5 or
1,000 features.

Shape is observations × features; times and recalls are medians over
three runs.

| Data          | Shape          | 5 trees  |            | 20 trees |            |
|---------------|----------------|----------|------------|----------|------------|
|               |                | Time (s) | Recall (%) | Time (s) | Recall (%) |
| MNIST         | 70,000 × 784   | 10.4     | 98.57      | 8.53     | 98.84      |
| Fashion-MNIST | 70,000 × 784   | 10.4     | 98.44      | 8.93     | 98.70      |
| COIL20        | 1,440 × 16,384 | 1.87     | 99.94      | 2.44     | 99.99      |
| Mammoth       | 10,000 × 3     | 0.163    | 99.99      | 0.106    | 100.00     |
| Gaussian      | 10,000 × 5     | 0.171    | 99.95      | 0.105    | 99.98      |
| Gaussian      | 10,000 × 1,000 | 1.55     | 36.68      | 1.73     | 38.57      |

As you can see, the recall is very good for all the datasets, except for
the high-dimensional Gaussian dataset and more trees is always better.
But it’s not just number of features that matters for overall recall,
because COIL-20 is high-dimensional but has good recall (but is much
smaller than the high-dimensional Gaussian or `macosko2015` datasets).
Although it’s not a massive difference, COIL-20 and the high-dimensional
Gaussian datasets also run more slowly overall when going to 20 trees.

## Comparing the same neighbors

One way to improve performance is to use more trees, but as the [hubness
article](https://jlmelville.github.io/rnndescent/articles/hubness.html)
notes, another way is to ask for more neighbors than you need, then cut
down to just the ones you need.

Below are results for the two difficult datasets from the previous
section (`macosko2015` and the high-dimensional Gaussian dataset), using
five and 20 trees, asking for 50 neighbors, then cutting down to 15
neighbors. We know from the previous section that asking for 50
neighbors requires substantially more time than asking for 15 neighbors,
so it’s reasonable to ask what if we stick with 15 neighbors but build a
*much* larger forest? So that result is also included below.

| Trees | 15 neighbors |  | 50 neighbors |  |
|----|----|----|----|----|
|  | Time (s) | Recall@15 (%) | Time (s) | Recall@15 (%) |
| macosko2015 |  |  |  |  |
| 5 | 27.8 | 43.19 | 162.9 | 82.55 |
| 20 | 35.5 | 46.51 | 158.2 | 83.71 |
| 200 | 172.7 | 57.93 | — | — |
| Gaussian, 1,000 features |  |  |  |  |
| 5 | 1.55 | 36.68 | 9.43 | 87.01 |
| 20 | 1.73 | 38.57 | 8.81 | 87.64 |
| 300 | 9.14 | 65.12 | — | — |

Building 50 neighbors substantially improves recall@15 for both
datasets, but at roughly four to six times the construction time with
five or 20 trees. Adding more trees at 15 neighbors also helps, but at a
similar runtime the 50-neighbor graph still has much higher recall.
Runtime isn’t everything, because RAM usage is also a factor, but in
terms of adjusting parameters, for high-dimensional datasets, increasing
`k` is probably better than increasing `n_trees`.

## Trying your own data

Sample code used to generate the sort of results shown above.

``` r

library(rnndescent)
reference <- brute_force_knn(X, k = 16, n_threads = 2)

set.seed(42)
tree_time <- system.time(
  graph <- nnd_knn(
    X,
    k = 16,
    metric = "euclidean",
    init = "tree",
    init_args = list(n_trees = 10),
    n_threads = 2
  )
)
set.seed(42)
random_time <- system.time(
  random_graph <- nnd_knn(
    X,
    k = 16,
    metric = "euclidean",
    init = "rand",
    n_threads = 2
  )
)

tree_time["elapsed"]
random_time["elapsed"]
100 *
  neighbor_overlap(
    remove_self_knn(graph),
    remove_self_knn(reference)
  )
100 *
  neighbor_overlap(
    remove_self_knn(random_graph),
    remove_self_knn(reference)
  )
```

To reuse a larger result with fewer neighbors, keep its first columns
with
[`truncate_knn()`](https://jlmelville.github.io/rnndescent/dev/reference/knn-utilities.md).

``` r

graph_10 <- truncate_knn(graph, k = 10)
```

[`rnnd_build()`](https://jlmelville.github.io/rnndescent/dev/reference/rnnd_build.md)
also accepts `n_trees` when you need a searchable index.
