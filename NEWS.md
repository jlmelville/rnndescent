# rnndescent (development version)

* New parameter: for `nnd_knn` and `rnnd_build`: `weight_by_degree`. If set
to `TRUE`, then the candidate list in nearest neighbor descent is weighted in
favor of low-degree items, which should make for a more diverse local join.
There is a minor increase in computation but also a minor increase in accuracy.
the knn graph with no index built. The index can be very large in size for high
dimensional or large datasets, so this function is useful if you only care about
the knn graph and won't ever want to query new data.
* New parameter for `rnnd_query` and `graph_knn_query`: `max_search_fraction`.
This parameter controls the maximum number of nodes that can be searched during
querying. If the number of nodes searched exceeds this fraction of the total
number of nodes in the graph, the search will be terminated. This can be
used in combination with `epsilon` to avoid excessive search times.

# rnndescent 0.1.3

CRAN resubmission to fix lingering UBSAN errors.

# rnndescent 0.1.2

## Bug fixes and minor improvements

* Some internal Address/Undefined Behavior Sanitizer fixes discovered by CRAN
checks on the 0.1.1 submission have been fixed.

# rnndescent 0.1.1

Initial CRAN submission.

# rnndescent 0.0.16

## Breaking changes

* `rnnd_build` now always prepares the search graph.
* The `rnnd_prepare` function has been removed. The option to not prepare
the search graph during index building only made sense if you were only
interested in the k-nearest neighbor graph. Now that `rnnd_knn` exists for
that purpose (see below), the logic of index building has been substantially
simplified.
* The `nn_to_sparse` function has been removed.
* The `merge_knn` function has been removed, and `merge_knnl` has been renamed
  to `merge_knn`. If you were running e.g. `merge_knn(nn1, nn2)`, you must now
  use `merge_knn(list(nn1, nn2))`. Also the parameter `nn_graphs` has been 
  renamed `graphs`.

## New features

* New function: `rnnd_knn`. Behaves a lot like `rnnd_build`, but *only* returns
the knn graph with no index built. The index can be very large in size for
high dimensional or large datasets, so this function is useful if you only
care about the knn graph and won't ever want to query new data.
* New function: `neighbor_overlap`. Measures the overlap of two knn graphs via
their shared indices. A similar function was used extensively in some vignettes
so it may have sufficient utility to be useful to others.

## Bug fixes and minor improvements

* The sparse `spearmanr` distance has been fixed.
* During tree-building with `n_threads = 0`, progress/interrupt monitoring was
not occurring.
* You can provide a user-defined graph to the `init` parameter of `rnnd_query`.
* `rnnd_query`: if `verbose = TRUE`, a summary of the min, max and average
number of distance queries will be logged. This can help tune `epsilon` and
`max_search_fraction`.

# rnndescent 0.0.15

## Breaking Changes

* Standalone distance functions have been removed. They hadn't expanded to 
match all the distances available in the nearest neighbor functions, nor was
sparse support added. Doing so would increase the size of this package's API
even further. They may show up in another package.
* The `local_scale_nn ` has been removed, for similar reasons to the removal
of the standalone distance functions. It remains in the `localscale` branch
of the github repo.
* The search graph returned from `prepare_search_graph` is now transposed. This
prevents having to repeatedly transpose inside every call to `graph_knn_query`
if multiple queries are being made. You will need to either regenerate any
saved search graphs or transpose them with `Matrix::t(search_graph)`.

## New features

* New functions: `rnnd_build`, `rnnd_query` and `rnnd_prepare`. These functions
streamline the process of building a k-nearest neighbor graph, preparing a
search graph and querying it.

# rnndescent 0.0.14

## Breaking Changes

* The `bhamming` metric no longer exists as a specialized metric. Instead, if
you pass a `logical` matrix to `data`, `reference` or `query` parameter
(depending on the function) and specify `metric = "hamming"` you will
automatically get the binary-specific version of the hamming metric.
* The `hamming` and `bhamming` metrics are now normalized with respect to the
number of features, to be consistent with the other binary-style metrics (and 
PyNNDescent). If you need the old distances, multiply the distance matrix by 
the number of columns, e.g. do something like:
    
    ```R
    res <- brute_force_knn(X, metric = "hamming")
    res$dist <- res$dist * ncol(X)
    ```

* The metric `l2sqr` has been renamed `sqeuclidean` to be consistent with 
PyNNDescent.

## New features

* Metrics? We got 'em! The `metric` parameter now accepts a much larger number
of metrics. See the rdoc for the full list of supported metrics. Currently, most
of the metrics from PyNNDescent which don't require extra parameters are
supported. The number of specialized binary metrics has also been expanded.
* New parameter for `rpf_knn` and `rpf_build`: `max_tree_depth` this controls
the depth of the tree and was set to 100 internally. This default has been 
doubled to 200 and can now be user-controlled. If `verbose = TRUE` and the 
largest leaf in the forest exceeds the `leaf_size` parameter, a message warning
you about this will be logged and indicates that the maximum tree depth has
been exceeded. Increasing `max_tree_depth` may not be the answer: it's more
likely there is something unusual about the distribution of the distances in
your dataset and a random initialization might be a better use of your time.

# rnndescent 0.0.13

## New features

* Sparse data is now supported. Pass a `dgCMatrix` to the `data`, `reference` or
`query` parameters where you would usually use a dense matrix or data frame.
`cosine`, `euclidean`, `manhattan`, `hamming` and `correlation` are all 
available, but alternative versions in the dense case, e.g. `cosine-preprocess`
or the  binary-specific `bhamming` for dense data is not.
* A new `init` option for `graph_knn_query`: you can now pass an RP forest and
initialize with that, e.g. from `rpf_build`, or by setting `ret_forest = TRUE`
on `nnd_knn` or `rpf_knn`. You may want to cut down the size of the forest
used for initialization with `rpf_filter` first, though (a single tree may be
enough). This will also use the metric data in the forest, so setting `metric`
(or `use_alt_metric`) in the function itself will be ignored.

## Bug fixes and minor improvements

* If the knn graph you pass to `prepare_search_graph` or to `graph_knn_query` 
contains missing data, this will no longer cause an error (it still might not be
the best idea though).

# rnndescent 0.0.12

## New features

* New function: `rpf_knn`. Calculates the approximate k-nearest neighbors using
a random partition forest.
* New function: `rpf_build`. Builds a random partition forest.
* New function: `rpf_knn_query`. Queries a random partition forest (built with
`rpf_build` to find the approximate nearest neighbors for the query points.
* New function: `rpf_filter`. Retains only the best "scoring" trees in a forest,
where each tree is scored based on how well it reproduces a given knn.
* New initialization method for `nnd_knn`: `init = "tree"`. Uses the RP Forest
initialization method.
* New parameter for `nnd_knn`: `ret_forest`. Returns the search forest used if
`init = "tree"` so it can be used for future searching or filtering.
* New parameter for `nnd_knn`: `init_opts`. Options that can be passed to the
RP forest initialization (same as in `rpf_knn`).

# rnndescent 0.0.11

## Bug fixes and minor improvements

* Progess report for `nnd_knn` with `n_threads > 0` was reporting double the
actual number of iterations. This made the progress bar way too optimistic.
* A bug with flagging neighbors in 0.0.10 made the nearest neighbor descent
inefficient.

# rnndescent 0.0.10

## New features

* A change to `metric`: `"cosine"` and `"correlation"` have been renamed
`"cosine-preprocess"` and `"correlation-preprocess"` respectively. This
reflects that they do some preprocessing of the data up front to make
subsequent distance calculations faster. I have endeavored to avoid unnecessary
allocations or copying in this preprocessing, but there is still a chance of
more memory usage.
* The `cosine` and `correlation` metrics are still available as an option, but 
now use an implementation that doesn't do any preprocessing. The preprocessing
and non-preprocessing version should give the same numerical results, give or
take some minor numerical differences, but when the distance should be zero,
the preprocessing versions may give values which are slightly different from 
zero (e.g. 1e-7).
* New functions: `correlation_distance`, `cosine_distance`,
`euclidean_distance`, `hamming_distance`, `l2sqr_distance`, `manhattan_distance`
for calculating the distance between two vectors, which may be useful for 
more arbitrary distance calculations than the nearest neighbor routines here,
although they won't be as efficient (they do call the same C++ code, though).
The cosine and correlation calculations here use the non-preprocessing
implementations.
* Generalize `hamming` metric to a standard definition. The old implementation
of `hamming` metric which worked on binary data only was renamed into
`bhamming`. (contributed by [Vitalie Spinu](https://github.com/vspinu))
* New parameter `obs` has been added to most functions: set `obs = "C"` and you
can pass the input data in column-oriented format.

## Bug fixes and minor improvements

* The `random_knn` function used to always return each item as its own neighbor,
so that only `n_nbrs - 1` of the returned neighbors were actually selected at
random. Even I forgot it did that and it doesn't make a lot of sense, so now you
really do just get back `n_nbrs` random selections.
* If providing pre-calculated neighbors as the `init` parameter to `nnd_knn` or
`graph_knn_query`: previously, if `k` was specified and larger than the number
of neighbors included in `init`, this gave an error. Now, `init` will be
augmented with random neighbors to reach the desired `k`. This could be useful
as a way to "restart" a neighbor search from a better-than-random location if
`k` has been found to have been too small initially. Note that the random
selection does not take into account the identities of the already chosen
neighbors, so duplicates may be included in the augmented result, which will
reduce the effective size of the initialized number of neighbors.
* Removed the `block_size` and `grain_size` parameters from functions. These
were related to the amount of work done per thread, but it's not obvious to
an outside user how to set these.
* Most long-running computations should update any progress indicators more
frequently (if `verbose = TRUE`) and respond to user-requested cancellation.

# rnndescent 0.0.9 (20 June 2021)

## New features

* `nnd_knn_query` has been renamed to `graph_knn_query` and now more closely
follows the current pynndescent graph search method (including backtracking
search).
* New function: `prepare_search_graph` for preparing a search graph from a
neighbor graph for use in `graph_knn_query`, by using reverse nearest neighbors,
occlusion pruning and truncation.
* Sparse graphs are supported as input to `graph_knn_query`.

# rnndescent 0.0.8 (10 October 2020)

There was a major rewrite of the internal organization of the C++ to be less
R-specific.

## Minor License Change

The license for rnndescent has changed from GPLv3 to GPLv3 or later.

## New features

* New metric: `"correlation"`. This is (1 minus) the Pearson correlation.
* New function: `k_occur` which counts the k-occurrences of each item in the
`idx` matrix, which is the number of times an item appears in the k-nearest
neighbor list in the dataset. The distribution of the k-occurrences can be
used to diagnose the "hubness" of a dataset. Items with a large k-occurrence
(>> k, e.g. 10k), may indicate low accuracy of the approximate nearest neighbor
result.

# rnndescent 0.0.7 (1 March 2020)

## Bug fixes and minor improvements

To avoid undefined behavior issues, rnndescent now uses an internal
implementation of RcppParallel's `parallelFor` loop that works with
`std::thread` and does not load Intel's TBB library.

# rnndescent 0.0.6 (29 November 2019)

## Bug fixes and minor improvements

* For some reason, I thought it would be ok to use the `dqrng` sample routines
from inside a thread, despite it clearly using the R API extensively. It's not
ok and causes lots of crashes. There is now a re-implementation of `dqrng`'s 
sample routines using plain `std::vector`s in `src/rnn_sample.h`. That file is 
licensed under the AGPL (`rnndescent` as a whole remains GPL3).

# rnndescent 0.0.5 (23 November 2019)

## New features

* New function: `merge_knn`, to combine two nearest neighbor graphs. Useful for
combining the results of multiple runs of `nnd_knn` or `random_knn`. Also,
`merge_knnl`, which operates on a list of multiple neighbor graphs, and can
provide a speed up over `merge_knn` if you don't mind storing multiple graphs
in memory at once.

## Bug fixes and minor improvements

* There was a thread-locking issue to do with converting R matrices to the 
internal heap data structure that affected `nnd_knn` with `n_threads > 1` 
and `random_knn` with `n_threads > 1` and `order_by_distance = TRUE`.
* Potential minor speed improvement for `nnd_knn` with `n_threads > 1` due to
the use of a mutex pool.

# rnndescent 0.0.4 (21 November 2019)

Mainly an internal clean-up to reduce duplication.

## Bug fixes and minor improvements

* By default, `nnd_knn` and `nnd_knn_query` use the same progress bar as
the brute force and random neighbor functions. Bring back the old per-iteration
logging that also showed the current distance sum of the knn with the
`progress = "dist"` option.
* For `random_knn` and `random_knn_query`, when `order_by_distance = TRUE` and
`n_threads > 0`, the final sorting of the knn graph will be multi-threaded.
* Initialization of the nearest neighbor descent data structures are also
multi-threaded if `n_threads > 0`.
* Progress bar updating and cancellation should now be more consistent and less
likely to cause hanging and crashing across the different methods.
* Using Cosine and Hamming distance may take up less memory or run a bit faster.

# rnndescent 0.0.3 (15 November 2019)

## New features

* There are now "query" versions of the three functions: `nnd_knn_query` being
the most useful, but `brute_force_knn_query` and `random_knn_query` are also
available. This allows for `query` data to search `reference` data, i.e. the
returned indices and distances are relative to the `reference` data, not any
other member of `query`. These methods are also available in multi-threaded
mode, and `nnd_knn_query` has a low and high memory version.

## Bug fixes and minor improvements

* Incremental search in nearest neighbor descent didn't work correctly, 
because retained new neighbors were marked as new rather than old. This made the
search repeat distance calculations unnecessarily.
* Heap initialization ignored existing distances in the input distance matrix.
* The `l2` metric has been renamed to `l2sqr` to more accurately reflect what 
it is: the square of the L2 (Euclidean) metric.
* New option `use_alt_metric`. Set to `FALSE` if you don't want alternative, 
faster metrics (which keep the distance ordering of `metric`) to be used in 
internal calculations. Currently only applies to `metric = "euclidean"`, where
the squared Euclidean distance is used internally. Only worth setting this to
`FALSE` if you think the alternative is causing numerical issues (which is
a bug, so please report it!).
* Random and brute force methods will make use of alternative metrics.
* New option `block_size` for parallel methods, which determines the amount
of work done in parallel before checking for user interrupt request and updating
any progress.
* `random_knn` now returns its results in sorted order. You can turn this off
with `order_distances = FALSE`, if you don't need the sorting (e.g. you are 
using the results as input to something else).
* Progress bars for the `brute_force` and `random` methods should now be 
correct.

# rnndescent 0.0.2 (7 November 2019)

## New features

* Brute force nearest neighbor function has been renamed to `brute_force_knn`.
* Random nearest neighbors has been renamed to `random_knn`.
* Brute force and random nearest neighbors are now interruptible.
* Progress bar will be shown if `verbose = TRUE`.
* `fast_rand` option has been removed, as it only applied to single-threading,
and had a negligible effect.

Also, a number of changes inspired by recent work in
<https://github.com/lmcinnes/pynndescent>:

* The parallel nearest neighbor descent should now be faster.
* The `rho` sampling parameter has been removed. The size of the candidates
(general neighbors) list is now controlled entirely by `max_candidates`.
* Default `max_candidates` has been reduced to 20.
* The `use_set` logical flag has been replaced by `low_memory`, which has the
opposite meaning. It now also works when using multiple threads. While it
follows the pynndescent implementation, it's still experimental, so
`low_memory = TRUE` by default for the moment.
* The `low_memory = FALSE` implementation for `n_threads = 0` (originally 
equivalent to `use_set = TRUE`) is faster.
* New parameter `block_size`, which balances interleaving of queuing updates 
versus applying them to the current graph.

## Bug fixes and minor improvements

* In the incremental search, the neighbors were marked as having been selected
for the new candidate list even if they were later removed due to the finite
size of the candidates heap. Now, only those indices that are still retained
after candidate building are marked as new.
* Improved man pages (examples plus link to nearest neighbor descent reference).
* Removed dependency on Boost headers.

# rnndescent 0.0.1 (27 October 2019)

* Initial release.

## New features

* Nearest Neighbor Descent with the following metrics: Euclidean, Cosine,
  Manhattan, Hamming.
* Support for multi-threading with RcppParallel.
* Initialization with random neighbors.
* Brute force nearest neighbor calculation.
