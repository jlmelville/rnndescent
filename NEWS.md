# rnndescent 0.0.10

## New features

* Generalize `hamming` metric to a standard definition. The old implementation
of `hamming` metric which worked on binary data only was renamed into
`bhamming`. (contributed by [Vitalie Spinu](https://github.com/vspinu))

## Bug fixes and minor improvements

* The `random_knn` function used to always return each item as its own neighbor,
so that only `n_nbrs - 1` of the returned neighbors were actually selected at
random. Even I forgot it did that and it doesn't make a lot of sense, so now you
really do just get back `n_nbrs` random selections.
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
