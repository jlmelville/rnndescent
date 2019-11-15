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
