# rnndescent 0.0.2

## New features

* Brute force nearest neighbor function has been renamed to `brute_force_knn`.
* Random nearest neighbors has been renamed to `random_knn`.
* Brute force and random nearest neighbors are now interruptible.
* Progress bar will be shown if `verbose = TRUE`.

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
