#' Calculate exact nearest neighbors by brute force.
#'
#' @param data Matrix of \code{n} items to generate random neighbors for.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
#'   or \code{"hamming"}.
#' @param n_threads Number of threads to use.
#' @param grain_size Minimum batch size for multithreading. If the number of
#'   items to process in a thread falls below this number, then no threads will
#'   be used. Ignored if \code{n_threads < 1}.
#' @param verbose If \code{TRUE}, log information to the console.
#' @return a list containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the nearest neighbor
#'   indices.
#'   \item \code{dist} an n by k matrix containing the nearest neighbor
#'    distances.
#' }
#' @export
nn_brute_force <- function(
                           data,
                           k,
                           metric = "euclidean",
                           n_threads = 0,
                           grain_size = 1,
                           verbose = FALSE) {
  data <- x2m(data)
  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }
  rnn_brute_force(data, k, metric, parallelize, grain_size, verbose)
}

#' Randomly select nearest neighbors.
#'
#' @param data Matrix of \code{n} items to generate random neighbors for.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
#'   or \code{"hamming"}.
#' @param use_cpp If \code{TRUE}, use the faster C++ code path.
#' @param n_threads Number of threads to use. Ignored if \code{use_cpp = FALSE}.
#' @return a list containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the nearest neighbor
#'   indices.
#'   \item \code{dist} an n by k matrix containing the nearest neighbor
#'    distances.
#' }
#' @export
random_nbrs <- function(data, k, metric = "euclidean", use_cpp = FALSE,
                        n_threads = 0) {
  data <- x2m(data)
  if (use_cpp) {
    parallelize <- n_threads > 0
    if (parallelize) {
      RcppParallel::setThreadOptions(numThreads = n_threads)
    }
    random_nbrs_cpp(data, k, metric, parallelize)
  }
  else {
    random_nbrs_R(X = data, k = k, metric = metric)
  }
}

#' Find Nearest Neighbors and Distances
#'
#' @param data Matrix of \code{n} items to search.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
#'   or \code{"hamming"}.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item.
#' @param delta precision parameter. Routine will terminate early if
#'   fewer than \eqn{\delta k N}{delta x k x n} updates are made to the nearest
#'   neighbor list in a given iteration.
#' @param rho Sample rate. This fraction of possible items will be used in the
#'   local join stage.
#' @param use_cpp If \code{TRUE}, use the faster C++ code path.
#' @param use_set If \code{TRUE}, cache pair lookups in a set. This increases
#'   speed at the cost of a memory use. Applies only if \code{use_cpp = TRUE}.
#' @param fast_rand If \code{TRUE}, use a faster random number generator than
#'   the R PRNG. Probably acceptable for the needs of the NN descent algorithm.
#'   Applies only if \code{use_cpp = TRUE}.
#' @param n_threads Number of threads to use.
#' @param grain_size Minimum batch size for multithreading. If the number of
#'   items to process in a thread falls below this number, then no threads will
#'   be used. Ignored if \code{n_threads < 1}.
#' @param verbose If \code{TRUE}, log information to the console.
#' @return a list containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the nearest neighbor indices.
#'   \item \code{dist} an n by k matrix containing the nearest neighbor
#'    distances.
#' }
#' @export
nnd_knn <- function(data, k,
                    metric = "euclidean",
                    n_iters = 10,
                    max_candidates = 50,
                    delta = 0.001, rho = 0.5,
                    use_cpp = TRUE,
                    use_set = FALSE,
                    fast_rand = FALSE,
                    n_threads = 0,
                    grain_size = 1,
                    verbose = FALSE) {
  data <- x2m(data)

  # As a minor optimization, we will use L2 internally if the user asks for
  # Euclidean and only take the square root of the final distances.
  actual_metric <- metric
  if (metric == "euclidean") {
    actual_metric <- "l2"
  }

  tsmessage("Initializing from random neighbors")
  init <- random_nbrs(data, k,
    metric = actual_metric, use_cpp = use_cpp,
    n_threads = n_threads
  )
  tsmessage("Init dsum = ", formatC(sum(init$dist)))
  init$idx <- init$idx - 1

  if (use_cpp) {
    parallelize <- n_threads > 0
    if (parallelize) {
      RcppParallel::setThreadOptions(numThreads = n_threads)
    }

    res <- nn_descent(data, init$idx, init$dist,
      metric = actual_metric,
      n_iters = n_iters, max_candidates = max_candidates,
      delta = delta, rho = rho, use_set = use_set, fast_rand = fast_rand,
      parallelize = parallelize, grain_size = grain_size,
      verbose = verbose
    )
  }
  else {
    res <- nn_descent_optl(data, init,
      metric = actual_metric, n_iters = n_iters,
      max_candidates = max_candidates,
      delta = delta, rho = rho, verbose = verbose
    )
    res$idx <- res$idx + 1
  }

  if (metric == "euclidean") {
    res$dist <- sqrt(res$dist)
  }
  tsmessage("Final dsum = ", formatC(sum(res$dist)))
  res
}

#' Nearest Neighbor Descent
#'
#' This function uses the nearest neighbor descent method to improve
#' approximate nearest neighbor data.
#'
#' @param data Matrix of \code{n} items to search.
#' @param idx an n by k matrix containing the initial nearest neighbor indices,
#'   where n is the number of items in \code{data} and k is the number of
#'   neighbors.
#' @param dist an n by k matrix containing the initial nearest neighbor
#'   distances, where n is the number of items in \code{data} and k is the
#'   number of neighbors.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"cosine"}, \code{"manhattan"} or \code{"hamming"}.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item.
#' @param delta precision parameter. Routine will terminate early if
#'   fewer than \eqn{\delta k N}{delta x k x n} updates are made to the nearest
#'   neighbor list in a given iteration.
#' @param rho Sample rate. This fraction of possible items will be used in the
#'   local join stage
#' @param verbose If \code{TRUE}, log information to the console.
#' @name nn_descent
#' @return a list containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the improved nearest neighbor
#'    indices.
#'   \item \code{dist} an n by k matrix containing the improved nearest neighbor
#'    distances.
#' }
#' @export
NULL


# Internals ---------------------------------------------------------------


random_nbrs_R <- function(X, k, metric = "euclidean") {
  nr <- nrow(X)
  idx <- matrix(0, nrow = nr, ncol = k)
  dist <- matrix(Inf, nrow = nr, ncol = k)
  dist_fn <- create_dist_fn(metric)

  for (i in 1:nr) {
    # we include i as its own neighbor
    # now sample k - 1  from 1:nr, excluding i
    # same as sampling from 1:(nr - 1) and adding one if its >= i
    idxi <- sample.int(nr - 1, k - 1)
    idxi[idxi >= i] <- idxi[idxi >= i] + 1
    idx[i, ] <- c(i, idxi)
    for (j in 2:k) {
      dist[i, j] <- dist_fn(X, i, idx[i, j])
    }
  }
  dist[, 1] <- 0.0
  list(idx = idx, dist = dist)
}

create_dist_fn <- function(metric) {
  switch(metric,
    l2 = l2_dist,
    euclidean = euc_dist,
    cosine = cos_dist,
    manhattan = manhattan_dist,
    hamming = hamming_dist,
    stop("Unknown metric '", metric, "'")
  )
}

det_nbrs <- function(X, k, metric = "euclidean") {
  dist_fn <- create_dist_fn(metric)
  nr <- nrow(X)
  indices <- matrix(0, nrow = nr, ncol = k)
  dist <- matrix(Inf, nrow = nr, ncol = k)

  for (i in 1:nr) {
    for (j in 1:k) {
      idx <- i + j
      if (idx > nr) {
        idx <- idx %% nr
      }
      indices[i, j] <- idx
      dist[i, j] <- dist_fn(X, i, idx)
    }
  }

  list(indices = indices, dist = dist)
}

#' @useDynLib rnndescent, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom RcppParallel RcppParallelLibs
.onUnload <- function(libpath) {
  library.dynam.unload("rnndescent", libpath)
}
