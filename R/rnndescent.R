# kNN Construction --------------------------------------------------------

#' Calculate Exact Nearest Neighbors by Brute Force
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
#' @examples
#' # Find the 4 nearest neighbors using Euclidean distance
#' # If you pass a data frame, non-numeric columns are removed
#' iris_nn <- brute_force_knn(iris, k = 4, metric = "euclidean")
#'
#' # Manhattan (l1) distance
#' iris_nn <- brute_force_knn(iris, k = 4, metric = "manhattan")
#'
#' # Multi-threading: you can choose the number of threads to use: in real
#' # usage, you will want to set n_threads to at least 2
#' iris_nn <- brute_force_knn(iris, k = 4, metric = "manhattan", n_threads = 1)
#'
#' # Use verbose flag to see information about progress
#' iris_nn <- brute_force_knn(iris, k = 4, metric = "euclidean", verbose = TRUE)
#' @export
brute_force_knn <- function(
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
#' @examples
#' # Find 4 random neighbors and calculate their Euclidean distance
#' # If you pass a data frame, non-numeric columns are removed
#' iris_nn <- random_knn(iris, k = 4, metric = "euclidean")
#'
#' # Manhattan (l1) distance
#' iris_nn <- random_knn(iris, k = 4, metric = "manhattan")
#'
#' # Multi-threading: you can choose the number of threads to use: in real
#' # usage, you will want to set n_threads to at least 2
#' iris_nn <- random_knn(iris, k = 4, metric = "manhattan", n_threads = 1)
#'
#' # Use verbose flag to see information about progress
#' iris_nn <- random_knn(iris, k = 4, metric = "euclidean", verbose = TRUE)
#'
#' # These results can be improved by nearest neighbors descent. You don't need
#' # to specify k here because this is worked out from the initial input
#' iris_nn <- nnd_knn(iris, init = iris_nn, metric = "euclidean", verbose = TRUE)
#' @export
random_knn <- function(data, k, metric = "euclidean", use_cpp = TRUE,
                       n_threads = 0, grain_size = 1, verbose = FALSE) {
  data <- x2m(data)
  nr <- nrow(data)
  if (k > nr) {
    stop("k must be <= ", nr)
  }
  parallelize <- n_threads > 0
  if (use_cpp || parallelize) {
    if (parallelize) {
      RcppParallel::setThreadOptions(numThreads = n_threads)
    }
    random_knn_cpp(data, k, metric, parallelize, grain_size = grain_size, verbose = verbose)
  }
  else {
    random_knn_R(X = data, k = k, metric = metric)
  }
}

#' Find Nearest Neighbors and Distances
#'
#' @param data Matrix of \code{n} items to search.
#' @param k Number of nearest neighbors to return. Optional if \code{init} is
#'   specified.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
#'   or \code{"hamming"}.
#' @param init Initial data to optimize. If not provided, \code{k} random
#'   neighbors are created. The input format should be the same as the return
#'   value: a list containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the nearest neighbor indices.
#'   \item \code{dist} an n by k matrix containing the nearest neighbor
#'   distances.
#' }
#' If \code{k} and \code{init} are provided then \code{k} must be equal to or
#' smaller than the number of neighbors provided in \code{init}. If smaller,
#' only the \code{k} closest value in \code{init} are retained.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item in each iteration. Use relative to \code{k} to emulate the "rho"
#'   sampling parameter in the nearest neighbor descent paper.
#' @param delta precision parameter. Routine will terminate early if
#'   fewer than \eqn{\delta k N}{delta x k x n} updates are made to the nearest
#'   neighbor list in a given iteration.
#' @param use_cpp If \code{TRUE}, use the faster C++ code path.
#' @param low_memory If \code{TRUE}, use a lower memory, but more
#'   computationally expensive approach to index construction. Applies only if
#'   \code{use_cpp = TRUE}. If set to \code{FALSE}, you should see a noticeable
#'   speed improvement, especially when using a smaller number of threads, so
#'   this is worth trying if you have the memory to spare.
#' @param n_threads Number of threads to use.
#' @param block_size Batch size for creating/applying local join updates. A
#'  smaller value will apply the update more often, which may help reduce the
#'  number of unnecessary distance calculations, at the cost of more overhead
#'  associated with multi-threading code. Ignored if \code{n_threads < 1}.
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
#' @examples
#' # Find 4 (approximate) nearest neighbors using Euclidean distance
#' # If you pass a data frame, non-numeric columns are removed
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean")
#'
#' # Manhattan (l1) distance
#' iris_nn <- nnd_knn(iris, k = 4, metric = "manhattan")
#'
#' # Multi-threading: you can choose the number of threads to use: in real
#' # usage, you will want to set n_threads to at least 2
#' iris_nn <- nnd_knn(iris, k = 4, metric = "manhattan", n_threads = 1)
#'
#' # Use verbose flag to see information about progress
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", verbose = TRUE)
#'
#' # Nearest neighbor descent uses random initialization, but you can pass any
#' # approximation using the init argument (as long as the metrics used to
#' # calculate the initialization are compatible with the metric options used
#' # by nnd_knn).
#' iris_nn <- random_knn(iris, k = 4, metric = "euclidean")
#' iris_nn <- nnd_knn(iris, init = iris_nn, metric = "euclidean", verbose = TRUE)
#'
#' # Number of iterations controls how much optimization is attempted. A smaller
#' # value will run faster but give poorer results
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", n_iters = 2)
#'
#' # You can also control the amount of work done within an iteration by
#' # setting max_candidates
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", max_candidates = 50)
#'
#' # Optimization may also stop early if not much progress is being made. This
#' # convergence criterion can be controlled via delta. A larger value will
#' # stop progress earlier. The verbose flag will provide some information if
#' # convergence is occurring before all iterations are carried out.
#' set.seed(1337)
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", n_iters = 5, delta = 0.5)
#'
#' # To ensure that descent only stops if no improvements are made, set delta = 0
#' set.seed(1337)
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", n_iters = 5, delta = 0)
#'
#' # A faster version of the algorithm is available that avoids repeated
#' # distance calculations at the cost of using more RAM. Set low_memory to
#' # FALSE to try it.
#' set.seed(1337)
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", low_memory = FALSE)
#' @references
#' Dong, W., Moses, C., & Li, K. (2011, March).
#' Efficient k-nearest neighbor graph construction for generic similarity measures.
#' In \emph{Proceedings of the 20th international conference on World Wide Web}
#' (pp. 577-586).
#' ACM.
#' \url{doi.org/10.1145/1963405.1963487}.
#' @export
nnd_knn <- function(data, k = NULL,
                    metric = "euclidean",
                    init = NULL,
                    n_iters = 10,
                    max_candidates = 20,
                    delta = 0.001,
                    use_cpp = TRUE,
                    low_memory = TRUE,
                    n_threads = 0,
                    block_size = 16384,
                    grain_size = 1,
                    verbose = FALSE) {
  data <- x2m(data)

  # As a minor optimization, we will use L2 internally if the user asks for
  # Euclidean and only take the square root of the final distances.
  actual_metric <- metric
  if (metric == "euclidean") {
    actual_metric <- "l2"
  }

  if (is.null(init)) {
    if (is.null(k)) {
      stop("Must provide k")
    }
    tsmessage("Initializing from random neighbors")
    init <- random_knn(data, k,
      metric = actual_metric, use_cpp = use_cpp,
      n_threads = n_threads, verbose = verbose
    )
  }
  else {
    if (is.null(k)) {
      k <- ncol(init$idx)
    }
    else if (k != ncol(init$idx)) {
      if (k > ncol(init$idx)) {
        stop("Not enough initial neighbors provided for k = ", k)
      }
      init$idx <- init$idx[, 1:k]
      init$dist <- init$dist[, 1:k]
    }
  }
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
      delta = delta, low_memory = low_memory,
      parallelize = parallelize, grain_size = grain_size,
      block_size = block_size, verbose = verbose
    )
  }
  else {
    res <- nn_descent_optl(data, init,
      metric = actual_metric, n_iters = n_iters,
      max_candidates = max_candidates,
      delta = delta, verbose = verbose
    )
    res$idx <- res$idx + 1
  }

  if (metric == "euclidean") {
    res$dist <- sqrt(res$dist)
  }
  tsmessage("Final dsum = ", formatC(sum(res$dist)))
  res
}


# kNN Queries -------------------------------------------------------------

#' Query Exact Nearest Neighbors by Brute Force
#'
#' @param reference Matrix of \code{m} reference items. The nearest neighbors to the
#'   queries are calculated from this data.
#' @param query Matrix of \code{n} query items.
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
#'   indices in \code{reference}.
#'   \item \code{dist} an n by k matrix containing the nearest neighbor
#'    distances to the items in \code{reference}.
#' }
#' @examples
#' # 100 reference iris items
#' iris_ref <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#'
#' # 50 query items
#' iris_query <- iris[iris$Species == "versicolor", ]
#'
#' # For each item in iris_query find the 4 nearest neighbors in iris_ref
#' # If you pass a data frame, non-numeric columns are removed
#' # set verbose = TRUE to get details on the progress being made
#' iris_query_nn <- brute_force_knn_query(iris_ref, iris_query,
#'   k = 4, metric = "euclidean",
#'   verbose = TRUE
#' )
#'
#' # Manhattan (l1) distance
#' iris_query_nn <- brute_force_knn_query(iris_ref, iris_query, k = 4, metric = "manhattan")
#' @export
brute_force_knn_query <- function(
                                  reference,
                                  query,
                                  k,
                                  metric = "euclidean",
                                  n_threads = 0,
                                  grain_size = 1,
                                  verbose = FALSE) {
  reference <- x2m(reference)
  query <- x2m(query)

  if (k > nrow(reference)) {
    stop(
      k, " neighbors asked for, but only ", nrow(reference),
      " items in the reference data"
    )
  }

  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }
  rnn_brute_force_query(reference, query, k, metric, parallelize, grain_size, verbose)
}

#' Nearest Neighbors Query by Random Selection
#'
#' @param reference Matrix of \code{m} reference items. The nearest neighbors to the
#'   queries are randomly selected from this data.
#' @param query Matrix of \code{n} query items.
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
#' @examples
#' # 100 reference iris items
#' iris_ref <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#'
#' # 50 query items
#' iris_query <- iris[iris$Species == "versicolor", ]
#'
#' # For each item in iris_query find 4 random neighbors in iris_ref
#' # If you pass a data frame, non-numeric columns are removed
#' # set verbose = TRUE to get details on the progress being made
#' iris_query_random_nbrs <- random_knn_query(iris_ref, iris_query,
#'   k = 4, metric = "euclidean", verbose = TRUE
#' )
#'
#' # Manhattan (l1) distance
#' iris_query_random_nbrs <- random_knn_query(iris_ref, iris_query, k = 4, metric = "manhattan")
#' @export
random_knn_query <- function(reference, query, k, metric = "euclidean",
                             n_threads = 0, grain_size = 1, verbose = FALSE) {
  reference <- x2m(reference)
  query <- x2m(query)
  nr <- nrow(reference)
  if (k > nr) {
    stop(
      k, " neighbors asked for, but only ", nrow(reference),
      " items in the reference data"
    )
  }
  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }
  random_knn_query_cpp(reference, query, k, metric, parallelize, grain_size = grain_size, verbose = verbose)
}


#' Find Nearest Neighbors and Distances
#'
#' @param reference Matrix of \code{m} reference items. The nearest neighbors to the
#'   queries are calculated from this data.
#' @param reference_idx Matrix of \code{m} by \code{k} integer indices
#'   representing the (possibly approximate) \code{k}-nearest neighbors graph
#'   of the \code{reference} data.
#' @param query Matrix of \code{n} query items.
#' @param k Number of nearest neighbors to return. Optional if \code{init} is
#'   specified.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
#'   or \code{"hamming"}.
#' @param init Initial data to optimize. If not provided, \code{k} random
#'   neighbors are created. The input format should be the same as the return
#'   value: a list containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the nearest neighbor indices
#'   of the data in \code{reference}.
#'   \item \code{dist} an n by k matrix containing the nearest neighbor
#'   distances.
#' }
#' If \code{k} and \code{init} are provided then \code{k} must be equal to or
#' smaller than the number of neighbors provided in \code{init}. If smaller,
#' only the \code{k} closest value in \code{init} are retained.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item in each iteration. Use relative to \code{k} to emulate the "rho"
#'   sampling parameter in the nearest neighbor descent paper.
#' @param delta precision parameter. Routine will terminate early if
#'   fewer than \eqn{\delta k N}{delta x k x n} updates are made to the nearest
#'   neighbor list in a given iteration.
#' @param low_memory If \code{TRUE}, use a lower memory, but more
#'   computationally expensive approach to index construction. If set to
#'   \code{FALSE}, you should see a noticeable speed improvement, especially
#'   when using a smaller number of threads, so this is worth trying if you
#'   have the memory to spare.
#' @param n_threads Number of threads to use.
#' @param block_size Batch size for creating/applying local join updates. A
#'  smaller value will apply the update more often, which may help reduce the
#'  number of unnecessary distance calculations, at the cost of more overhead
#'  associated with multi-threading code. Ignored if \code{n_threads < 1}.
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
#' @examples
#' # 100 reference iris items
#' iris_ref <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#'
#' # 50 query items
#' iris_query <- iris[iris$Species == "versicolor", ]
#'
#' # First, find the approximate 4-nearest neighbor graph for the references:
#' iris_ref_knn <- nnd_knn(iris_ref, k = 4, use_cpp = TRUE)
#'
#' # For each item in iris_query find the 4 nearest neighbors in iris_ref.
#' # You need to pass both the reference data and the knn graph indices (the
#' # 'idx' matrix in the return value of nnd_knn).
#' # If you pass a data frame, non-numeric columns are removed.
#' # set verbose = TRUE to get details on the progress being made
#' iris_query_nn <- nnd_knn_query(iris_ref, iris_ref_knn$idx, iris_query,
#'   k = 4, metric = "euclidean",
#'   verbose = TRUE
#' )
#' @references
#' Dong, W., Moses, C., & Li, K. (2011, March).
#' Efficient k-nearest neighbor graph construction for generic similarity measures.
#' In \emph{Proceedings of the 20th international conference on World Wide Web}
#' (pp. 577-586).
#' ACM.
#' \url{doi.org/10.1145/1963405.1963487}.
#' @export
nnd_knn_query <- function(reference, reference_idx, query, k = NULL,
                          metric = "euclidean",
                          init = NULL,
                          n_iters = 10,
                          max_candidates = 20,
                          delta = 0.001,
                          low_memory = TRUE,
                          n_threads = 0,
                          block_size = 16384,
                          grain_size = 1,
                          verbose = FALSE) {
  reference <- x2m(reference)
  query <- x2m(query)

  # As a minor optimization, we will use L2 internally if the user asks for
  # Euclidean and only take the square root of the final distances.
  actual_metric <- metric
  if (metric == "euclidean") {
    actual_metric <- "l2"
  }

  if (is.null(init)) {
    if (is.null(k)) {
      stop("Must provide k")
    }
    tsmessage("Initializing from random neighbors")
    init <- random_knn_query(reference, query, k,
      metric = actual_metric, n_threads = n_threads,
      verbose = verbose
    )
  }
  else {
    if (is.null(k)) {
      k <- ncol(init$idx)
    }
    else if (k != ncol(init$idx)) {
      if (k > ncol(init$idx)) {
        stop("Not enough initial neighbors provided for k = ", k)
      }
      init$idx <- init$idx[, 1:k]
      init$dist <- init$dist[, 1:k]
    }
  }
  tsmessage("Init dsum = ", formatC(sum(init$dist)))
  init$idx <- init$idx - 1
  reference_idx <- reference_idx - 1

  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }

  res <- nn_descent_query(reference, reference_idx, query, init$idx, init$dist,
    metric = actual_metric,
    n_iters = n_iters, max_candidates = max_candidates,
    delta = delta, low_memory = low_memory,
    parallelize = parallelize, grain_size = grain_size,
    block_size = block_size, verbose = verbose
  )

  if (metric == "euclidean") {
    res$dist <- sqrt(res$dist)
  }
  tsmessage("Final dsum = ", formatC(sum(res$dist)))
  res
}


# Internals ---------------------------------------------------------------


random_knn_R <- function(X, k, metric = "euclidean") {
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


nn_accuracy <- function(idx, ref_idx, k = NULL, include_self = TRUE) {
  if (is.list(idx)) {
    idx <- idx$idx
  }
  if (is.list(ref_idx)) {
    ref_idx <- ref_idx$idx
  }

  if (is.null(k)) {
    k <- min(ncol(idx), ncol(ref_idx))
    message("Using k = ", k)
  }

  if (ncol(ref_idx) < k) {
    stop("Not enough columns in ref_idx for k = ", k)
  }

  n <- nrow(idx)
  if (nrow(ref_idx) != n) {
    stop("Not enough rows in ref_idx")
  }

  nbr_start <- 1
  nbr_end <- k

  ref_start <- nbr_start
  ref_end <- nbr_end
  if (!include_self) {
    ref_start <- ref_start + 1
    if (nbr_end < ncol(ref_idx)) {
      ref_end <- ref_end + 1
    }
    else {
      nbr_end <- nbr_end - 1
    }
  }

  nbr_range <- nbr_start:nbr_end
  ref_range <- ref_start:ref_end

  total_intersect <- 0
  for (i in 1:n) {
    total_intersect <- total_intersect + length(intersect(idx[i, nbr_range], ref_idx[i, ref_range]))
  }

  total_intersect / (n * k)
}
