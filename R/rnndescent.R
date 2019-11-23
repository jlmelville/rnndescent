# kNN Construction --------------------------------------------------------

#' Calculate Exact Nearest Neighbors by Brute Force
#'
#' @param data Matrix of \code{n} items to generate random neighbors for.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
#'   or \code{"hamming"}.
#' @param use_alt_metric If \code{TRUE}, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   \code{metric = "euclidean"}), then apply a correction at the end. Probably
#'   the only reason to set this to \code{FALSE} is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param n_threads Number of threads to use.
#' @param block_size Number of items to generate neighbors for in each
#'   multi-threaded batch. Reducing this number will increase the frequency
#'   with which R will check for cancellation, and if \code{verbose = TRUE},
#'   the frequency with which progress will be logged to the console. This value
#'   should not be set too low (and not lower than \code{grain_size}), or the
#'   overhead of cancellation checking and other multi-threaded house keeping
#'   will reduce the efficiency of the parallel computation. Ignored if
#'   \code{n_threads < 1}.
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
                            use_alt_metric = TRUE,
                            n_threads = 0,
                            block_size = 64,
                            grain_size = 1,
                            verbose = FALSE) {
  data <- x2m(data)
  check_k(k, nrow(data))

  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
  }
  else {
    actual_metric <- metric
  }

  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }
  res <- rnn_brute_force(data, k, actual_metric, parallelize, block_size, grain_size, verbose)

  if (use_alt_metric) {
    res$dist <- apply_alt_metric_correction(metric, res$dist)
  }

  res
}

#' Randomly select nearest neighbors.
#'
#' @param data Matrix of \code{n} items to generate random neighbors for.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
#'   or \code{"hamming"}.
#' @param use_alt_metric If \code{TRUE}, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   \code{metric = "euclidean"}), then apply a correction at the end. Probably
#'   the only reason to set this to \code{FALSE} is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param order_by_distance If \code{TRUE} (the default), then results for each
#'   item are returned by increasing distance. If you don't need the results
#'   sorted, e.g. you are going to pass the results as initialization to another
#'   routine like \code{\link{nnd_knn}}, set this to \code{FALSE} to save a
#'   small amount of computational time.
#' @param n_threads Number of threads to use.
#' @param block_size Number of items to generate neighbors for in each
#'   multi-threaded batch. Reducing this number will increase the frequency
#'   with which R will check for cancellation, and if \code{verbose = TRUE},
#'   the frequency with which progress will be logged to the console. This value
#'   should not be set too low (and not lower than \code{grain_size}), or the
#'   overhead of cancellation checking and other multi-threaded house keeping
#'   will reduce the efficiency of the parallel computation. Ignored if
#'   \code{n_threads < 1}.
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
random_knn <- function(data, k, metric = "euclidean", use_alt_metric = TRUE,
                       order_by_distance = TRUE,
                       n_threads = 0, block_size = 4096, grain_size = 1,
                       verbose = FALSE) {
  data <- x2m(data)
  check_k(k, nrow(data))

  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
  }
  else {
    actual_metric <- metric
  }

  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }

  res <- random_knn_cpp(data, k, actual_metric, order_by_distance, parallelize, block_size, grain_size, verbose)

  if (use_alt_metric) {
    res$dist <- apply_alt_metric_correction(metric, res$dist)
  }

  res
}

#' Find Nearest Neighbors and Distances
#'
#' @param data Matrix of \code{n} items to search.
#' @param k Number of nearest neighbors to return. Optional if \code{init} is
#'   specified.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
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
#' only the \code{k} closest value in \code{init} are retained. The input
#' distances may be ignored if \code{use_alt_metric = TRUE} and no
#' transformation from the distances to the internal distance representation
#' is available. In this case, the distances will be recalculated internally
#' and only the contents of \code{init$idx} will be used.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item in each iteration. Use relative to \code{k} to emulate the "rho"
#'   sampling parameter in the nearest neighbor descent paper.
#' @param delta Precision parameter. Routine will terminate early if
#'   fewer than \eqn{\delta k N}{delta x k x n} updates are made to the nearest
#'   neighbor list in a given iteration.
#' @param low_memory If \code{TRUE}, use a lower memory, but more
#'   computationally expensive approach to index construction. If set to
#'   \code{FALSE}, you should see a noticeable speed improvement, especially
#'   when using a smaller number of threads, so this is worth trying if you have
#'   the memory to spare.
#' @param use_alt_metric If \code{TRUE}, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   \code{metric = "euclidean"}), then apply a correction at the end. Probably
#'   the only reason to set this to \code{FALSE} is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param n_threads Number of threads to use.
#' @param block_size Batch size for creating/applying local join updates. A
#'  smaller value will apply the update more often, which may help reduce the
#'  number of unnecessary distance calculations, at the cost of more overhead
#'  associated with multi-threading code. Ignored if \code{n_threads < 1}.
#' @param grain_size Minimum batch size for multithreading. If the number of
#'   items to process in a thread falls below this number, then no threads will
#'   be used. Ignored if \code{n_threads < 1}.
#' @param verbose If \code{TRUE}, log information to the console.
#' @param progress Determines the type of progress information logged if
#'   \code{verbose = TRUE}. Options are:
#'   \itemize{
#'     \item \code{"bar"}: a simple text progress bar.
#'     \item \code{"dist"}: the sum of the distances in the approximate knn
#'     graph at the end of each iteration.
#'   }
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
                    low_memory = TRUE,
                    use_alt_metric = TRUE,
                    n_threads = 0,
                    block_size = 16384,
                    grain_size = 1,
                    verbose = FALSE,
                    progress = "bar") {
  stopifnot(tolower(progress) %in% c("bar", "dist"))

  data <- x2m(data)

  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
    if (!(is.null(init))) {
      init$dist <- apply_alt_metric_uncorrection(metric, init$dist)
    }
  }
  else {
    actual_metric <- metric
  }

  if (is.null(init)) {
    if (is.null(k)) {
      stop("Must provide k")
    }
    tsmessage("Initializing from random neighbors")
    init <- random_knn(data, k,
      metric = actual_metric,
      order_by_distance = FALSE,
      n_threads = n_threads,
      block_size = block_size,
      grain_size = grain_size,
      verbose = verbose
    )
  }
  else {
    if (is.null(k)) {
      k <- ncol(init$idx)
    }
    else {
      init <- prepare_init_graph(init, k)
    }
  }

  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }

  tsmessage("Running nearest neighbor descent for ", n_iters, " iterations")
  res <- nn_descent(data, init$idx, init$dist,
    metric = actual_metric,
    n_iters = n_iters, max_candidates = max_candidates,
    delta = delta, low_memory = low_memory,
    parallelize = parallelize,
    block_size = block_size, grain_size = grain_size, verbose = verbose,
    progress = progress
  )

  if (use_alt_metric) {
    res$dist <- apply_alt_metric_correction(metric, res$dist)
  }
  tsmessage("Finished")
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
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
#'   or \code{"hamming"}.
#' @param use_alt_metric If \code{TRUE}, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   \code{metric = "euclidean"}), then apply a correction at the end. Probably
#'   the only reason to set this to \code{FALSE} is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param n_threads Number of threads to use.
#' @param block_size Number of items to generate neighbors for in each
#'   multi-threaded batch. Reducing this number will increase the frequency
#'   with which R will check for cancellation, and if \code{verbose = TRUE},
#'   the frequency with which progress will be logged to the console. This value
#'   should not be set too low (and not lower than \code{grain_size}), or the
#'   overhead of cancellation checking and other multi-threaded house keeping
#'   will reduce the efficiency of the parallel computation. Ignored if
#'   \code{n_threads < 1}.
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
                                  use_alt_metric = TRUE,
                                  n_threads = 0,
                                  block_size = 64,
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

  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
  }
  else {
    actual_metric <- metric
  }

  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }

  res <- rnn_brute_force_query(
    reference, query, k, actual_metric, parallelize, block_size,
    grain_size, verbose
  )

  if (use_alt_metric) {
    res$dist <- apply_alt_metric_correction(metric, res$dist)
  }
  res
}

#' Nearest Neighbors Query by Random Selection
#'
#' @param reference Matrix of \code{m} reference items. The nearest neighbors to the
#'   queries are randomly selected from this data.
#' @param query Matrix of \code{n} query items.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
#'   or \code{"hamming"}.
#' @param use_alt_metric If \code{TRUE}, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   \code{metric = "euclidean"}), then apply a correction at the end. Probably
#'   the only reason to set this to \code{FALSE} is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param order_by_distance If \code{TRUE} (the default), then results for each
#'   item are returned by increasing distance. If you don't need the results
#'   sorted, e.g. you are going to pass the results as initialization to another
#'   routine like \code{\link{nnd_knn_query}}, set this to \code{FALSE} to save a
#'   small amount of computational time.
#' @param n_threads Number of threads to use.
#' @param block_size Number of items to generate neighbors for in each
#'   multi-threaded batch. Reducing this number will increase the frequency
#'   with which R will check for cancellation, and if \code{verbose = TRUE},
#'   the frequency with which progress will be logged to the console. This value
#'   should not be set too low (and not lower than \code{grain_size}), or the
#'   overhead of cancellation checking and other multi-threaded house keeping
#'   will reduce the efficiency of the parallel computation. Ignored if
#'   \code{n_threads < 1}.
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
                             use_alt_metric = TRUE, order_by_distance = TRUE,
                             n_threads = 0, block_size = 4096, grain_size = 1,
                             verbose = FALSE) {
  reference <- x2m(reference)
  query <- x2m(query)
  nr <- nrow(reference)

  if (k > nr) {
    stop(
      k, " neighbors asked for, but only ", nrow(reference),
      " items in the reference data"
    )
  }

  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
  }
  else {
    actual_metric <- metric
  }

  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }

  res <- random_knn_query_cpp(
    reference, query, k, actual_metric, order_by_distance, parallelize, block_size,
    grain_size, verbose
  )
  if (use_alt_metric) {
    res$dist <- apply_alt_metric_correction(metric, res$dist)
  }
  res
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
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
#'   or \code{"hamming"}.
#' @param use_alt_metric If \code{TRUE}, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   \code{metric = "euclidean"}), then apply a correction at the end. Probably
#'   the only reason to set this to \code{FALSE} is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
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
#' only the \code{k} closest value in \code{init} are retained. The input
#' distances may be ignored if \code{use_alt_metric = TRUE} and no
#' transformation from the distances to the internal distance representation
#' is available. In this case, the distances will be recalculated internally
#' and only the contents of \code{init$idx} will be used.
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
#' @param progress Determines the type of progress information logged if
#'   \code{verbose = TRUE}. Options are:
#'   \itemize{
#'     \item \code{"bar"}: a simple text progress bar.
#'     \item \code{"dist"}: the sum of the distances in the approximate knn
#'     graph at the end of each iteration.
#'   }
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
#' iris_ref_knn <- nnd_knn(iris_ref, k = 4)
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
                          use_alt_metric = TRUE,
                          n_threads = 0,
                          block_size = 16384,
                          grain_size = 1,
                          verbose = FALSE,
                          progress = "bar") {
  stopifnot(tolower(progress) %in% c("bar", "dist"))
  reference <- x2m(reference)
  query <- x2m(query)

  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
    if (!(is.null(init))) {
      init$dist <- apply_alt_metric_uncorrection(metric, init$dist)
    }
  }
  else {
    actual_metric <- metric
  }

  if (is.null(init)) {
    if (is.null(k)) {
      k <- ncol(reference_idx)
      message("Using k = ", k, " from reference graph indices")
    }
    tsmessage("Initializing from random neighbors")
    init <- random_knn_query(reference, query, k,
      order_by_distance = FALSE,
      metric = actual_metric, n_threads = n_threads,
      verbose = verbose
    )
  }
  else {
    if (is.null(k)) {
      k <- ncol(init$idx)
      message("Using k = ", k, " from initial graph")
    }
    else {
      init <- prepare_init_graph(init, k)
    }
  }
  # init indices are zero indexed in the C++ code
  # reference indices need to be zero-indexed manually
  reference_idx <- prepare_ref_idx(reference_idx, k) - 1

  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }

  res <- nn_descent_query(reference, reference_idx, query, init$idx, init$dist,
    metric = actual_metric,
    n_iters = n_iters, max_candidates = max_candidates,
    delta = delta, low_memory = low_memory,
    parallelize = parallelize,
    block_size = block_size, grain_size = grain_size, verbose = verbose,
    progress = progress
  )

  if (use_alt_metric) {
    res$dist <- apply_alt_metric_correction(metric, res$dist)
  }
  res
}



# Merge -------------------------------------------------------------------

merge_knn <- function(nn_graph1, nn_graph2, is_query = FALSE,
                      n_threads = 0, block_size = 4096, grain_size = 1,
                      verbose = FALSE) {
  validate_are_mergeable(nn_graph1, nn_graph2)

  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }

  merge_nn(
    nn_graph1$idx, nn_graph1$dist, nn_graph2$idx, nn_graph2$dist,
    is_query, parallelize, block_size, grain_size
  )
}

merge_knnl <- function(nn_graphs, is_query = FALSE,
                       n_threads = 0, block_size = 4096, grain_size = 1,
                       verbose = FALSE) {
  if (length(nn_graphs) == 0) {
    return(list())
  }
  validate_are_mergeablel(nn_graphs)

  parallelize <- n_threads > 0
  if (parallelize) {
    RcppParallel::setThreadOptions(numThreads = n_threads)
  }

  merge_nn_all(
    nn_graphs,
    is_query, parallelize, block_size, grain_size
  )
}

# Internals ---------------------------------------------------------------

#' @useDynLib rnndescent, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom RcppParallel RcppParallelLibs
.onUnload <- function(libpath) {
  library.dynam.unload("rnndescent", libpath)
}

check_k <- function(k, max_k) {
  if (k > max_k) {
    stop("k must be <= ", max_k)
  }
}

prepare_init_graph <- function(nn, k) {
  if (k != ncol(nn$idx)) {
    if (k > ncol(nn$idx)) {
      stop("Not enough initial neighbors provided for k = ", k)
    }
    nn$idx <- nn$idx[, 1:k]
    nn$dist <- nn$dist[, 1:k]
  }
  nn
}

prepare_ref_idx <- function(idx, k) {
  if (k != ncol(idx)) {
    if (k > ncol(idx)) {
      stop("Not enough reference indices provided for k = ", k)
    }
    idx <- idx[, 1:k]
  }
  idx
}

find_alt_metric <- function(metric) {
  switch(metric,
    euclidean = "l2sqr",
    metric
  )
}

apply_alt_metric_uncorrection <- function(metric, dist) {
  switch(metric,
    euclidean = dist * dist,
    dist
  )
}

apply_alt_metric_correction <- function(metric, dist) {
  switch(metric,
    euclidean = sqrt(dist),
    dist
  )
}


validate_are_mergeablel <- function(nn_graphs) {
  nn_graph1 <- nn_graphs[[1]]
  validate_nn_graph(nn_graph1)
  n_graphs <- length(nn_graphs)
  if (n_graphs > 1) {
    for (i in 2:n_graphs) {
      validate_are_mergeable(nn_graph1, nn_graphs[[i]], validate1 = FALSE)
    }
  }
}

validate_are_mergeable <- function(nn_graph1, nn_graph2, validate1 = TRUE) {
  if (validate1) {
    validate_nn_graph(nn_graph1)
  }
  validate_nn_graph(nn_graph2)
  nr1 <- nrow(nn_graph1$idx)
  nr2 <- nrow(nn_graph2$idx)
  if (nr1 != nr2) {
    stop("Graphs must have same number of rows, but are ", nr1, ", ", nr2)
  }
}

validate_nn_graph <- function(nn_graph) {
  if (is.null(nn_graph$idx)) {
    stop("NN graph must contain 'idx' matrix")
  }
  if (is.null(nn_graph$dist)) {
    stop("NN graph must contain 'dist' matrix")
  }
  nr <- nrow(nn_graph$idx)
  nc <- ncol(nn_graph$idx)
  validate_nn_graph_matrix(nn_graph$dist, nr, nc, msg = "nn matrix")
}

validate_nn_graph_matrix <- function(nn, nr, nc, msg = "matrix") {
  nnr <- nrow(nn)
  nnc <- ncol(nn)
  if (nr != nnr) {
    stop(msg, " has ", nnr, " rows, should have ", nr)
  }
  if (nc != nnc) {
    stop(msg, " has ", nnc, " cols, should have ", nc)
  }
}
