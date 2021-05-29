# kNN Construction --------------------------------------------------------

#' Calculate Exact Nearest Neighbors by Brute Force
#'
#' @param data Matrix of \code{n} items to generate random neighbors for.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"},
#'   \code{"correlation"} (1 minus the Pearson correlation), or
#'   \code{"hamming"}.
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
brute_force_knn <- function(data,
                            k,
                            metric = "euclidean",
                            use_alt_metric = TRUE,
                            n_threads = 0,
                            block_size = 64,
                            grain_size = 1,
                            verbose = FALSE) {
  data <- x2m(data)
  check_k(k, nrow(data))

  if (metric == "correlation") {
    data <- row_center(data)
    metric <- "cosine"
  }
  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
  }
  else {
    actual_metric <- metric
  }

  tsmessage("Calculating brute force k-nearest neighbors with k = ", k)
  res <-
    rnn_brute_force(
      data,
      k,
      actual_metric,
      block_size = block_size,
      n_threads = n_threads,
      grain_size = grain_size,
      verbose = verbose
    )
  res$idx <- res$idx + 1

  if (use_alt_metric) {
    res$dist <- apply_alt_metric_correction(metric, res$dist)
  }
  tsmessage("Finished")
  res
}

#' Randomly select nearest neighbors.
#'
#' @param data Matrix of \code{n} items to generate random neighbors for.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"},
#'   \code{"correlation"} (1 minus the Pearson correlation), or
#'   \code{"hamming"}.
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
random_knn <-
  function(data,
           k,
           metric = "euclidean",
           use_alt_metric = TRUE,
           order_by_distance = TRUE,
           n_threads = 0,
           block_size = 4096,
           grain_size = 1,
           verbose = FALSE) {
    data <- x2m(data)
    check_k(k, nrow(data))
    if (metric == "correlation") {
      data <- row_center(data)
      metric <- "cosine"
    }
    if (use_alt_metric) {
      actual_metric <- find_alt_metric(metric)
    }
    else {
      actual_metric <- metric
    }

    tsmessage("Generating random k-nearest neighbor graph with k = ", k)
    res <-
      random_knn_cpp(
        data,
        k,
        actual_metric,
        order_by_distance,
        block_size = block_size,
        n_threads = n_threads,
        grain_size = grain_size,
        verbose = verbose
      )
    res$idx <- res$idx + 1

    if (use_alt_metric) {
      res$dist <- apply_alt_metric_correction(metric, res$dist)
    }

    tsmessage("Finished")
    res
  }

#' Find Nearest Neighbors and Distances
#'
#' @param data Matrix of \code{n} items to search.
#' @param k Number of nearest neighbors to return. Optional if \code{init} is
#'   specified.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"},
#'   \code{"correlation"} (1 minus the Pearson correlation), or
#'   \code{"hamming"}.
#' @param init Initial data to optimize. If not provided, \code{k} random
#'   neighbors are created. The input format should be the same as the return
#'   value: a list containing:
#' \itemize{
#'   \item \code{idx} an \code{n} by \code{k} matrix containing the nearest neighbor indices.
#'   \item \code{dist} (optional) an \code{n} by \code{k} matrix containing the nearest neighbor
#'   distances.
#' }
#' If \code{k} and \code{init} are provided then \code{k} must be equal to or
#' smaller than the number of neighbors provided in \code{init}. If smaller,
#' only the \code{k} closest value in \code{init} are retained. If the input
#' distances are omitted, they will be calculated for you.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item in each iteration. Use relative to \code{k} to emulate the "rho"
#'   sampling parameter in the nearest neighbor descent paper. By default, this
#'   is set to \code{k} or \code{60}, whichever is smaller.
#' @param delta The minimum relative change in the neighbor graph allowed before
#'   early stopping. Should be a value between 0 and 1. The smaller the value,
#'   the smaller the amount of progress between iterations is allowed. Default
#'   value of \code{0.001} means that at least 0.1% of the neighbor graph must
#'   be updated at each iteration.
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
#' @param weighted If \code{TRUE} then instead of ordering general neighbor
#'   candidates randomly, they are ordered by increasing in-degree. This gives
#'   priority to nodes which don't appear in many other neighbor lists, and
#'   which might get crowded out of the few candidate lists they appear in by
#'   more "popular" nodes. For high dimensional datasets, this can slightly
#'   ameliorate the effect of hubs and modestly improve query times (for a given
#'   accuracy).
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
#' \url{https://doi.org/10.1145/1963405.1963487}.
#' @export
nnd_knn <- function(data,
                    k = NULL,
                    metric = "euclidean",
                    init = NULL,
                    n_iters = 10,
                    max_candidates = NULL,
                    delta = 0.001,
                    low_memory = TRUE,
                    use_alt_metric = TRUE,
                    weighted = FALSE,
                    n_threads = 0,
                    block_size = 16384,
                    grain_size = 1,
                    verbose = FALSE,
                    progress = "bar") {
  stopifnot(tolower(progress) %in% c("bar", "dist"))
  data <- x2m(data)
  if (metric == "correlation") {
    data <- row_center(data)
    metric <- "cosine"
  }
  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
    if (!is.null(init) && !is.null(init$dist)) {
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
    init <- random_knn(
      data,
      k,
      metric = actual_metric,
      order_by_distance = FALSE,
      block_size = block_size,
      n_threads = n_threads,
      grain_size = grain_size,
      verbose = verbose
    )
  }
  else {
    if (is.null(k)) {
      k <- ncol(init$idx)
    }
  }
  init <-
    prepare_init_graph(
      init,
      k,
      data = data,
      metric = actual_metric,
      n_threads = n_threads,
      grain_size = grain_size,
      verbose = verbose
    )

  if (is.null(max_candidates)) {
    max_candidates <- min(k, 60)
  }
  tsmessage("Running nearest neighbor descent for ", n_iters, " iterations")
  res <- nn_descent(
    data,
    init$idx,
    init$dist,
    metric = actual_metric,
    n_iters = n_iters,
    max_candidates = max_candidates,
    delta = delta,
    low_memory = low_memory,
    block_size = block_size,
    n_threads = n_threads,
    grain_size = grain_size,
    verbose = verbose,
    progress = progress,
    weighted = weighted
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
#' @param query Matrix of \code{n} query items.
#' @param k Number of nearest neighbors to return.
#' @param reference Matrix of \code{m} reference items. The nearest neighbors to the
#'   queries are calculated from this data.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"},
#'   \code{"correlation"} (1 minus the Pearson correlation), or
#'   \code{"hamming"}.
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
#' iris_query_nn <- brute_force_knn_query(iris_query,
#'   reference = iris_ref,
#'   k = 4, metric = "euclidean", verbose = TRUE
#' )
#'
#' # Manhattan (l1) distance
#' iris_query_nn <- brute_force_knn_query(iris_query,
#'   reference = iris_ref,
#'   k = 4, metric = "manhattan"
#' )
#' @export
brute_force_knn_query <- function(query,
                                  reference,
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
      k,
      " neighbors asked for, but only ",
      nrow(reference),
      " items in the reference data"
    )
  }
  if (metric == "correlation") {
    reference <- row_center(reference)
    query <- row_center(query)
    metric <- "cosine"
  }
  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
  }
  else {
    actual_metric <- metric
  }

  tsmessage("Calculating brute force k-nearest neighbors from reference with k = ", k)
  res <- rnn_brute_force_query(
    reference,
    query,
    k,
    actual_metric,
    block_size = block_size,
    n_threads = n_threads,
    grain_size = grain_size,
    verbose = verbose
  )
  res$idx <- res$idx + 1

  if (use_alt_metric) {
    res$dist <- apply_alt_metric_correction(metric, res$dist)
  }
  tsmessage("Finished")

  res
}

#' Nearest Neighbors Query by Random Selection
#'
#' @param query Matrix of \code{n} query items.
#' @param reference Matrix of \code{m} reference items. The nearest neighbors to the
#'   queries are randomly selected from this data.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"},
#'   \code{"correlation"} (1 minus the Pearson correlation), or
#'   \code{"hamming"}.
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
#' iris_query_random_nbrs <- random_knn_query(iris_query,
#'   reference = iris_ref,
#'   k = 4, metric = "euclidean", verbose = TRUE
#' )
#'
#' # Manhattan (l1) distance
#' iris_query_random_nbrs <- random_knn_query(iris_query,
#'   reference = iris_ref,
#'   k = 4, metric = "manhattan"
#' )
#' @export
random_knn_query <-
  function(query,
           reference,
           k,
           metric = "euclidean",
           use_alt_metric = TRUE,
           order_by_distance = TRUE,
           n_threads = 0,
           block_size = 4096,
           grain_size = 1,
           verbose = FALSE) {
    reference <- x2m(reference)
    query <- x2m(query)
    nr <- nrow(reference)

    if (k > nr) {
      stop(
        k,
        " neighbors asked for, but only ",
        nrow(reference),
        " items in the reference data"
      )
    }

    if (metric == "correlation") {
      reference <- row_center(reference)
      query <- row_center(query)
      metric <- "cosine"
    }
    if (use_alt_metric) {
      actual_metric <- find_alt_metric(metric)
    }
    else {
      actual_metric <- metric
    }

    tsmessage("Generating random k-nearest neighbor graph from reference with k = ", k)
    res <- random_knn_query_cpp(
      reference,
      query,
      k,
      actual_metric,
      order_by_distance,
      block_size = block_size,
      n_threads = n_threads,
      grain_size = grain_size,
      verbose = verbose
    )
    res$idx <- res$idx + 1
    if (use_alt_metric) {
      res$dist <- apply_alt_metric_correction(metric, res$dist)
    }
    tsmessage("Finished")

    res
  }


#' Find Nearest Neighbors and Distances
#'
#' @param query Matrix of \code{n} query items.
#' @param reference Matrix of \code{m} reference items. The nearest neighbors to the
#'   items in \code{query} are calculated from this data.
#' @param reference_graph Nearest neighbor graph of the \code{reference} data,
#'   the output of running \code{nnd_knn}. The format is a list containing:
#' \itemize{
#'   \item \code{idx} an \code{m} by \code{k} matrix containing the nearest
#'   neighbor indices of the data in \code{reference}.
#'   \item \code{dist} an \code{m} by \code{k} matrix containing the nearest
#'   neighbor distances.
#' }
#' @param k Number of nearest neighbors to return. Optional if \code{init} is
#'   specified.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2sqr"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"},
#'   \code{"correlation"} (1 minus the Pearson correlation), or
#'   \code{"hamming"}.
#' @param use_alt_metric If \code{TRUE}, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   \code{metric = "euclidean"}), then apply a correction at the end. Probably
#'   the only reason to set this to \code{FALSE} is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param init Initial \code{query} neighbor graph to optimize. If not
#'   provided, \code{k} random neighbors from \code{reference} are used. The
#'   format should be the same as the return value of this function, a list
#'   containing:
#' \itemize{
#'   \item \code{idx} a \code{n} by \code{k} matrix containing the nearest neighbor
#'   indices specifying the row of the neighbor in \code{reference}.
#'   \item \code{dist} (optional) a \code{n} by \code{k} matrix containing the
#'   nearest neighbor distances.
#' }
#'   If \code{k} and \code{init} are provided then \code{k} must be equal to or
#'   smaller than the number of neighbors provided in \code{init}. If smaller,
#'   only the \code{k} closest value in \code{init} are retained. If the input
#' distances are omitted, they will be calculated for you.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#'   This is set to \code{Inf} by default. For controlling the time cost of
#'   querying, it is recommended to modify \code{epsilon} initially. However,
#'   setting this parameter can also provide a safe-guard against excessive
#'   search time.
#' @param epsilon Controls trade-off between accuracy and search cost, by
#'   specifying a distance tolerance on whether to explore the neighbors of
#'   candidate points. The larger the value, the more neighbors will be
#'   searched. A value of 0.1 allows query-candidate distances to be 10% larger
#'   than the current most-distant neighbor of the query point, 0.2 means 20%,
#'   and so on. Suggested values are between 0-0.5, although this value is
#'   highly dependent on the distribution of distances in the dataset (higher
#'   dimensional data should choose a smaller cutoff). Too large a value of
#'   \code{epsilon} will result in the query search approaching brute force
#'   comparison. Use this parameter in conjunction with \code{n_iters} and
#'   \code{max_candidates} to prevent excessive run time. Default is 0.1.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item in each iteration. Use relative to \code{k} to emulate the "rho"
#'   sampling parameter in the nearest neighbor descent paper. By default, this
#'   is set to \code{k} or \code{60}, whichever is smaller.
#' @param n_threads Number of threads to use.
#' @param grain_size Minimum batch size for multithreading. If the number of
#'   items to process in a thread falls below this number, then no threads will
#'   be used. Ignored if \code{n_threads < 1}.
#' @param verbose If \code{TRUE}, log information to the console.
#' @return the optimized query neighbor graph, a list containing:
#' \itemize{
#'   \item \code{idx} a \code{n} by \code{k} matrix containing the nearest neighbor
#'   indices specifying the row of the neighbor in \code{reference}.
#'   \item \code{dist} a \code{n} by \code{k} matrix containing the nearest neighbor
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
#' iris_ref_graph <- nnd_knn(iris_ref, k = 4)
#'
#' # For each item in iris_query find the 4 nearest neighbors in iris_ref.
#' # You need to pass both the reference data and the reference graph.
#' # If you pass a data frame, non-numeric columns are removed.
#' # set verbose = TRUE to get details on the progress being made
#' iris_query_nn <- nnd_knn_query(iris_query, iris_ref, iris_ref_graph,
#'   k = 4, metric = "euclidean", verbose = TRUE
#' )
#' @references
#' Hajebi, K., Abbasi-Yadkori, Y., Shahbazi, H., & Zhang, H. (2011, June).
#' Fast approximate nearest-neighbor search with k-nearest neighbor graph.
#' In \emph{Twenty-Second International Joint Conference on Artificial Intelligence}.
#'
#' Harwood, B., & Drummond, T. (2016).
#' Fanng: Fast approximate nearest neighbour graphs.
#' In \emph{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}
#' (pp. 5713-5722).
#'
#' Iwasaki, M., & Miyazaki, D. (2018).
#' Optimization of indexing based on k-nearest neighbor graph for proximity
#' search in high-dimensional data.
#' \emph{arXiv preprint arXiv:1810.07355.}
#'
#' @export
nnd_knn_query <- function(query,
                          reference,
                          reference_graph,
                          k = NULL,
                          metric = "euclidean",
                          init = NULL,
                          n_iters = Inf,
                          epsilon = 0.1,
                          max_candidates = NULL,
                          use_alt_metric = TRUE,
                          n_threads = 0,
                          grain_size = 1,
                          verbose = FALSE) {
  reference <- x2m(reference)
  query <- x2m(query)

  if (metric == "correlation") {
    reference <- row_center(reference)
    query <- row_center(query)
    metric <- "cosine"
  }
  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
    if (!is.null(init) && !is.null(init$dist)) {
      init$dist <- apply_alt_metric_uncorrection(metric, init$dist)
    }
  }
  else {
    actual_metric <- metric
  }

  if (is.null(init)) {
    if (is.null(k)) {
      if (is.list(reference_graph)) {
        k <- get_reference_graph_k(reference_graph)
        tsmessage("Using k = ", k, " from nnd model")
      }
      else {
        stop("Must provide k")
      }
    }
    tsmessage("Initializing from random neighbors")
    init <- random_knn_query(
      query = query,
      reference = reference,
      k = k,
      order_by_distance = FALSE,
      metric = actual_metric,
      n_threads = n_threads,
      verbose = verbose
    )
  }
  else {
    if (is.null(k)) {
      k <- ncol(init$idx)
      tsmessage("Using k = ", k, " from initial graph")
    }
  }
  init <-
    prepare_init_graph(
      nn = init,
      k = k,
      query = query,
      data = reference,
      metric = actual_metric,
      n_threads = n_threads,
      grain_size = grain_size,
      verbose = verbose
    )
  if (is.null(max_candidates)) {
    max_candidates <- min(k, 60)
  }
  # We need to convert from Inf to an actual integer value. This is sufficiently
  # big for our purposes
  if (is.infinite(n_iters)) {
    n_iters <- .Machine$integer.max
  }

  stopifnot(!is.null(query), methods::is(query, "matrix"))
  stopifnot(
    !is.null(init$idx),
    methods::is(init$idx, "matrix"),
    ncol(init$idx) == k,
    nrow(init$idx) == nrow(query)
  )
  stopifnot(
    !is.null(init$dist),
    methods::is(init$dist, "matrix"),
    ncol(init$dist) == k,
    nrow(init$dist) == nrow(query)
  )

  if (is.list(reference_graph)) {
    reference_graph <- prepare_reference_graph(reference_graph, max_candidates,
      verbose = verbose
    )
    reference_dist <- reference_graph$dist
    reference_idx <- reference_graph$idx
    stopifnot(!is.null(reference), methods::is(reference, "matrix"))
    stopifnot(
      !is.null(reference_idx),
      methods::is(reference_idx, "matrix"),
      nrow(reference_idx) == nrow(reference)
    )
    stopifnot(
      !is.null(reference_dist),
      methods::is(reference_dist, "matrix"),
      nrow(reference_dist) == nrow(reference)
    )
    reference_graph_list <- graph_to_list(reference_graph)
  }
  else {
    stopifnot(methods::is(reference_graph, "sparseMatrix"))
    reference_graph_list <- sparse_to_list(reference_graph)
  }

  tsmessage(thread_msg("Searching nearest neighbor graph", n_threads))
  res <-
    nn_descent_query(
      reference = reference,
      reference_graph_list = reference_graph_list,
      query = query,
      nn_idx = init$idx,
      nn_dist = init$dist,
      metric = actual_metric,
      max_candidates = max_candidates,
      epsilon = epsilon,
      n_iters = n_iters,
      n_threads = n_threads,
      grain_size = grain_size,
      verbose = verbose
    )
  if (use_alt_metric) {
    res$dist <- apply_alt_metric_correction(metric, res$dist)
  }
  tsmessage("Finished")
  res
}

# Search Graph Preparation ------------------------------------------------

diversify <- function(data,
                      graph,
                      metric = "euclidean",
                      prune_probability = 1.0,
                      verbose = FALSE) {
  idx <- graph$idx
  dist <- graph$dist

  stopifnot(
    methods::is(idx, "matrix"),
    nrow(data) == nrow(idx),
    nrow(data) >= ncol(idx),
    max(idx) <= nrow(data)
  )
  nnz_before <- sum(idx != 0)
  res <- diversify_cpp(x2m(data), idx, dist, prune_probability = prune_probability)
  nnz_after <- sum(res$idx != 0)
  tsmessage("Diversifying reduced # edges from ", nnz_before,
            " to ", nnz_after,
            " (", formatC(100 * nn_sparsity(res)), "% sparse)")
  res
}

# Harwood, B., & Drummond, T. (2016).
# Fanng: Fast approximate nearest neighbour graphs.
# In \emph{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}
# (pp. 5713-5722).
# "Occlusion pruning"
diversify_sp <- function(data,
                         graph,
                         metric = "euclidean",
                         prune_probability = 1.0,
                         verbose = FALSE) {
  nnz_before <- Matrix::nnzero(graph)
  sp_before <- nn_sparsity_sp(graph)
  stopifnot(
    methods::is(graph, "sparseMatrix")
  )
  gl <- sparse_to_list(graph)

  if (prune_probability < 1) {
    gl_div <- diversify_sp_cpp(data = x2m(data), graph_list = gl, metric = metric,
                               prune_probability = prune_probability)
  }
  else {
    gl_div <- diversify_always_sp_cpp(data = x2m(data), graph_list = gl, metric = metric)
  }
  res <- list_to_sparse(gl_div)
  nnz_after <- Matrix::nnzero(res)
  tsmessage("Diversifying reduced # edges from ", nnz_before,
            " to ", nnz_after,
            " (",  formatC(100 * sp_before), "% to ",
            formatC(100 * nn_sparsity_sp(res)),   "% sparse)")
  res
}

# FANNG: Fast Approximate Nearest Neighbour Graphs
# "truncating"
# Harwood, B., & Drummond, T. (2016).
# Fanng: Fast approximate nearest neighbour graphs.
# In \emph{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}
# (pp. 5713-5722).
degree_prune <- function(graph, max_degree = 20, verbose = FALSE) {
  nnz_before <- Matrix::nnzero(graph)
  sp_before <- nn_sparsity_sp(graph)
  stopifnot(
    methods::is(graph, "sparseMatrix")
  )
  gl <- sparse_to_list(graph)

  gl_div <- degree_prune_cpp(gl, max_degree)
  res <- list_to_sparse(gl_div)
  nnz_after <- Matrix::nnzero(res)
  tsmessage("Degree pruning to max ", max_degree, " reduced # edges from ", nnz_before,
            " to ", nnz_after,
            " (",  formatC(100 * sp_before), "% to ",
            formatC(100 * nn_sparsity_sp(res)),   "% sparse)")
  res
}


merge_graphs_sp <- function(g1, g2) {
  gl1 <- sparse_to_list(g1)
  gl2 <- sparse_to_list(g2)

  gl_merge <- merge_graph_lists_cpp(gl1, gl2)
  list_to_sparse(gl_merge)
}

nn_sparsity <- function(graph) {
  sum(graph$idx == 0) / prod(dim(graph$idx))
}

nn_sparsity_sp <- function(graph) {
  Matrix::nnzero(graph) / prod(graph@Dim)
}

# Merge -------------------------------------------------------------------

#' Merge two approximate nearest neighbors graphs
#'
#' @param nn_graph1 A nearest neighbor graph to merge. Should consist of a list
#'   containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the k nearest neighbor
#'   indices.
#'   \item \code{dist} an n by k matrix containing k nearest neighbor
#'    distances.
#' }
#' @param nn_graph2 Another nearest neighbor graph to merge with the same
#'   format as \code{nn_graph1}. The number of neighbors can differ between
#'   graphs, but the merged result will have the same number of neighbors as
#'   specified in \code{nn_graph1}.
#' @param is_query If \code{TRUE} then the graphs are treated as the result of
#'   a knn query, not a knn building process. This should be set to
#'   \code{TRUE} if \code{nn_graph1} and \code{nn_graph2} are the results of
#'   using e.g. \code{\link{nnd_knn_query}} or \code{\link{random_knn_query}},
#'   and set to \code{FALSE} if these are the results of
#'   \code{\link{nnd_knn}} or \code{\link{random_knn}}. The difference is that
#'   if \code{is_query = FALSE}, if an index \code{p} is found in
#'   \code{nn_graph1[i, ]}, i.e. \code{p} is a neighbor of \code{i} with
#'   distance \code{d}, then it is assumed that \code{i} is a neighbor of
#'   \code{p} with the same distance. If \code{is_query = TRUE}, then \code{i}
#'   and \code{p} are indexes into two different datasets and the symmetry does
#'   not hold. If you aren't sure what case applies to you, it's safe (but
#'   potentially inefficient) to set \code{is_query = TRUE}.
#' @param n_threads Number of threads to use.
#' @param block_size Number of items to process in each multi-threaded batch.
#'   Reducing this number will increase the frequency with which R will check
#'   for cancellation, and if \code{verbose = TRUE}, the frequency with which
#'   progress will be logged to the console. This value should not be set too
#'   low (and not lower than \code{grain_size}), or the overhead of cancellation
#'   checking and other multi-threaded house keeping will reduce the efficiency
#'   of the parallel computation. Ignored if \code{n_threads < 1}.
#' @param grain_size Minimum batch size for multithreading. If the number of
#'   items to process in a thread falls below this number, then no threads will
#'   be used. Ignored if \code{n_threads < 1}.
#' @param verbose If \code{TRUE}, log information to the console.
#' @return a list containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the merged nearest neighbor
#'   indices.
#'   \item \code{dist} an n by k matrix containing the merged nearest neighbor
#'    distances.
#' }
#'   The size of \code{k} in the output graph is the same as that of
#'   \code{nn_graph1}.
#' @examples
#' set.seed(1337)
#' # Nearest neighbor descent with 15 neighbors for iris three times,
#' # starting from a different random initialization each time
#' iris_rnn1 <- nnd_knn(iris, k = 15, n_iters = 1)
#' iris_rnn2 <- nnd_knn(iris, k = 15, n_iters = 1)
#'
#' # Merged results should be an improvement over either individual results
#' iris_mnn <- merge_knn(iris_rnn1, iris_rnn2)
#' sum(iris_mnn$dist) < sum(iris_rnn1$dist)
#' sum(iris_mnn$dist) < sum(iris_rnn2$dist)
#' @export
merge_knn <- function(nn_graph1,
                      nn_graph2,
                      is_query = FALSE,
                      n_threads = 0,
                      block_size = 4096,
                      grain_size = 1,
                      verbose = FALSE) {
  validate_are_mergeable(nn_graph1, nn_graph2)

  merge_nn(
    nn_graph1$idx,
    nn_graph1$dist,
    nn_graph2$idx,
    nn_graph2$dist,
    is_query,
    block_size = block_size,
    n_threads = n_threads,
    grain_size = grain_size,
    verbose = verbose
  )
}

#' Merge a list of approximate nearest neighbors graphs
#'
#' @param nn_graphs A list of nearest neighbor graph to merge. Each item in the
#'   list should consist of a sub-list
#'   containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the k nearest neighbor
#'   indices.
#'   \item \code{dist} an n by k matrix containing k nearest neighbor
#'    distances.
#' }
#'   The number of neighbors can differ between graphs, but the merged result
#'   will have the same number of neighbors as the first graph in the list.
#' @param is_query If \code{TRUE} then the graphs are treated as the result of
#'   a knn query, not a knn building process. This should be set to
#'   \code{TRUE} if \code{nn_graphs} are the results of
#'   using e.g. \code{\link{nnd_knn_query}} or \code{\link{random_knn_query}},
#'   and set to \code{FALSE} if these are the results of \code{\link{nnd_knn}}
#'   or \code{\link{random_knn}}. The difference is that if \code{is_query =
#'   FALSE}, if an index \code{p} is found in \code{nn_graph1[i, ]}, i.e.
#'   \code{p} is a neighbor of \code{i} with distance \code{d}, then it is
#'   assumed that \code{i} is a neighbor of \code{p} with the same distance. If
#'   \code{is_query = TRUE}, then \code{i} and \code{p} are indexes into two
#'   different datasets and the symmetry does not hold. If you aren't sure what
#'   case applies to you, it's safe (but potentially inefficient) to set
#'   \code{is_query = TRUE}.
#' @param n_threads Number of threads to use.
#' @param block_size Number of items to process in each multi-threaded batch.
#'   Reducing this number will increase the frequency with which R will check
#'   for cancellation, and if \code{verbose = TRUE}, the frequency with which
#'   progress will be logged to the console. This value should not be set too
#'   low (and not lower than \code{grain_size}), or the overhead of cancellation
#'   checking and other multi-threaded house keeping will reduce the efficiency
#'   of the parallel computation. Ignored if \code{n_threads < 1}.
#' @param grain_size Minimum batch size for multithreading. If the number of
#'   items to process in a thread falls below this number, then no threads will
#'   be used. Ignored if \code{n_threads < 1}.
#' @param verbose If \code{TRUE}, log information to the console.
#' @return a list containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the merged nearest neighbor
#'   indices.
#'   \item \code{dist} an n by k matrix containing the merged nearest neighbor
#'    distances.
#' }
#'   The size of \code{k} in the output graph is the same as that of the first
#'   item in \code{nn_graphs}.
#' @examples
#' set.seed(1337)
#' # Nearest neighbor descent with 15 neighbors for iris three times,
#' # starting from a different random initialization each time
#' iris_rnn1 <- nnd_knn(iris, k = 15, n_iters = 1)
#' iris_rnn2 <- nnd_knn(iris, k = 15, n_iters = 1)
#' iris_rnn3 <- nnd_knn(iris, k = 15, n_iters = 1)
#'
#' # Merged results should be an improvement over individual results
#' iris_mnn <- merge_knnl(list(iris_rnn1, iris_rnn2, iris_rnn3))
#' sum(iris_mnn$dist) < sum(iris_rnn1$dist)
#' sum(iris_mnn$dist) < sum(iris_rnn2$dist)
#' sum(iris_mnn$dist) < sum(iris_rnn3$dist)
#'
#' # and slightly faster than running:
#' # iris_mnn <- merge_knn(iris_rnn1, iris_rnn2)
#' # iris_mnn <- merge_knn(iris_mnn, iris_rnn3)
#' @export
merge_knnl <- function(nn_graphs,
                       is_query = FALSE,
                       n_threads = 0,
                       block_size = 4096,
                       grain_size = 1,
                       verbose = FALSE) {
  if (length(nn_graphs) == 0) {
    return(list())
  }
  validate_are_mergeablel(nn_graphs)

  merge_nn_all(
    nn_graphs,
    is_query,
    block_size = block_size,
    n_threads = n_threads,
    grain_size = grain_size,
    verbose = verbose
  )
}

# Hubness -----------------------------------------------------------------

#' k-Occurrence
#'
#' Calculates the the k-occurrence for a given nearest neighbor matrix
#'
#' The k-occurrence of an object is the number of times it occurs among the
#' k-nearest neighbors of objects in a dataset. This can take values between 0
#' and the size of the dataset. The larger the k-occurrence for an object, the
#' more "popular" it is. Very large values of the k-occurrence (much larger than
#' k) indicates that an object is a "hub" and also implies the existence of
#' "anti-hubs": objects that never appear as k-nearest neighbors of other
#' objects.
#'
#' The presence of hubs can reduce the accuracy of nearest-neighbor descent
#' and other approximate nearest neighbor algorithms in terms of retrieving the
#' exact k-nearest neighbors. However the appearance of hubs can still be
#' detected in these approximate results, so calculating the k-occurrences for
#' the output of nearest neighbor descent is a useful diagnostic step.
#'
#' @param idx integer matrix containing the nearest neighbor indices, integers
#'   labeled starting at 1. Note that the integer labels do \emph{not} have to
#'   refer to the rows of \code{idx}, for example if the nearest neighbor result
#'   is from querying one set of objects with respect to another (for instance
#'   from running \code{\link{nnd_knn_query}}). You may also pass a nearest
#'   neighbor graph object (e.g. the output of running \code{\link{nnd_knn}}),
#'   and the indices will be extracted from it.
#' @param k The number of closest neighbors to use. Must be between 1 and the
#'   number of columns in \code{idx}. By default, all columns of \code{idx} are
#'   used.
#' @param include_self logical indicating whether the label \code{i} in
#'   \code{idx} is considered to be a valid neighbor when found in row \code{i}.
#'   By default this is \code{TRUE}. This can be set to \code{FALSE} when the
#'   the labels in \code{idx} refer to the row indices of \code{idx}, as in the
#'   case of results from \code{\link{nnd_knn}}. In this case you may not want
#'   to consider the trivial case of an object being a neighbor of itself. In
#'   all other cases leave this set to \code{TRUE}.
#' @return a vector of length \code{max(idx)}, containing the number of times an
#'   object in \code{idx} was found in the nearest neighbor list of the objects
#'   represented by the row indices of \code{idx}.
#' @examples
#' iris_nbrs <- brute_force_knn(iris, k = 15)
#' iris_ko <- k_occur(iris_nbrs$idx)
#' # items 42 and 107 are not in 15 nearest neighbors of any other members of
#' # iris
#' # for convenience you can also pass iris_nbrs directly:
#' # iris_ko <- k_occur(iris_nbrs)
#' which(iris_ko == 1) # they are only their own nearest neighbor
#' max(iris_ko) # most "popular" item appears on 29 15-nearest neighbor lists
#' which(iris_ko == max(iris_ko)) # it's iris item 64
#' # with k = 15, a maximum k-occurrence = 29 ~= 1.9 * k, which is not a cause
#' # for concern
#' @references
#' Radovanovic, M., Nanopoulos, A., & Ivanovic, M. (2010).
#' Hubs in space: Popular nearest neighbors in high-dimensional data.
#' \emph{Journal of Machine Learning Research}, \emph{11}, 2487-2531.
#' \url{https://www.jmlr.org/papers/v11/radovanovic10a.html}
#'
#' Bratic, B., Houle, M. E., Kurbalija, V., Oria, V., & Radovanovic, M. (2019).
#' The Influence of Hubness on NN-Descent.
#' \emph{International Journal on Artificial Intelligence Tools}, \emph{28}(06), 1960002.
#' \url{https://doi.org/10.1142/S0218213019600029}
#' @export
k_occur <- function(idx,
                    k = NULL,
                    include_self = TRUE) {
  if (is.list(idx)) {
    idx <- idx$idx
  }
  stopifnot(methods::is(idx, "matrix"))
  if (is.null(k)) {
    k <- ncol(idx)
  }
  nc <- ncol(idx)
  stopifnot(k >= 1)
  stopifnot(k <= nc)
  len <- max(idx)
  reverse_nbr_size_impl(idx, k, len, include_self)
}

check_graph <- function(idx, dist = NULL, k = NULL) {
  if (is.null(dist) && is.list(idx)) {
    dist <- idx$dist
  }
  if (is.list(idx)) {
    idx <- idx$idx
  }
  stopifnot(methods::is(idx, "matrix"))
  stopifnot(methods::is(dist, "matrix"))
  stopifnot(dim(idx) == dim(dist))
  if (is.null(k)) {
    k <- ncol(idx)
  }
  stopifnot(k > 0)
  list(idx = idx, dist = dist, k = k)
}

reverse_knn <- function(idx, dist = NULL, k = NULL) {
  cg_res <- check_graph(idx = idx, dist = dist, k = k)
  Matrix::t(Matrix::drop0(graph_to_sparse_r(cg_res)))
}

reverse_knn_sp <- function(graph) {
  stopifnot(methods::is(graph, "sparseMatrix"))
  Matrix::t(Matrix::drop0(graph))
}

mutualize_knn <- function(idx, dist = NULL, k = NULL) {
  cg_res <- check_graph(idx = idx, dist = dist, k = k)
  spg <- graph_to_cs(cg_res)
  rspg <-reverse_knn(cg_res)
  Matrix::drop0(spg + rspg) / 2
}

partial_mutualize_knn <- function(idx, dist = NULL, k = NULL) {
  cg_res <- check_graph(idx = idx, dist = dist, k = k)
  partial_mutualize_graph_impl(idx = cg_res$idx, dist = cg_res$dist, n_nbrs = cg_res$k)
}

ko_adj_graph <- function(idx, dist = NULL, rev_k = NULL, fwd_k = NULL) {
  if (is.null(dist) && is.list(idx)) {
    dist <- idx$dist
  }
  if (is.list(idx)) {
    idx <- idx$idx
  }
  if (is.null(rev_k)) {
    rev_k <- ncol(idx)
  }
  if (is.null(fwd_k)) {
    fwd_k <- ncol(idx)
  }

  stopifnot(methods::is(idx, "matrix"))
  stopifnot(methods::is(dist, "matrix"))
  stopifnot(dim(idx) == dim(dist))
  stopifnot(rev_k > 0)
  stopifnot(fwd_k > 0)

  ko_adj_graph_impl(idx = idx, dist = dist, n_rev_nbrs = rev_k, n_adj_nbrs = fwd_k)
}

deg_adj_graph <- function(idx, dist = NULL, rev_k = NULL, fwd_k = NULL) {
  if (is.null(dist) && is.list(idx)) {
    dist <- idx$dist
  }
  if (is.list(idx)) {
    idx <- idx$idx
  }
  if (is.null(rev_k)) {
    rev_k <- ncol(idx)
  }
  if (is.null(fwd_k)) {
    fwd_k <- ncol(idx)
  }

  stopifnot(methods::is(idx, "matrix"))
  stopifnot(methods::is(dist, "matrix"))
  stopifnot(dim(idx) == dim(dist))
  stopifnot(rev_k > 0)
  stopifnot(fwd_k > 0)

  deg_adj_graph_impl(idx = idx, dist = dist, n_rev_nbrs = rev_k, n_adj_nbrs = fwd_k)
}

unreachable <- function(idx, k = NULL) {
  if (is.list(idx)) {
    idx <- idx$idx
  }
  ko <- k_occur(idx, k = k, include_self = FALSE)
  sum(ko == 0) / length(ko)
}

reachable <- function(idx, k = NULL) {
  if (is.list(idx)) {
    idx <- idx$idx
  }
  ko <- k_occur(idx, k = k, include_self = FALSE)
  1 - (sum(ko == 0) / length(ko))
}

graph_components <- function(graph, n_nbrs = NULL, n_ref = NULL) {
  connected_components(graph_to_sparse_r(graph, n_nbrs = n_nbrs))$n_components
}

connected_components <- function(X_csr) {
  X_t_csr <- Matrix::t(X_csr)
  connected_components_undirected(nrow(X_csr), X_csr@j, X_csr@p, X_t_csr@j, X_t_csr@p)
}

create_pruned_search_graph <- function(data, graph, metric = "euclidean",
                                       prune_probability = 1.0,
                                       pruning_degree_multiplier = 1.5,
                                       verbose = FALSE) {
  tsmessage("Converting graph to sparse format")
  sp <- graph_to_cs(graph)
  if (!is.null(prune_probability) && prune_probability > 0) {
    tsmessage("Diversifying forward graph")
    fdiv <- diversify_sp(data, sp, metric = metric, prune_probability = prune_probability,
                         verbose = verbose)
  }
  else {
    fdiv <- sp
    tsmessage("Forward graph has # edges = ", Matrix::nnzero(fdiv), " (",
              formatC(100 * nn_sparsity_sp(fdiv)), "% sparse)")
  }
  rsp <-  reverse_knn_sp(fdiv)

  if (!is.null(prune_probability) && prune_probability > 0) {
    tsmessage("Diversifying reverse graph")
    rdiv <- diversify_sp(data, rsp, metric = metric, prune_probability = prune_probability,
                         verbose = verbose)
  }
  else {
    rdiv <- rsp
    tsmessage("Reverse graph has # edges = ", Matrix::nnzero(rdiv), " (",
              formatC(100 * nn_sparsity_sp(rdiv)), "% sparse)")
  }
  tsmessage("Merging diversified forward and reverse graph")
  merged <- merge_graphs_sp(fdiv, rdiv)
  if (!is.null(pruning_degree_multiplier) && !is.infinite(pruning_degree_multiplier)) {
    max_degree <- round(ncol(graph$idx) * pruning_degree_multiplier)
    tsmessage("Degree pruning merged graph to max degree: ", max_degree)
    res <- degree_prune(merged, max_degree = max_degree, verbose = verbose)
  }
  else {
    res <- merged
    tsmessage("Merged graph has # edges = ", Matrix::nnzero(res), " (",
              formatC(100 * nn_sparsity_sp(res)), "% sparse)")
  }
  tsmessage("Finished")
  res
}


# Idx to Graph ------------------------------------------------------------

idx_to_graph <-
  function(data,
           idx,
           metric = "euclidean",
           n_threads = 0,
           grain_size = 1,
           verbose = FALSE) {
    if (is.list(idx)) {
      if (is.null(idx$idx)) {
        stop("Couldn't find 'idx' matrix in graph")
      }
      idx <- idx$idx
    }
    stopifnot(
      methods::is(idx, "matrix"),
      nrow(data) == nrow(idx),
      nrow(data) >= ncol(idx),
      max(idx) <= nrow(data)
    )

    tsmessage("Calculating distances for neighbor indexes")
    rnn_idx_to_graph_self(x2m(data), idx, metric = metric, n_threads = n_threads)
  }

idx_to_graph_query <-
  function(query,
           reference,
           idx,
           metric = "euclidean",
           n_threads = 0,
           grain_size = 1,
           verbose = FALSE) {
    if (is.list(idx)) {
      if (is.null(idx$idx)) {
        stop("Couldn't find 'idx' matrix in graph")
      }
      idx <- idx$idx
    }
    stopifnot(
      methods::is(idx, "matrix"),
      nrow(query) == nrow(idx),
      nrow(reference) >= ncol(idx),
      max(idx) <= nrow(reference)
    )

    tsmessage("Calculating distances for neighbor indexes")
    rnn_idx_to_graph_query(
      reference = x2m(reference),
      query = x2m(query),
      idx = idx, metric = metric, n_threads = n_threads
    )
  }

# Internals ---------------------------------------------------------------

#' @useDynLib rnndescent, .registration = TRUE
#' @importFrom Rcpp sourceCpp
# Suppress R CMD check note "Namespace in Imports field not imported from"
#' @importFrom dqrng dqset.seed
.onUnload <- function(libpath) {
  library.dynam.unload("rnndescent", libpath)
}

check_k <- function(k, max_k) {
  if (k > max_k) {
    stop("k must be <= ", max_k)
  }
}

prepare_init_graph <-
  function(nn,
           k,
           data,
           query = NULL,
           metric = "euclidean",
           n_threads = 0,
           grain_size = 1,
           verbose = FALSE) {
    if (k != ncol(nn$idx)) {
      if (k > ncol(nn$idx)) {
        stop("Not enough initial neighbors provided for k = ", k)
      }
      nn$idx <- nn$idx[, 1:k, drop = FALSE]
    }
    if (!is.null(nn$dist)) {
      if (k > ncol(nn$dist)) {
        stop("Not enough initial distances provided for k = ", k)
      }
      nn$dist <- nn$dist[, 1:k, drop = FALSE]
    }
    else {
      tsmessage("Generating distances for initial indices")
      if (!is.null(query)) {
        nn <-
          rnn_idx_to_graph_query(
            reference = data,
            query = query,
            idx = nn$idx,
            metric = metric,
            n_threads = n_threads,
            grain_size = grain_size,
            verbose = verbose
          )
      }
      else {
        nn <-
          rnn_idx_to_graph_self(
            data = data,
            idx = nn$idx,
            metric = metric,
            n_threads = n_threads,
            grain_size = grain_size,
            verbose = verbose
          )
      }
    }

    nn
  }

prepare_reference_graph <- function(reference_graph, max_candidates, verbose = FALSE) {
  for (name in c("idx", "dist")) {
    m <- reference_graph[[name]]
    stopifnot(
      !is.null(m),
      methods::is(m, "matrix")
    )
    reference_graph[[name]] <- m
  }
  stopifnot(dim(reference_graph$idx) == dim(reference_graph$dist))
  if (ncol(reference_graph$idx) < max_candidates) {
    tsmessage(
      "Note: reference graph columns = ", ncol(reference_graph$idx),
      " < requested max_candidates = ", max_candidates
    )
    max_candidates <- ncol(reference_graph$idx)
  }
  else if (ncol(reference_graph$idx) > max_candidates) {
    tsmessage("Reducing reference graph columns to ", max_candidates)
  }

  for (name in c("idx", "dist")) {
    m <- reference_graph[[name]]
    if (max_candidates != ncol(m)) {
      m <- m[, 1:max_candidates, drop = FALSE]
    }
    reference_graph[[name]] <- m
  }

  reference_graph
}

get_reference_graph_k <- function(reference_graph) {
  ncol(reference_graph$idx)
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

validate_are_mergeable <-
  function(nn_graph1, nn_graph2, validate1 = TRUE) {
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

row_center <- function(data) {
  sweep(data, 1, rowMeans(data))
}


# Sparse ------------------------------------------------------------------

graph_to_sparse_r <- function(graph, n_nbrs = NULL, n_ref = NULL) {
  if (is.list(graph)) {
    idx <- graph$idx
    dist <- graph$dist
  }
  else {
    idx <- graph
    dist <- 1
  }
  if (is.null(n_nbrs)) {
    n_nbrs <- ncol(idx)
  }
  else {
    idx <- idx[, 1:n_nbrs]
    if (methods::is(dist, "matrix")) {
      dist <- dist[, 1:n_nbrs]
    }
  }
  n_row <- nrow(idx)
  # If this is query data, you need to provide the number of reference items
  # yourself: we can't guarantee that all indices are included in idx
  if (is.null(n_ref)) {
    n_ref <- nrow(idx)
  }
  Matrix::sparseMatrix(
    i = rep(1:n_row, times = n_nbrs),
    j = as.vector(idx),
    x = as.vector(dist),
    dims = c(n_row, n_ref),
    repr = "R"
  )
}

graph_to_cs <- function(graph, n_nbrs = NULL, n_ref = NULL) {
  if (is.list(graph)) {
    idx <- graph$idx
    dist <- graph$dist
  }
  else {
    idx <- graph
    dist <- 1
  }
  if (is.null(n_nbrs)) {
    n_nbrs <- ncol(idx)
  }
  else {
    idx <- idx[, 1:n_nbrs]
    if (methods::is(dist, "matrix")) {
      dist <- dist[, 1:n_nbrs]
    }
  }
  n_row <- nrow(idx)
  # If this is query data, you need to provide the number of reference items
  # yourself: we can't guarantee that all indices are included in idx
  if (is.null(n_ref)) {
    n_ref <- nrow(idx)
  }
  Matrix::drop0(Matrix::sparseMatrix(
    i = rep(1:n_row, times = n_nbrs),
    j = as.vector(idx),
    x = as.vector(dist),
    dims = c(n_row, n_ref),
    repr = "C"
  ))
}

sparse_to_list_r <- function(spr) {
  list(row_ptr = spr@p, col_idx = spr@j, dist = spr@x)
}

sparse_to_list_c <- function(spc) {
  spct <- Matrix::t(spc)
  list(row_ptr = spct@p, col_idx = spct@i, dist = spct@x)
}

sparse_to_list <- function(sp) {
  if (methods::is(sp, "RsparseMatrix")) {
    sparse_to_list_r(sp)
  }
  else if (methods::is(sp, "CsparseMatrix")) {
    sparse_to_list_c(sp)
  }
  else {
    sparse_to_list_c(sparse_to_c(sp))
  }
}

list_to_sparse <- function(l) {
  Matrix::drop0(Matrix::sparseMatrix(
    p = l$row_ptr,
    j = l$col_idx,
    x = l$dist,
    dims = c(length(l$row_ptr) - 1, length(l$row_ptr) - 1),
    repr = "C",
    index1 = FALSE
  ))
}


graph_to_list <- function(graph) {
  sr <- graph_to_sparse_r(graph)
  sparse_to_list_r(sr)
}

sparse_to_r <- function(sp) {
  as(sp, "RsparseMatrix")
}

sparse_to_c <- function(sp) {
  as(sp, "CsparseMatrix")
}
