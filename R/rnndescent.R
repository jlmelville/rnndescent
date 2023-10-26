# kNN Construction --------------------------------------------------------

#' Calculate Exact Nearest Neighbors by Brute Force
#'
#' @param data Matrix of `n` items to generate neighbors for, with observations
#'   in the rows and features in the columns. Optionally, input can be passed
#'   with observations in the columns, by setting `obs = "C"`, which should be
#'   more efficient.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of:
#'   - `"euclidean"`.
#'   - `"l2sqr"` (squared Euclidean).
#'   - `"cosine"`.
#'   - `"cosine-preprocessing"`: cosine with preprocessing: this trades memory for a
#'   potential speed up during the distance calculation.It should give the
#'   same results as `cosine`, give or take minor numerical changes. Be aware
#'   that the distance between two identical items may not always give exactly
#'   zero with this method.
#'   - `"manhattan"`.
#'   - `"correlation"` (1 minus the Pearson correlation).
#'   - `"correlation-preprocess"`: `correlation` with preprocessing. This trades
#'   memory for a potential speed up during the distance calculation. It should
#'   give the same results as `correlation`, give or take minor numerical
#'   changes. Be aware that the distance between two identical items may not
#'   always give exactly zero with this method.
#'   - `"hamming"`.
#'   - `"bhamming"` (hamming on binary data with bitset internal memory
#'   optimization).
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return the nearest neighbor graph as a list containing:
#'   * `idx` an n by k matrix containing the nearest neighbor indices.
#'   * `dist` an n by k matrix containing the nearest neighbor distances.
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
                            verbose = FALSE,
                            obs = "R") {
  obs <- match.arg(toupper(obs), c("C", "R"))
  n_obs <- switch(obs,
    R = nrow,
    C = ncol,
    stop("Unknown obs type")
  )

  data <- x2m(data)
  check_k(k, n_obs(data))

  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
  } else {
    actual_metric <- metric
  }

  tsmessage(
    thread_msg(
      "Calculating brute force k-nearest neighbors with k = ",
      k,
      n_threads = n_threads
    )
  )

  if (obs == "R") {
    data <- t(data)
  }
  res <-
    rnn_brute_force(data,
      k,
      actual_metric,
      n_threads = n_threads,
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
#' @param data Matrix of `n` items to generate random neighbors for, with
#'   observations in the rows and features in the columns. Optionally, input can
#'   be passed with observations in the columns, by setting `obs = "C"`, which
#'   should be more efficient.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of:
#'   - `"euclidean"`.
#'   - `"l2sqr"` (squared Euclidean).
#'   - `"cosine"`.
#'   - `"cosine-preprocess"`: cosine with preprocessing: this trades memory for a
#'   potential speed up during the distance calculation.It should give the
#'   same results as `cosine`, give or take minor numerical changes. Be aware
#'   that the distance between two identical items may not always give exactly
#'   zero with this method.
#'   - `"manhattan"`.
#'   - `"correlation"` (1 minus the Pearson correlation).
#'   - `"correlation-preprocess"`: `correlation` with preprocessing. This trades
#'   memory for a potential speed up during the distance calculation. It should
#'   give the same results as `correlation`, give or take minor numerical
#'   changes. Be aware that the distance between two identical items may not
#'   always give exactly zero with this method.
#'   - `"hamming"`.
#'   - `"bhamming"` (hamming on binary data with bitset internal memory
#'   optimization).
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param order_by_distance If `TRUE` (the default), then results for each
#'   item are returned by increasing distance. If you don't need the results
#'   sorted, e.g. you are going to pass the results as initialization to another
#'   routine like [nnd_knn()], set this to `FALSE` to save a small amount of
#'   computational time.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return a random neighbor graph as a list containing:
#'   * `idx` an n by k matrix containing the nearest neighbor indices.
#'   * `dist` an n by k matrix containing the nearest neighbor distances.
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
           verbose = FALSE,
           obs = "R") {
    obs <- match.arg(toupper(obs), c("C", "R"))
    n_obs <- switch(obs,
      R = nrow,
      C = ncol,
      stop("Unknown obs type")
    )
    data <- x2m(data)
    check_k(k, n_obs(data))

    if (use_alt_metric) {
      actual_metric <- find_alt_metric(metric)
    } else {
      actual_metric <- metric
    }

    if (obs == "R") {
      data <- t(data)
    }
    random_knn_impl(
      reference = data,
      k = k,
      metric = metric,
      use_alt_metric = use_alt_metric,
      actual_metric = actual_metric,
      order_by_distance = order_by_distance,
      n_threads = n_threads,
      verbose = verbose
    )
  }

#' Find Nearest Neighbors and Distances
#'
#' @param data Matrix of `n` items to generate neighbors for, with observations
#'   in the rows and features in the columns. Optionally, input can be passed
#'   with observations in the columns, by setting `obs = "C"`, which should be
#'   more efficient.
#' @param k Number of nearest neighbors to return. Optional if `init` is
#'   specified.
#' @param metric Type of distance calculation to use. One of:
#'   - `"euclidean"`.
#'   - `"l2sqr"` (squared Euclidean).
#'   - `"cosine"`.
#'   - `"cosine-preprocess"`: cosine with preprocessing: this trades memory for a
#'   potential speed up during the distance calculation.It should give the
#'   same results as `cosine`, give or take minor numerical changes. Be aware
#'   that the distance between two identical items may not always give exactly
#'   zero with this method.
#'   - `"manhattan"`.
#'   - `"correlation"` (1 minus the Pearson correlation).
#'   - `"correlation-preprocess"`: `correlation` with preprocessing. This trades
#'   memory for a potential speed up during the distance calculation. It should
#'   give the same results as `correlation`, give or take minor numerical
#'   changes. Be aware that the distance between two identical items may not
#'   always give exactly zero with this method.
#'   - `"hamming"`.
#'   - `"bhamming"` (hamming on binary data with bitset internal memory
#'   optimization).
#' @param init Name of the initialization strategy or initial `data` neighbor
#'   graph to optimize. One of:
#'   - `"rand"` random initialization (the default).
#'   - `"tree"` use the random projection tree method of Dasgupta and Freund
#'   (2008).
#'   - a pre-calculated neighbor graph. A list containing:
#'       - `idx` an `n` by `k` matrix containing the nearest neighbor indices.
#'       - `dist` (optional) an `n` by `k` matrix containing the nearest
#'       neighbor distances. If the input distances are omitted, they will be
#'       calculated for you.'
#'
#'   If `k` and `init` are specified as arguments to this function, and the
#'   number of neighbors provided in `init` is not equal to `k` then:
#'
#'   * if `k` is smaller, only the `k` closest values in `init` are retained.
#'   * if `k` is larger, then random neighbors will be chosen to fill `init` to
#'   the size of `k`. Note that there is no checking if any of the random
#'   neighbors are duplicates of what is already in `init` so effectively fewer
#'   than `k` neighbors may be chosen for some observations under these
#'   circumstances.
#'
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item in each iteration. Use relative to `k` to emulate the "rho"
#'   sampling parameter in the nearest neighbor descent paper. By default, this
#'   is set to `k` or `60`, whichever is smaller.
#' @param delta The minimum relative change in the neighbor graph allowed before
#'   early stopping. Should be a value between 0 and 1. The smaller the value,
#'   the smaller the amount of progress between iterations is allowed. Default
#'   value of `0.001` means that at least 0.1% of the neighbor graph must
#'   be updated at each iteration.
#' @param low_memory If `TRUE`, use a lower memory, but more
#'   computationally expensive approach to index construction. If set to
#'   `FALSE`, you should see a noticeable speed improvement, especially
#'   when using a smaller number of threads, so this is worth trying if you have
#'   the memory to spare.
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param progress Determines the type of progress information logged if
#'   `verbose = TRUE`. Options are:
#'   * `"bar"`: a simple text progress bar.
#'   * `"dist"`: the sum of the distances in the approximate knn graph at the
#'     end of each iteration.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return the approximate nearest neighbor graph as a list containing:
#'   * `idx` an n by k matrix containing the nearest neighbor indices.
#'   * `dist` an n by k matrix containing the nearest neighbor distances.
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
#' Dasgupta, S., & Freund, Y. (2008, May).
#' Random projection trees and low dimensional manifolds.
#' In *Proceedings of the fortieth annual ACM symposium on Theory of computing*
#' (pp. 537-546).
#' <https://doi.org/10.1145/1374376.1374452>.
#'
#' Dong, W., Moses, C., & Li, K. (2011, March).
#' Efficient k-nearest neighbor graph construction for generic similarity measures.
#' In *Proceedings of the 20th international conference on World Wide Web*
#' (pp. 577-586).
#' ACM.
#' <https://doi.org/10.1145/1963405.1963487>.
#' @export
nnd_knn <- function(data,
                    k = NULL,
                    metric = "euclidean",
                    init = "rand",
                    n_iters = 10,
                    max_candidates = NULL,
                    delta = 0.001,
                    low_memory = TRUE,
                    use_alt_metric = TRUE,
                    n_threads = 0,
                    verbose = FALSE,
                    progress = "bar",
                    obs = "R") {
  stopifnot(tolower(progress) %in% c("bar", "dist"))
  obs <- match.arg(toupper(obs), c("C", "R"))

  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
    if (!is.null(init) && is.list(init) && !is.null(init$dist)) {
      init$dist <- apply_alt_metric_uncorrection(metric, init$dist)
    }
  } else {
    actual_metric <- metric
  }

  data <- x2m(data)
  if (obs == "R") {
    data <- t(data)
  }

  # data must be column-oriented at this point
  if (is.character(init)) {
    if (is.null(k)) {
      stop("Must provide k")
    }
    check_k(k, ncol(data))
    init <- match.arg(tolower(init), c("rand", "tree"))

    tsmessage("Initializing neighbors using '", init, "' method")
    init <- switch(
      init,
      "rand" = random_knn_impl(
        reference = data,
        k = k,
        metric = actual_metric,
        use_alt_metric = FALSE,
        actual_metric = actual_metric,
        order_by_distance = FALSE,
        n_threads = n_threads,
        verbose = verbose
      ),
      "tree" = rpf_knn_impl(
        data,
        k = k,
        metric = actual_metric,
        use_alt_metric = FALSE,
        actual_metric = actual_metric,
        n_trees = NULL,
        leaf_size = NULL,
        include_self = FALSE,
        n_threads = n_threads,
        verbose = verbose,
      ),
      stop("Unknown initialization option '", init, "'")
    )
  } else {
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
      verbose = verbose
    )

  if (is.null(max_candidates)) {
    max_candidates <- min(k, 60)
  }
  tsmessage(
    thread_msg(
      "Running nearest neighbor descent for ",
      n_iters,
      " iterations",
      n_threads = n_threads
    )
  )
  res <- nn_descent(
    data,
    init$idx,
    init$dist,
    metric = actual_metric,
    n_iters = n_iters,
    max_candidates = max_candidates,
    delta = delta,
    low_memory = low_memory,
    n_threads = n_threads,
    verbose = verbose,
    progress_type = progress
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
#' @param query Matrix of `n` query items, with observations in the rows and
#'   features in the columns. Optionally, the data may be passed with the
#'   observations in the columns, by setting `obs = "C"`, which should be more
#'   efficient. The `reference` data must be passed in the same orientation as
#'   `query`.
#' @param reference Matrix of `m` reference items, with observations in the rows
#'   and features in the columns. The nearest neighbors to the queries are
#'   calculated from this data. Optionally, the data may be passed with the
#'   observations in the columns, by setting `obs = "C"`, which should be more
#'   efficient. The `query` data must be passed in the same orientation as
#'   `reference`.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of:
#'   - `"euclidean"`.
#'   - `"l2sqr"` (squared Euclidean).
#'   - `"cosine"`.
#'   - `"cosine-preprocess"`: cosine with preprocessing: this trades memory for a
#'   potential speed up during the distance calculation.It should give the
#'   same results as `cosine`, give or take minor numerical changes. Be aware
#'   that the distance between two identical items may not always give exactly
#'   zero with this method.
#'   - `"manhattan"`.
#'   - `"correlation"` (1 minus the Pearson correlation).
#'   - `"correlation-preprocess"`: `correlation` with preprocessing. This trades
#'   memory for a potential speed up during the distance calculation. It should
#'   give the same results as `correlation`, give or take minor numerical
#'   changes. Be aware that the distance between two identical items may not
#'   always give exactly zero with this method.
#'   - `"hamming"`.
#'   - `"bhamming"` (hamming on binary data with bitset internal memory
#'   optimization).
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `query` and `reference`
#'   orientation stores each observation as a column (the orientation must be
#'   consistent). The default `"R"` means that observations are stored in each
#'   row. Storing the data by row is usually more convenient, but internally
#'   your data will be converted to column storage. Passing it already
#'   column-oriented will save some memory and (a small amount of) CPU usage.
#' @return the nearest neighbor graph as a list containing:
#'   * `idx` an n by k matrix containing the nearest neighbor indices in
#'   `reference`.
#'   * `dist` an n by k matrix containing the nearest neighbor distances to the
#'   items in `reference`.
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
                                  verbose = FALSE,
                                  obs = "R") {
  obs <- match.arg(toupper(obs), c("C", "R"))

  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
  } else {
    actual_metric <- metric
  }

  reference <- x2m(reference)
  query <- x2m(query)
  if (obs == "R") {
    reference <- t(reference)
    query <- t(query)
  }
  check_k(k, ncol(reference))

  tsmessage(
    thread_msg(
      "Calculating brute force k-nearest neighbors from reference with k = ",
      k,
      n_threads = n_threads
    )
  )
  res <- rnn_brute_force_query(reference,
    query,
    k,
    actual_metric,
    n_threads = n_threads,
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
#' @param query Matrix of `n` query items, with observations in the rows and
#'   features in the columns. Optionally, the data may be passed with the
#'   observations in the columns, by setting `obs = "C"`, which should be more
#'   efficient. The `reference` data must be passed in the same orientation as
#'   `query`.
#' @param reference Matrix of `m` reference items, with observations in the rows
#'   and features in the columns. The nearest neighbors to the queries are
#'   randomly selected from this data. Optionally, the data may be passed with
#'   the observations in the columns, by setting `obs = "C"`, which should be
#'   more efficient. The `query` data must be passed in the same orientation as
#'   `reference`.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of:
#'   - `"euclidean"`.
#'   - `"l2sqr"` (squared Euclidean).
#'   - `"cosine"`.
#'   - `"cosine-preprocess"`: cosine with preprocessing: this trades memory for a
#'   potential speed up during the distance calculation.It should give the
#'   same results as `cosine`, give or take minor numerical changes. Be aware
#'   that the distance between two identical items may not always give exactly
#'   zero with this method.
#'   - `"manhattan"`.
#'   - `"correlation"` (1 minus the Pearson correlation).
#'   - `"correlation-preprocess"`: `correlation` with preprocessing. This trades
#'   memory for a potential speed up during the distance calculation. It should
#'   give the same results as `correlation`, give or take minor numerical
#'   changes. Be aware that the distance between two identical items may not
#'   always give exactly zero with this method.
#'   - `"hamming"`.
#'   - `"bhamming"` (hamming on binary data with bitset internal memory
#'   optimization).
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param order_by_distance If `TRUE` (the default), then results for each
#'   item are returned by increasing distance. If you don't need the results
#'   sorted, e.g. you are going to pass the results as initialization to another
#'   routine like [graph_knn_query()], set this to `FALSE` to save a
#'   small amount of computational time.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `query` and `reference`
#'   orientation stores each observation as a column (the orientation must be
#'   consistent). The default `"R"` means that observations are stored in each
#'   row. Storing the data by row is usually more convenient, but internally
#'   your data will be converted to column storage. Passing it already
#'   column-oriented will save some memory and (a small amount of) CPU usage.
#' @return an approximate nearest neighbor graph as a list containing:
#'   * `idx` an n by k matrix containing the nearest neighbor indices.
#'   * `dist` an n by k matrix containing the nearest neighbor distances.
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
           verbose = FALSE,
           obs = "R") {
    obs <- match.arg(toupper(obs), c("C", "R"))
    n_obs <- switch(obs,
      R = nrow,
      C = ncol,
      stop("Unknown obs type")
    )
    reference <- x2m(reference)
    query <- x2m(query)
    check_k(k, n_obs(reference))

    if (use_alt_metric) {
      actual_metric <- find_alt_metric(metric)
    } else {
      actual_metric <- metric
    }

    if (obs == "R") {
      reference <- t(reference)
      query <- t(query)
    }

    random_knn_impl(
      reference = reference,
      query = query,
      k = k,
      metric = metric,
      use_alt_metric = use_alt_metric,
      actual_metric = actual_metric,
      order_by_distance = order_by_distance,
      n_threads = n_threads,
      verbose = verbose
    )
  }

#' Find Nearest Neighbors and Distances
#'
#' @param query Matrix of `n` query items, with observations in the rows and
#'   features in the columns. Optionally, the data may be passed with the
#'   observations in the columns, by setting `obs = "C"`, which should be more
#'   efficient. The `reference` data must be passed in the same orientation as
#'   `query`.
#' @param reference Matrix of `m` reference items, with observations in the rows
#'   and features in the columns. The nearest neighbors to the queries are
#'   calculated from this data. Optionally, the data may be passed with the
#'   observations in the columns, by setting `obs = "C"`, which should be more
#'   efficient. The `query` data must be passed in the same orientation as
#'   `reference`.
#' @param reference_graph Search graph of the `reference` data. A neighbor
#'   graph, such as that output from [nnd_knn()] can be used, but
#'   preferably a suitably prepared sparse search graph should be used, such as
#'   that output by [prepare_search_graph()].
#' @param k Number of nearest neighbors to return. Optional if `init` is
#'   specified.
#' @param metric Type of distance calculation to use. One of `"euclidean"`,
#'   `"l2sqr"` (squared Euclidean), `"cosine"`, `"manhattan"`, `"correlation"`
#'   (1 minus the Pearson correlation), `"hamming"` or `"bhamming"` (hamming
#'   on binary data with bitset internal memory optimization).
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param init Initial `query` neighbor graph to optimize. If not provided, `k`
#'   random neighbors are created. If provided, the input format should be a
#'   list containing:
#'
#'   * `idx` an `n` by `k` matrix containing the nearest neighbor indices.
#'   * `dist` (optional) an `n` by `k` matrix containing the nearest neighbor
#'   distances.
#'
#'   If `k` and `init` are specified as arguments to this function, and the
#'   number of neighbors provided in `init` is not equal to `k` then:
#'
#'   * if `k` is smaller, only the `k` closest values in `init` are retained.
#'   * if `k` is larger, then random neighbors will be chosen to fill `init` to
#'   the size of `k`. Note that there is no checking if any of the random
#'   neighbors are duplicates of what is already in `init` so effectively fewer
#'   than `k` neighbors may be chosen for some observations under these
#'   circumstances.
#'
#'   If the input distances are omitted, they will be calculated for you.
#' @param epsilon Controls trade-off between accuracy and search cost, by
#'   specifying a distance tolerance on whether to explore the neighbors of
#'   candidate points. The larger the value, the more neighbors will be
#'   searched. A value of 0.1 allows query-candidate distances to be 10% larger
#'   than the current most-distant neighbor of the query point, 0.2 means 20%,
#'   and so on. Suggested values are between 0-0.5, although this value is
#'   highly dependent on the distribution of distances in the dataset (higher
#'   dimensional data should choose a smaller cutoff). Too large a value of
#'   `epsilon` will result in the query search approaching brute force
#'   comparison. Use this parameter in conjunction with
#'   [prepare_search_graph()] to prevent excessive run time. Default is 0.1.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `query` and `reference`
#'   orientation stores each observation as a column (the orientation must be
#'   consistent). The default `"R"` means that observations are stored in each
#'   row. Storing the data by row is usually more convenient, but internally
#'   your data will be converted to column storage. Passing it already
#'   column-oriented will save some memory and (a small amount of) CPU usage.
#' @return the approximate nearest neighbor graph as a list containing:
#'   * `idx` a `n` by `k` matrix containing the nearest neighbor indices
#'     specifying the row of the neighbor in `reference`.
#'   * `dist` a `n` by `k` matrix containing the nearest neighbor distances.
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
#' iris_query_nn <- graph_knn_query(iris_query, iris_ref, iris_ref_graph,
#'   k = 4, metric = "euclidean", verbose = TRUE
#' )
#' @references
#' Hajebi, K., Abbasi-Yadkori, Y., Shahbazi, H., & Zhang, H. (2011, June).
#' Fast approximate nearest-neighbor search with k-nearest neighbor graph.
#' In *Twenty-Second International Joint Conference on Artificial Intelligence*.
#'
#' Harwood, B., & Drummond, T. (2016).
#' Fanng: Fast approximate nearest neighbour graphs.
#' In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
#' (pp. 5713-5722).
#'
#' Iwasaki, M., & Miyazaki, D. (2018).
#' Optimization of indexing based on k-nearest neighbor graph for proximity
#' search in high-dimensional data.
#' *arXiv preprint arXiv:1810.07355*.
#'
#' @export
graph_knn_query <- function(query,
                            reference,
                            reference_graph,
                            k = NULL,
                            metric = "euclidean",
                            init = NULL,
                            epsilon = 0.1,
                            use_alt_metric = TRUE,
                            n_threads = 0,
                            verbose = FALSE,
                            obs = "R") {
  obs <- match.arg(toupper(obs), c("C", "R"))

  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
    if (!is.null(init) && !is.null(init$dist)) {
      init$dist <- apply_alt_metric_uncorrection(metric, init$dist)
    }
  } else {
    actual_metric <- metric
  }

  reference <- x2m(reference)
  query <- x2m(query)
  if (obs == "R") {
    reference <- t(reference)
    query <- t(query)
  }

  # reference and query must be column-oriented at this point
  if (is.null(init)) {
    if (is.null(k)) {
      if (is.list(reference_graph)) {
        k <- get_reference_graph_k(reference_graph)
        tsmessage("Using k = ", k, " from graph")
      } else {
        stop("Must provide k")
      }
    }
    check_k(k, ncol(reference))
    tsmessage("Initializing from random neighbors")
    init <- random_knn_impl(
      query = query,
      reference = reference,
      k = k,
      metric = actual_metric,
      use_alt_metric = FALSE,
      actual_metric = actual_metric,
      order_by_distance = FALSE,
      n_threads = n_threads,
      verbose = verbose
    )
  } else {
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
      verbose = verbose
    )

  stopifnot(!is.null(query), methods::is(query, "matrix"))
  stopifnot(
    !is.null(init$idx),
    methods::is(init$idx, "matrix"),
    ncol(init$idx) == k,
    nrow(init$idx) == ncol(query)
  )
  stopifnot(
    !is.null(init$dist),
    methods::is(init$dist, "matrix"),
    ncol(init$dist) == k,
    nrow(init$dist) == ncol(query)
  )

  if (is.list(reference_graph)) {
    reference_dist <- reference_graph$dist
    reference_idx <- reference_graph$idx
    stopifnot(!is.null(reference), methods::is(reference, "matrix"))
    stopifnot(
      !is.null(reference_idx),
      methods::is(reference_idx, "matrix"),
      nrow(reference_idx) == ncol(reference)
    )
    stopifnot(
      !is.null(reference_dist),
      methods::is(reference_dist, "matrix"),
      nrow(reference_dist) == ncol(reference)
    )
    reference_graph_list <- graph_to_list(reference_graph)
  } else {
    stopifnot(methods::is(reference_graph, "sparseMatrix"))
    reference_graph_list <- csparse_to_list(reference_graph)
  }

  tsmessage(thread_msg("Searching nearest neighbor graph", n_threads = n_threads))
  res <-
    nn_query(
      reference = reference,
      reference_graph_list = reference_graph_list,
      query = query,
      nn_idx = init$idx,
      nn_dist = init$dist,
      metric = actual_metric,
      epsilon = epsilon,
      n_threads = n_threads,
      verbose = verbose
    )
  if (use_alt_metric) {
    res$dist <- apply_alt_metric_correction(metric, res$dist)
  }
  tsmessage("Finished")
  res
}

# Search Graph Preparation ------------------------------------------------

#' Nearest Neighbor Graph Refinement
#'
#' Create a graph using existing nearest neighbor data to balance search
#' speed and accuracy using the occlusion pruning and truncation strategies
#' of Harwood and Drummond (2016).
#'
#' An approximate nearest neighbor graph is not very useful for querying via
#' [graph_knn_query()], especially if the query data is initialized randomly:
#' some items in the data set may not be in the nearest neighbor list of any
#' other item and can therefore never be returned as a neighbor, no matter how
#' close they are to the query. Even those which do appear in at least one
#' neighbor list may not be reachable by expanding an arbitrary starting list if
#' the neighbor graph contains disconnected components.
#'
#' Converting the directed graph represented by the neighbor graph to an
#' undirected graph by adding an edge from item `j` to `i` if
#' an edge exists from `i` to `j` (i.e. creating the mutual neighbor
#' graph) solves the problems above, but can result in inefficient searches.
#' Although the out-degree of each item is restricted to the number of neighbors
#' the in-degree has no such restrictions: a given item could be very "popular"
#' and in a large number of neighbors lists. Therefore mutualizing the neighbor
#' graph can result in some items with a large number of neighbors to search.
#' These usually have very similar neighborhoods so there is nothing to be
#' gained from searching all of them.
#'
#' To balance accuracy and search time, the following procedure is carried out:
#'
#' 1. The graph is "diversified" by occlusion pruning.
#' 1. The reverse graph is formed by reversing the direction of all edges in
#' the pruned graph.
#' 1. The reverse graph is diversified by occlusion pruning.
#' 1. The pruned forward and pruned reverse graph are merged.
#' 1. The outdegree of each node in the merged graph is truncated.
#' 1. The truncated merged graph is returned as the prepared search graph.
#'
#' Explicit zero distances in the `graph` will be converted to a small positive
#' number to avoid being dropped in the sparse representation. The one exception
#' is the "self" distance, i.e. any edge in the `graph` which links a node to
#' itself (the diagonal of the sparse distance matrix). These trivial edges
#' aren't useful for search purposes and are always dropped.
#'
#' @param data Matrix of `n` items, with observations in the rows and features
#'   in the columns. Optionally, input can be passed with observations in the
#'   columns, by setting `obs = "C"`, which should be more efficient.
#' @param graph neighbor graph for `data`, a list containing:
#'   * `idx` an `n` by `k` matrix containing the nearest neighbor indices of
#'   the data in `data`.
#'   * `dist` an `n` by `k` matrix containing the nearest neighbor distances.
#' @param metric Type of distance calculation to use. One of:
#'   - `"euclidean"`.
#'   - `"l2sqr"` (squared Euclidean).
#'   - `"cosine"`.
#'   - `"cosine-preprocess"`: cosine with preprocessing: this trades memory for a
#'   potential speed up during the distance calculation.It should give the
#'   same results as `cosine`, give or take minor numerical changes. Be aware
#'   that the distance between two identical items may not always give exactly
#'   zero with this method.
#'   - `"manhattan"`.
#'   - `"correlation"` (1 minus the Pearson correlation).
#'   - `"correlation-preprocess"`: `correlation` with preprocessing. This trades
#'   memory for a potential speed up during the distance calculation. It should
#'   give the same results as `correlation`, give or take minor numerical
#'   changes. Be aware that the distance between two identical items may not
#'   always give exactly zero with this method.
#'   - `"hamming"`.
#'   - `"bhamming"` (hamming on binary data with bitset internal memory
#'   optimization).
#' @param diversify_prob the degree of diversification of the search graph
#'   by removing unnecessary edges through occlusion pruning. This should take a
#'   value between `0` (no diversification) and `1` (remove as many edges as
#'   possible) and is treated as the probability of a neighbor being removed if
#'   it is found to be an "occlusion". If item `p` and `q`, two members of the
#'   neighbor list of item `i`, are closer to each other than they are to `i`,
#'   then the nearer neighbor `p` is said to "occlude" `q`. It is likely that
#'   `q` will be in the neighbor list of `p` so there is no need to retain it in
#'   the neighbor list of `i`. You may also set this to `NULL` to skip any
#'   occlusion pruning. Note that occlusion pruning is carried out twice, once
#'   to the forward neighbors, and once to the reverse neighbors.
#' @param pruning_degree_multiplier How strongly to truncate the final neighbor
#'   list for each item. The neighbor list of each item will be truncated to
#'   retain only the closest `d` neighbors, where
#'   `d = k * pruning_degree_multiplier`, and `k` is the
#'   original number of neighbors per item in `graph`. Roughly, values
#'   larger than `1` will keep all the nearest neighbors of an item, plus
#'   the given fraction of reverse neighbors (if they exist). For example,
#'   setting this to `1.5` will keep all the forward neighbors and then
#'   half as many of the reverse neighbors, although exactly which neighbors are
#'   retained is also dependent on any occlusion pruning that occurs. Set this
#'   to `NULL` to skip this step.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return a search graph for `data` based on `graph`, represented as a sparse
#'   matrix, suitable for use with [graph_knn_query()].
#' @examples
#' # 100 reference iris items
#' iris_ref <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#'
#' # 50 query items
#' iris_query <- iris[iris$Species == "versicolor", ]
#'
#' # First, find the approximate 4-nearest neighbor graph for the references:
#' ref_ann_graph <- nnd_knn(iris_ref, k = 4)
#'
#' # Create a graph for querying with
#' ref_search_graph <- prepare_search_graph(iris_ref, ref_ann_graph)
#'
#' # Using the search graph rather than the ref_ann_graph directly may give
#' # more accurate or faster results
#' iris_query_nn <- graph_knn_query(
#'   query = iris_query, reference = iris_ref,
#'   reference_graph = ref_search_graph, k = 4, metric = "euclidean",
#'   verbose = TRUE
#' )
#' @references
#' Harwood, B., & Drummond, T. (2016).
#' Fanng: Fast approximate nearest neighbour graphs.
#' In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
#' (pp. 5713-5722).
#' @export
prepare_search_graph <- function(data,
                                 graph,
                                 metric = "euclidean",
                                 diversify_prob = 1.0,
                                 pruning_degree_multiplier = 1.5,
                                 n_threads = 0,
                                 verbose = FALSE,
                                 obs = "R") {
  obs <- match.arg(toupper(obs), c("C", "R"))

  if (!is.null(pruning_degree_multiplier)) {
    stopifnot(pruning_degree_multiplier > 0)
  }
  if (!is.null(diversify_prob)) {
    stopifnot(
      diversify_prob <= 1,
      diversify_prob >= 0
    )
  }
  n_nbrs <- check_graph(graph)$k
  max_degree <- max(round(n_nbrs * pruning_degree_multiplier), 1)

  tsmessage("Converting graph to sparse format")
  sp <- graph_to_csparse(graph)

  sp <- preserve_zeros(sp)

  data <- x2m(data)

  if (obs == "R") {
    data <- t(data)
  }

  if (!is.null(diversify_prob) && diversify_prob > 0) {
    tsmessage("Diversifying forward graph")
    fdiv <- diversify(
      data,
      sp,
      metric = metric,
      prune_probability = diversify_prob,
      verbose = verbose,
      n_threads = n_threads
    )
  } else {
    fdiv <- sp
    tsmessage(
      "Forward graph has # edges = ",
      Matrix::nnzero(fdiv),
      " (",
      formatC(100 * nn_sparsity_sp(fdiv)),
      "% sparse)"
    )
  }
  rsp <- reverse_knn_sp(fdiv)
  if (!is.null(diversify_prob) && diversify_prob > 0) {
    tsmessage("Diversifying reverse graph")
    rdiv <- diversify(
      data,
      rsp,
      metric = metric,
      prune_probability = diversify_prob,
      verbose = verbose,
      n_threads = n_threads
    )
  } else {
    rdiv <- rsp
    tsmessage(
      "Reverse graph has # edges = ",
      Matrix::nnzero(rdiv),
      " (",
      formatC(100 * nn_sparsity_sp(rdiv)),
      "% sparse)"
    )
  }
  tsmessage("Merging diversified forward and reverse graph")
  merged <- merge_graphs_sp(fdiv, rdiv)

  if (!is.null(pruning_degree_multiplier) &&
    !is.infinite(pruning_degree_multiplier)) {
    max_degree <- round(n_nbrs * pruning_degree_multiplier)
    tsmessage("Degree pruning merged graph to max degree: ", max_degree)
    res <-
      degree_prune(
        merged,
        max_degree = max_degree,
        verbose = verbose,
        n_threads = n_threads
      )
  } else {
    res <- merged
    tsmessage(
      "Merged graph has # edges = ",
      Matrix::nnzero(res),
      " (",
      formatC(100 * nn_sparsity_sp(res)),
      "% sparse)"
    )
  }
  tsmessage("Finished preparing search graph")
  res
}

# Harwood, B., & Drummond, T. (2016).
# Fanng: Fast approximate nearest neighbour graphs.
# In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
# (pp. 5713-5722).
# "Occlusion pruning"
diversify <- function(data,
                      graph,
                      metric = "euclidean",
                      prune_probability = 1.0,
                      n_threads = 0,
                      verbose = FALSE) {
  nnz_before <- Matrix::nnzero(graph)
  sp_before <- nn_sparsity_sp(graph)
  stopifnot(methods::is(graph, "sparseMatrix"))
  gl <- csparse_to_list(graph)

  gl_div <- diversify_cpp(
    data = x2m(data),
    graph_list = gl,
    metric = metric,
    prune_probability = prune_probability,
    n_threads = n_threads
  )
  res <- list_to_sparse(gl_div)
  nnz_after <- Matrix::nnzero(res)
  tsmessage(
    "Diversifying reduced # edges from ",
    nnz_before,
    " to ",
    nnz_after,
    " (",
    formatC(100 * sp_before),
    "% to ",
    formatC(100 * nn_sparsity_sp(res)),
    "% sparse)"
  )
  res
}

# FANNG: Fast Approximate Nearest Neighbour Graphs
# "truncating"
# Harwood, B., & Drummond, T. (2016).
# Fanng: Fast approximate nearest neighbour graphs.
# In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
# (pp. 5713-5722).
degree_prune <-
  function(graph,
           max_degree = 20,
           n_threads = 0,
           verbose = FALSE) {
    stopifnot(methods::is(graph, "sparseMatrix"))
    nnz_before <- Matrix::nnzero(graph)
    sp_before <- nn_sparsity_sp(graph)
    gl <- csparse_to_list(graph)

    gl_div <-
      degree_prune_cpp(gl, max_degree, n_threads = n_threads)
    res <- list_to_sparse(gl_div)
    nnz_after <- Matrix::nnzero(res)
    tsmessage(
      "Degree pruning to max ",
      max_degree,
      " reduced # edges from ",
      nnz_before,
      " to ",
      nnz_after,
      " (",
      formatC(100 * sp_before),
      "% to ",
      formatC(100 * nn_sparsity_sp(res)),
      "% sparse)"
    )
    res
  }


merge_graphs_sp <- function(g1, g2) {
  gl1 <- csparse_to_list(g1)
  gl2 <- csparse_to_list(g2)

  gl_merge <- merge_graph_lists_cpp(gl1, gl2)
  list_to_sparse(gl_merge)
}

nn_sparsity <- function(graph) {
  sum(graph$idx == 0) / prod(dim(graph$idx))
}

nn_sparsity_sp <- function(graph) {
  Matrix::nnzero(graph) / prod(graph@Dim)
}

reverse_knn_sp <- function(graph) {
  stopifnot(methods::is(graph, "sparseMatrix"))
  Matrix::t(Matrix::drop0(graph))
}

# Merge -------------------------------------------------------------------

#' Merge two approximate nearest neighbors graphs
#'
#' @param nn_graph1 A nearest neighbor graph to merge. Should consist of a list
#'   containing:
#'   * `idx` an n by k matrix containing the k nearest neighbor indices.
#'   * `dist` an n by k matrix containing k nearest neighbor distances.
#' @param nn_graph2 Another nearest neighbor graph to merge with the same
#'   format as `nn_graph1`. The number of neighbors can differ between
#'   graphs, but the merged result will have the same number of neighbors as
#'   specified in `nn_graph1`.
#' @param is_query If `TRUE` then the graphs are treated as the result of a knn
#'   query, not a knn building process. Or: is the graph bipartite? This should
#'   be set to `TRUE` if `nn_graphs` are the results of using e.g.
#'   [graph_knn_query()] or [random_knn_query()], and set to `FALSE` if these
#'   are the results of [nnd_knn()] or [random_knn()]. The difference is that if
#'   `is_query = FALSE`, if an index `p` is found in `nn_graph1[i, ]`, i.e. `p`
#'   is a neighbor of `i` with distance `d`, then it is assumed that `i` is a
#'   neighbor of `p` with the same distance. If `is_query = TRUE`, then `i` and
#'   `p` are indexes into two different datasets and the symmetry does not hold.
#'   If you aren't sure what case applies to you, it's safe (but potentially
#'   inefficient) to set `is_query = TRUE`
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @return a list containing:
#'   * `idx` an n by k matrix containing the merged nearest neighbor
#'   indices.
#'   * `dist` an n by k matrix containing the merged nearest neighbor
#'    distances.
#'
#'   The size of `k` in the output graph is the same as that of
#'   `nn_graph1`.
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
                      verbose = FALSE) {
  validate_are_mergeable(nn_graph1, nn_graph2)

  merge_nn(
    nn_graph1$idx,
    nn_graph1$dist,
    nn_graph2$idx,
    nn_graph2$dist,
    is_query,
    n_threads = n_threads,
    verbose = verbose
  )
}

#' Merge a list of approximate nearest neighbors graphs
#'
#' @param nn_graphs A list of nearest neighbor graph to merge. Each item in the
#'   list should consist of a sub-list
#'   containing:
#'   * `idx` an n by k matrix containing the k nearest neighbor indices.
#'   * `dist` an n by k matrix containing k nearest neighbor distances.
#'   The number of neighbors can differ between graphs, but the merged result
#'   will have the same number of neighbors as the first graph in the list.
#' @param is_query If `TRUE` then the graphs are treated as the result of a knn
#'   query, not a knn building process. Or: is the graph bipartite? This should
#'   be set to `TRUE` if `nn_graphs` are the results of using e.g.
#'   [graph_knn_query()] or [random_knn_query()], and set to `FALSE` if these
#'   are the results of [nnd_knn()] or [random_knn()]. The difference is that if
#'   `is_query = FALSE`, if an index `p` is found in `nn_graph1[i, ]`, i.e. `p`
#'   is a neighbor of `i` with distance `d`, then it is assumed that `i` is a
#'   neighbor of `p` with the same distance. If `is_query = TRUE`, then `i` and
#'   `p` are indexes into two different datasets and the symmetry does not hold.
#'   If you aren't sure what case applies to you, it's safe (but potentially
#'   inefficient) to set `is_query = TRUE`.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @return a list containing:
#'   * `idx` an n by k matrix containing the merged nearest neighbor indices.
#'   * `dist` an n by k matrix containing the merged nearest neighbor distances.
#'
#'   The size of `k` in the output graph is the same as that of the first
#'   item in `nn_graphs`.
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
                       verbose = FALSE) {
  if (length(nn_graphs) == 0) {
    return(list())
  }
  validate_are_mergeablel(nn_graphs)

  merge_nn_all(nn_graphs,
    is_query,
    n_threads = n_threads,
    verbose = verbose
  )
}
