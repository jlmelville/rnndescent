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

# data and query must be column-oriented
# recalculate_distances if TRUE even if a distance matrix is present,
# recalculate distances from data/query and the indices
# If augment_low_k = TRUE then if the desired k is > the size of the input nn
# then add random neighbors to make up the difference
prepare_init_graph <-
  function(nn,
           k,
           data,
           query = NULL,
           metric = "euclidean",
           recalculate_distances = FALSE,
           augment_low_k = TRUE,
           n_threads = 0,
           verbose = FALSE) {
    if (is.matrix(nn)) {
      nn <- list(idx = nn)
    }
    ## nn is a list dist may be NULL

    # idx has too few or too many columns
    if (k != ncol(nn$idx)) {
      if (k > ncol(nn$idx)) {
        if (!augment_low_k) {
          stop("Not enough initial neighbors provided for k = ", k)
        }
      }
      else if (k < ncol(nn$dist)) {
        nn$idx <- nn$idx[, 1:k, drop = FALSE]
      }
      # else k == ncol and we need do nothing
    }

    if (!is.null(nn$dist) && !recalculate_distances) {
      # dist exists and we intend to use the existing values
      if (k > ncol(nn$dist)) {
        if (!augment_low_k) {
          stop("Not enough initial distances provided for k = ", k)
        }
      }
      else if (k < ncol(nn$dist)) {
        nn$dist <- nn$dist[, 1:k, drop = FALSE]
      }
      # otherwise k == ncol and we need do nothing
    }

    # ensure nn$dist is not NULL and has correct distances at this point
    if (is.null(nn$dist) || recalculate_distances) {
      tsmessage("Generating distances for initial indices")
      if (!is.null(query)) {
        nn <-
          rnn_idx_to_graph_query(
            reference = data,
            query = query,
            idx = nn$idx,
            metric = metric,
            n_threads = n_threads,
            verbose = verbose
          )
      } else {
        nn <-
          rnn_idx_to_graph_self(
            data = data,
            idx = nn$idx,
            metric = metric,
            n_threads = n_threads,
            verbose = verbose
          )
      }
    }

    # nn$idx and nn$dist exist and are either the right size or too big
    if (ncol(nn$idx) == k && ncol(nn$dist) == k) {
      return(nn)
    }

    n_aug_nbrs <- k - ncol(nn$idx)
    tsmessage("Augmenting input graph with ", n_aug_nbrs, " random neighbors")
    nn_aug <- random_knn_impl(
        query = query,
        reference = data,
        k = n_aug_nbrs,
        metric = metric,
        use_alt_metric = FALSE,
        actual_metric = metric,
        order_by_distance = FALSE,
        n_threads = n_threads,
        verbose = verbose
      )
    nn$idx <- cbind(nn$idx, nn_aug$idx)
    nn$dist <- cbind(nn$dist, nn_aug$dist)

    if (ncol(nn$idx) == k && ncol(nn$dist) == k) {
      return(nn)
    }
    stop("Still not right!")
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
