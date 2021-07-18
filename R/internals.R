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
    } else {
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
      } else {
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
