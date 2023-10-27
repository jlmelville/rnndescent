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

# called by rpf_knn and nnd_knn
rpf_knn_impl <-
  function(data,
           k,
           metric,
           use_alt_metric,
           actual_metric,
           n_trees,
           leaf_size,
           include_self,
           ret_forest,
           n_threads,
           verbose,
           zero_index = FALSE) {

    if (is.null(n_trees)) {
      # data is transposed at this point so n_obs is in the number of columns
      n_trees <- 5 + as.integer(round(ncol(data) ^ 0.25))
      n_trees <- min(32, n_trees)
    }
    if (is.null(leaf_size)) {
      leaf_size <- max(10, k)
    }

    tsmessage(
      thread_msg(
        "Calculating rp tree k-nearest neighbors with k = ",
        k,
        " n_trees = ",
        n_trees,
        " max leaf size = ",
        leaf_size,
        n_threads = n_threads
      )
    )

    res <- rp_tree_knn_cpp(
      data,
      k,
      actual_metric,
      n_trees = n_trees,
      leaf_size = leaf_size,
      include_self = include_self,
      ret_forest = ret_forest,
      unzero = !zero_index,
      n_threads = n_threads,
      verbose = verbose
    )

    if (use_alt_metric) {
      res$dist <- apply_alt_metric_correction(metric, res$dist)
    }
    tsmessage("Finished")
    res
  }

# reference and query are column-oriented
random_knn_impl <-
  function(reference,
           k,
           metric,
           use_alt_metric,
           actual_metric,
           order_by_distance,
           n_threads,
           verbose,
           query = NULL,
           zero_index = FALSE) {
    if (is.null(query)) {
      msg <- "Generating random k-nearest neighbor graph with k = "
      fun <- random_knn_cpp
      args <- list(data = reference)
    } else {
      msg <-
        "Generating random k-nearest neighbor graph from reference with k = "
      fun <- random_knn_query_cpp
      args <- list(reference = reference, query = query)
    }

    args <- lmerge(
      args,
      list(
        nnbrs = k,
        metric = actual_metric,
        order_by_distance = order_by_distance,
        n_threads = n_threads,
        verbose = verbose
      )
    )
    tsmessage(thread_msg(msg,
                         k,
                         n_threads = n_threads
    ))
    res <- do.call(fun, args)

    if (!zero_index) {
      res$idx <- res$idx + 1
    }
    if (use_alt_metric) {
      res$dist <- apply_alt_metric_correction(metric, res$dist)
    }
    tsmessage("Finished")
    res
  }
