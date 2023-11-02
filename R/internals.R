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
    nn$idx <- cbind(nn$idx, matrix(0, nrow(nn$idx), ncol = n_aug_nbrs))
    nn$dist <- cbind(nn$dist, matrix(NA, nrow(nn$idx), ncol = n_aug_nbrs))

    if (ncol(nn$idx) == k && ncol(nn$dist) == k) {
      return(nn)
    }
    stop("Still not right!")
  }

get_reference_graph_k <- function(reference_graph) {
  ncol(reference_graph$idx)
}

find_margin_method <- function(margin, metric) {
  margin <- match.arg(tolower(margin), c("auto", "explicit", "implicit"))
  if (margin %in% c("explicit", "implicit")) {
    return(margin)
  }
  switch(metric,
         bhamming = "implicit",
         "explicit")
}


is_sparse <- function(x) {
  methods::is(x, "sparseMatrix")
}

find_sparse_alt_metric <- function(metric) {
  switch(metric,
         euclidean = "l2sqr",
         cosine = "alternative-cosine",
         metric
  )
}

find_dense_alt_metric <- function(metric) {
  switch(metric,
         euclidean = "l2sqr",
         metric
  )
}

find_alt_metric <- function(metric, is_sparse = FALSE) {
  if (is_sparse) {
    find_sparse_alt_metric(metric)
  }
  else {
    find_dense_alt_metric(metric)
  }
}

apply_alt_metric_uncorrection <- function(metric, dist) {
  switch(metric,
    euclidean = dist * dist,
    dist
  )
}

apply_dense_alt_metric_correction <- function(metric, dist) {
  switch(metric,
         euclidean = sqrt(dist),
         dist
  )
}

apply_sparse_alt_metric_correction <- function(metric, dist) {
  switch(metric,
    euclidean = sqrt(dist),
    cosine = apply(dist, c(1, 2), sparse_correct_alternative_cosine),
    dist
  )
}

apply_alt_metric_correction <- function(metric, dist, is_sparse = FALSE) {
  if (is_sparse) {
    apply_sparse_alt_metric_correction(metric, dist)
  }
  else {
    apply_dense_alt_metric_correction(metric, dist)
  }
}

isclose <- function(a, b, rtol = 1.0e-5, atol = 1.0e-8) {
  diff <- abs(a - b)
  diff <= (atol + rtol * abs(b))
}

sparse_correct_alternative_cosine <- function(dist) {
  if (isclose(0.0, abs(dist), atol = 1e-7) || dist < 0.0) {
    0.0
  }
  else {
    1.0 - (2.0 ^ -dist)
  }
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
           margin = "auto",
           n_threads = 0,
           verbose = FALSE,
           zero_index = FALSE) {

    if (is.null(n_trees)) {
      # data is transposed at this point so n_obs is in the number of columns
      n_trees <- 5 + as.integer(round(ncol(data) ^ 0.25))
      n_trees <- min(32, n_trees)
    }
    if (is.null(leaf_size)) {
      leaf_size <- max(10, k)
    }

    margin <- find_margin_method(margin, metric)

    tsmessage(
      thread_msg(
        "Calculating rp tree k-nearest neighbors with k = ",
        k,
        " n_trees = ",
        n_trees,
        " max leaf size = ",
        leaf_size,
        " margin = '", margin, "'",
        n_threads = n_threads
      )
    )

    if (margin == "implicit") {
      res <- rp_tree_knn_cpp2(
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
    }
    else {
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
    }

    if (use_alt_metric) {
      res$dist <-
        apply_alt_metric_correction(metric, res$dist, methods::is(data, "sparseMatrix"))
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
      if (methods::is(reference, "sparseMatrix")) {
        fun <- random_knn_sparse
        args <- list(data = reference@x, ind = reference@i, ptr = reference@p,
                     nobs = ncol(reference), ndim = nrow(reference))
      }
      else {
        fun <- random_knn_cpp
        args <- list(data = reference)
      }
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
      res$dist <-
        apply_alt_metric_correction(metric, res$dist, methods::is(reference, "sparseMatrix"))
    }
    tsmessage("Finished")
    res
  }
