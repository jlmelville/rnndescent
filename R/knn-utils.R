#' Manipulate nearest neighbor graphs
#'
#' Truncate or reorder the neighbors in a graph, or adjust self neighbors in a
#' graph of a dataset against itself. Each function returns a list containing
#' only `idx` and `dist` matrices.
#'
#' `truncate_knn()` keeps the first `k` columns. `sort_knn()` orders each row
#' by ascending distance, or by the supplied score matrix. Ties retain their
#' original order. Missing scores follow other scores, and missing neighbors
#' (`idx = 0`) always come last. Distances remain paired with their indices.
#'
#' `remove_self_knn()` keeps the first `k` present, nonself neighbors in each
#' row (by default, one fewer than the stored width, with a minimum of zero).
#' `self_first_knn()` puts self first, retaining the distance of the first
#' existing self entry, and follows it with the first `k - 1` present, nonself
#' neighbors. If self is absent, supply `self_dist` to insert it. Both functions
#' identify self by the row number, skip missing entries and other self entries,
#' and error if a row has too few nonself neighbors. They are for self graphs.
#' Do not use them on query results, where row numbers identify queries rather
#' than reference observations.
#'
#' @param graph A nearest neighbor graph with `idx` and `dist` matrices, such as
#'   the result of [brute_force_knn()].
#' @param k Number of columns to retain. For `truncate_knn()`, from zero to
#'   the stored width. For `remove_self_knn()`, it is the number of nonself
#'   neighbors. For `self_first_knn()`, it is the total width including self,
#'   at least one.
#'   Supplying `self_dist` permits a width one greater than the stored width
#'   when every row has enough nonself neighbors.
#' @param by Optional numeric matrix of scores with the same dimensions as the
#'   graph matrices. Defaults to the stored distances.
#' @param self_dist Finite numeric scalar used when a row has no self entry.
#'   The default `NULL` errors if self is absent.
#' @return A list containing the resulting `idx` and `dist` matrices.
#' @examples
#' graph <- brute_force_knn(iris[1:8, 1:4], k = 6)
#' truncate_knn(graph, k = 3)
#' remove_self_knn(graph, k = 5)
#' sort_knn(graph)
#' self_first_knn(graph)
#' @name knn-utilities
NULL

#' @rdname knn-utilities
#' @export
truncate_knn <- function(graph, k) {
  graph <- knn_matrices(graph)
  k <- check_count(k, "k", min = 0L, max = ncol(graph$idx))
  cols <- seq_len(k)
  list(
    idx = graph$idx[, cols, drop = FALSE],
    dist = graph$dist[, cols, drop = FALSE]
  )
}

#' @rdname knn-utilities
#' @export
sort_knn <- function(graph, by = NULL) {
  graph <- knn_matrices(graph)
  if (is.null(by)) {
    by <- graph$dist
  } else if (!is.numeric(by) || !identical(dim(by), dim(graph$idx))) {
    stop("'by' must be a numeric matrix with the same dimensions as 'idx'")
  }

  for (row in seq_len(nrow(graph$idx))) {
    positions <- order(graph$idx[row, ] == 0L, by[row, ], method = "radix")
    graph$idx[row, ] <- graph$idx[row, positions]
    graph$dist[row, ] <- graph$dist[row, positions]
  }
  graph
}

#' @rdname knn-utilities
#' @export
remove_self_knn <- function(graph, k = max(0L, ncol(graph$idx) - 1L)) {
  graph <- knn_matrices(graph)
  k <- check_count(k, "k", min = 0L, max = ncol(graph$idx))
  select_self_knn(graph, k, include_self = FALSE)
}

#' @rdname knn-utilities
#' @export
self_first_knn <- function(graph, k = ncol(graph$idx), self_dist = NULL) {
  graph <- knn_matrices(graph)
  max_k <- as.double(ncol(graph$idx)) + !is.null(self_dist)
  k <- check_count(k, "k", max = min(max_k, .Machine$integer.max))
  if (!is.null(self_dist) && !is_single_finite_number(self_dist)) {
    stop("'self_dist' must be a single finite number")
  }
  select_self_knn(graph, k, include_self = TRUE, self_dist = self_dist)
}

select_self_knn <- function(graph, k, include_self, self_dist = NULL) {
  n <- nrow(graph$idx)
  check_graph_entries(graph, n, "graph")

  n_nonself <- k - include_self
  result <- list(idx = matrix(0L, n, k), dist = matrix(NA_real_, n, k))
  for (row in seq_len(n)) {
    idx <- graph$idx[row, ]
    dist <- graph$dist[row, ]
    positions <- which(idx != row & idx != 0L)
    if (length(positions) < n_nonself) {
      stop(
        "graph row ",
        row,
        " has fewer than ",
        n_nonself,
        " nonself neighbors"
      )
    }
    positions <- positions[seq_len(n_nonself)]

    if (include_self) {
      self <- match(row, idx)
      if (is.na(self) && is.null(self_dist)) {
        stop("graph row ", row, " has no self neighbor; supply 'self_dist'")
      }
      idx <- c(row, idx[positions])
      dist <- c(if (is.na(self)) self_dist else dist[self], dist[positions])
    } else {
      idx <- idx[positions]
      dist <- dist[positions]
    }
    result$idx[row, ] <- idx
    result$dist[row, ] <- dist
  }
  result
}

#' Calculate distances for fixed neighbor indices
#'
#' Calculate distances for the supplied indices in their existing order. Rows
#' of `idx` correspond to observations, or to queries when `query` is supplied.
#' Indices always refer to observations in `data`. Missing indices (`0`) remain
#' paired with `NA` distances.
#'
#' @param data Reference observations, in a numeric matrix or data frame, a
#'   logical matrix, or a numeric sparse matrix.
#' @param idx Numeric matrix of whole-number neighbor indices. For a graph,
#'   pass `graph$idx`.
#' @param metric Distance metric, evaluated directly. See [brute_force_knn()].
#' @param query Optional query observations with the same feature layout as
#'   `data`. Query and reference must both be sparse or both dense, and
#'   their logical or numeric storage must match. Logical features cannot be `NA`.
#' @param n_threads Number of threads to use. Zero runs serially.
#' @param verbose Whether to report progress.
#' @param obs Whether observations are in rows (`"R"`) or columns (`"C"`).
#' @return A list containing the supplied `idx` and calculated `dist` matrices.
#' @examples
#' graph <- brute_force_knn(iris[1:8, 1:4], k = 5)
#' knn_distances(iris[1:8, 1:4], graph$idx, metric = "manhattan")
#' @export
knn_distances <- function(
  data,
  idx,
  metric = "euclidean",
  query = NULL,
  n_threads = 0,
  verbose = FALSE,
  obs = "R"
) {
  obs <- match.arg(toupper(obs), c("R", "C"))
  n_threads <- check_n_threads(n_threads)
  if (!is.matrix(idx) || !is.numeric(idx)) {
    stop("'idx' must be a numeric matrix of neighbor indices")
  }

  data <- x2m(data, obs)
  if (!is.null(query)) {
    query <- x2m(query, obs, "query")
    check_sparse(data, query)
    check_matching_features(data, query, obs)
  }
  if (obs == "R") {
    data <- Matrix::t(data)
    if (!is.null(query)) {
      query <- Matrix::t(query)
    }
  }

  n_points <- if (is.null(query)) ncol(data) else ncol(query)
  if (nrow(idx) != n_points) {
    stop("'idx' must have ", n_points, " rows, but has ", nrow(idx))
  }
  check_candidate_indices(idx, ncol(data), "'idx'")
  score_knn_indices(data, idx, query, metric, n_threads, verbose)
}

knn_matrices <- function(graph) {
  if (!is.list(graph)) {
    stop("'graph' must be a list with 'idx' and 'dist' matrices")
  }
  idx <- graph[["idx"]]
  dist <- graph[["dist"]]
  check_graph_matrices(idx, dist)
  if (!is.numeric(idx) || !(is.numeric(dist) || is.logical(dist))) {
    stop("Neighbor indices and distances must be numeric")
  }
  list(idx = idx, dist = dist)
}
