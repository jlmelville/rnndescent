# Sparse ------------------------------------------------------------------

#' Convert a neighbor graph to a sparse matrix
#'
#' Rows represent observations or queries and columns represent reference
#' observations. Set `transpose = TRUE` to reverse the axes. Missing neighbors
#' (`idx = 0`, `dist = NA`) are omitted. Stored zero distances are retained.
#' If a row contains the same neighbor more than once, the last stored distance
#' is used.
#'
#' @param graph A neighbor graph containing `idx` and `dist` matrices.
#' @param n_reference Number of reference observations. Supply this explicitly
#'   for query results when the reference size differs from the number of rows.
#' @param transpose Whether to put reference observations in rows.
#' @return A numeric `dgCMatrix` of distances.
#' @examples
#' graph <- brute_force_knn(iris[1:8, 1:4], k = 5)
#' knn_to_sparse(graph)
#' @export
knn_to_sparse <- function(
  graph,
  n_reference = nrow(graph$idx),
  transpose = FALSE
) {
  graph <- knn_matrices(graph)
  n_reference <- check_count(
    n_reference,
    "n_reference",
    min = 0L,
    max = .Machine$integer.max
  )
  if (!is.logical(transpose) || length(transpose) != 1L || is.na(transpose)) {
    stop("'transpose' must be TRUE or FALSE")
  }
  check_graph_entries(graph, n_reference, "graph")
  graph_to_sparse(graph, "C", n_reference, transpose, use_last = TRUE)
}

graph_to_sparse <- function(
  graph,
  repr,
  n_reference = nrow(graph$idx),
  transpose = FALSE,
  use_last = FALSE
) {
  n_row <- nrow(graph$idx)
  dims <- c(n_row, n_reference)
  keep <- !is.na(graph$dist)
  i <- rep(seq_len(n_row), times = ncol(graph$idx))[keep]
  j <- graph$idx[keep]
  x <- as.double(graph$dist[keep])

  if (transpose) {
    dims <- rev(dims)
  }
  Matrix::sparseMatrix(
    i = if (transpose) j else i,
    j = if (transpose) i else j,
    x = x,
    dims = dims,
    repr = repr,
    use.last.ij = use_last
  )
}

graph_to_rsparse <- function(graph) {
  graph_to_sparse(graph, "R")
}

graph_to_csparse <- function(graph) {
  graph_to_sparse(graph, "C")
}

rsparse_to_list <- function(spr) {
  list(row_ptr = spr@p, col_idx = spr@j, dist = spr@x)
}

csparse_to_list <- function(spc) {
  spct <- Matrix::t(spc)
  tcsparse_to_list(spct)
}

tcsparse_to_list <- function(spct) {
  list(row_ptr = spct@p, col_idx = spct@i, dist = spct@x)
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
  sr <- graph_to_rsparse(graph)
  rsparse_to_list(sr)
}

# Set explicit zero to a very small number so they aren't dropped.
# Diagonal distance is still dropped
preserve_zeros <- function(sp) {
  sp@x[sp@x == 0] <- .Machine$double.eps
  Matrix::diag(sp) <- 0
  Matrix::drop0(sp)
}
