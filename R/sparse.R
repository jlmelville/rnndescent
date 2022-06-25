# Sparse ------------------------------------------------------------------

#' Convert Neighbor Graph to Sparse Distance Matrix
#'
#' Distances are stored column-wise, i.e. the neighbors of the first
#' observation in `nn` are in the first column of the matrix, neighbors of the
#' second observation are in the second column, and so on.
#'
#' Zero distances are dropped. In the typical (non-bipartite) graph case, an
#' observation is usually a neighbor of itself with a distance of zero. These
#' distances are *not* retained in the output, and hence if the number of
#' neighbors in `nn` is `k`, only `k - 1` neighbors are stored in the sparse
#' matrix.
#'
#' @param nn A nearest neighbor graph.
#' @return the data in `nn` as a sparse distance matrix in `dgCMatrix` format.
#' @examples
#' # find 4 nearest neighbors of first ten iris data
#' i10nn <- brute_force_knn(iris[1:10, ], k = 4)
#' # Nearest neighbors of each item is itself
#' all(i10nn$idx[, 1] == 1:10) # TRUE
#' # Convert to sparse
#' i10nnsp <- nn_to_sparse(i10nn)
#' # 3 neighbors are retained in the sparse format because we drop 0 distances
#' all(diff(i10nnsp@p) == 3) # TRUE
#' @export
nn_to_sparse <- function(nn) {
  graph_to_sparse(nn, repr = "C", drop0 = TRUE, transpose = TRUE)
}

graph_to_sparse <- function(graph, repr, drop0 = FALSE, transpose = FALSE) {
  idx <- graph$idx
  dist <- graph$dist
  n_nbrs <- ncol(idx)
  n_row <- nrow(idx)
  n_ref <- nrow(idx)

  if (transpose) {
    i <- as.vector(idx)
    j <- rep(1:n_row, times = n_nbrs)
  }
  else {
    i <- rep(1:n_row, times = n_nbrs)
    j <- as.vector(idx)
  }

  res <- Matrix::sparseMatrix(
    i = i,
    j = j,
    x = as.vector(dist),
    dims = c(n_row, n_ref),
    repr = repr
  )

  if (drop0) {
    res <- Matrix::drop0(res)
  }
  res
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
