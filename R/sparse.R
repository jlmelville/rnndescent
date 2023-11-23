# Sparse ------------------------------------------------------------------

graph_to_sparse <- function(graph, repr, drop0 = FALSE, transpose = FALSE) {
  idx <- graph$idx
  dist <- graph$dist
  n_nbrs <- ncol(idx)
  n_row <- nrow(idx)
  n_ref <- nrow(idx)

  if (transpose) {
    i <- as.vector(idx)
    j <- rep(1:n_row, times = n_nbrs)
  } else {
    i <- rep(1:n_row, times = n_nbrs)
    j <- as.vector(idx)
  }
  x <- as.vector(dist)

  # filter out missing data
  keep_indices <- !is.na(x)
  i <- i[keep_indices]
  j <- j[keep_indices]
  x <- x[keep_indices]

  res <- Matrix::sparseMatrix(
    i = i,
    j = j,
    x = x,
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
