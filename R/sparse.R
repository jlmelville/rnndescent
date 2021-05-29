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
