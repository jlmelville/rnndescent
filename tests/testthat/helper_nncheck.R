are_valid_neighbors <- function(nnidx, i, nr, nc) {
  i != nnidx[i, 2:nc] & nnidx[i, 2:nc] > 0 & nnidx[i, 2:nc] <= nr
}

check_nbrs_idx <- function(nnidx) {
  nr <- nrow(nnidx)
  nc <- ncol(nnidx)
  for (i in 1:nr) {
    testthat::expect_true(all(are_valid_neighbors(nnidx, i, nr, nc), label = i))
  }
}

check_nbrs_dist <- function(nn, expected_dist, tol = .Machine$double.eps) {
  nr <- nrow(nn$idx)
  n_nbrs <- ncol(nn$idx)
  for (i in 1:nr) {
    for (j in 1:n_nbrs) {
      testthat::expect_equal(nn$dist[i, j], expected_dist[i, nn$idx[i, j]],
        tol = tol, label = paste0(i, ", ", j),
      )
    }
  }
}
