# Graph Construction Helpers ----------------------------------------------

# i should not appear in the "true" neighbors (i.e. it should be the first
# neighbor), indexes should be in the range (1, nr) and indexes should be unique
check_nbri <- function(nnidx, i) {
  nr <- nrow(nnidx)
  nc <- ncol(nnidx)
  true_nbrs <- nnidx[i, 2:nc]
  expect_true(all(i != true_nbrs), label = i)
  expect_true(all(true_nbrs > 0), label = i)
  expect_true(all(true_nbrs <= nr), label = i)
  expect_true(length(unique(true_nbrs)) == nc - 1)
}

check_nbri_unordered <- function(nnidx, i) {
  nr <- nrow(nnidx)
  nc <- ncol(nnidx)
  nbrs <- nnidx[i, , drop = FALSE]
  expect_true(all(nbrs > 0), label = i)
  expect_true(all(nbrs <= nr), label = i)
  expect_true(length(unique(nbrs)) == nc)
}

check_nbrs_idx <- function(nnidx, check_order = TRUE) {
  nr <- nrow(nnidx)
  for (i in 1:nr) {
    if (check_order) {
      check_nbri(nnidx, i)
    } else {
      check_nbri_unordered(nnidx, i)
    }
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

check_nbrs_order <- function(nn) {
  expect_true(all(apply(nn$dist, 1, order) ==
    matrix(rep(1:ncol(nn$idx), times = nrow(nn$idx)),
      nrow = ncol(nn$idx)
    )))
}

# check_idx_order = FALSE if you don't care about the order of the indices
# check_dist_order checks that distances are in increasing order for each row
# only reason for check_idx_order = FALSE, check_dist_order = TRUE is when
# there are ties in the returned distances (e.g. hamming)
check_nbrs <- function(nn, expected_dist, tol = .Machine$double.eps, check_idx_order = TRUE, check_dist_order = check_idx_order) {
  check_nbrs_idx(nn$idx, check_order = check_idx_order)
  check_nbrs_dist(nn, expected_dist, tol = tol)
  if (check_dist_order) {
    check_nbrs_order(nn)
  }
}

# Query Helpers -----------------------------------------------------------

# indexes should be in the range (1, nref) and indexes should be unique
are_valid_query_neighbors <- function(nnidx, i, nref) {
  nc <- ncol(nnidx)
  query_nbrs <- nnidx[i, ]
  all(query_nbrs > 0 & query_nbrs <= nref) && length(unique(query_nbrs)) == nc
}


check_query_nbrs_idx <- function(nnidx, nref) {
  nr <- nrow(nnidx)
  for (i in 1:nr) {
    testthat::expect_true(all(are_valid_query_neighbors(nnidx, i, nref), label = i))
  }
}

check_query_nbrs_dist <- function(nn, expected_dist, ref_range, query_range, tol = .Machine$double.eps) {
  n_queries <- nrow(nn$idx)
  n_nbrs <- ncol(nn$idx)
  for (i in 1:n_queries) {
    for (j in 1:n_nbrs) {
      testthat::expect_equal(nn$dist[i, j],
        expected_dist[query_range[i], ref_range[nn$idx[i, j]]],
        tol = tol, label = paste0(i, ", ", j),
      )
    }
  }
}

check_nn_matrix_dim <- function(m, query, k) {
  expect_equal(nrow(m), nrow(query))
  expect_equal(ncol(m), k)
}

check_query_nbrs <- function(nn, query, ref_range, query_range, k, expected_dist, tol = .Machine$double.eps,
                             check_order = TRUE) {
  check_nn_matrix_dim(nn$idx, query, k)
  check_nn_matrix_dim(nn$dist, query, k)
  nref <- length(ref_range)
  check_query_nbrs_idx(nn$idx, nref)
  check_query_nbrs_dist(nn, expected_dist, ref_range, query_range, tol)
  if (check_order) {
    # this checks that distances are in increasing order for each row
    expect_true(all(apply(nn$dist, 1, order) == matrix(rep(1:ncol(nn$idx), times = nrow(nn$idx)), nrow = ncol(nn$idx))))
  }
}

capture_everything <- function(code) {
  paste0(capture.output(type = "output", capture.output(type = "message", code)), collapse = "")
}

copy <- function(m) {
  matrix(m, nrow = nrow(m), ncol = ncol(m))
}
