# Rounded references and near-zero float distances use absolute error budgets.
# This helper compares finite numeric scalars/arrays, allowing matching NA entries.
expect_equal_abs <- function(
  actual,
  expected,
  tolerance,
  label = deparse(substitute(actual))
) {
  if (!identical(attributes(actual), attributes(expected))) {
    testthat::expect_identical(
      attributes(actual),
      attributes(expected),
      info = label
    )
    return(invisible(actual))
  }
  if (!identical(is.na(actual), is.na(expected))) {
    testthat::expect_identical(is.na(actual), is.na(expected), info = label)
    return(invisible(actual))
  }
  present <- !is.na(expected)
  difference <- abs(actual[present] - expected[present])
  testthat::expect_true(
    all(is.finite(difference) & difference < tolerance),
    label = label,
    info = paste(
      "Maximum absolute difference",
      max(difference, 0),
      "must be strictly below",
      tolerance
    )
  )
  invisible(actual)
}

check_query_nbrs_idx <- function(nnidx, nref) {
  testthat::expect_true(all(nnidx >= 1 & nnidx <= nref & nnidx == floor(nnidx)))
  testthat::expect_true(all(apply(nnidx, 1, anyDuplicated) == 0L))
}

check_nbrs_order <- function(nn) {
  testthat::expect_false(any(apply(nn$dist, 1, is.unsorted)))
}

# Compare every returned edge with its own reference distance and error budget.
check_query_nbrs_dist <- function(
  nn,
  expected_dist,
  ref_range,
  query_range,
  tolerance,
  absolute = FALSE
) {
  positions <- cbind(query_range[row(nn$idx)], ref_range[nn$idx])
  expected <- matrix(expected_dist[positions], nrow(nn$idx), ncol(nn$idx))
  testthat::expect_identical(dim(nn$dist), dim(expected))
  if (absolute) {
    expect_equal_abs(unname(nn$dist), expected, tolerance)
    return(invisible(NULL))
  }

  difference <- abs(nn$dist - expected)
  relative <- is.finite(expected) & abs(expected) > tolerance
  difference[relative] <- difference[relative] / abs(expected[relative])
  ok <- (is.na(nn$dist) & is.na(expected)) |
    (!is.na(nn$dist) &
      !is.na(expected) &
      (nn$dist == expected | difference < tolerance))
  testthat::expect_true(
    all(ok),
    info = paste(
      "Incorrect distances at matrix positions",
      toString(which(!ok))
    )
  )
}

check_nbrs <- function(
  nn,
  expected_dist,
  tolerance = .Machine$double.eps,
  check_idx_order = TRUE,
  check_dist_order = check_idx_order
) {
  rows <- seq_len(nrow(nn$idx))
  check_query_nbrs_idx(nn$idx, length(rows))
  if (check_idx_order) {
    # A sampled graph can omit self. When present, self must sort first.
    others <- nn$idx[, -1L, drop = FALSE]
    testthat::expect_true(all(others != row(others)))
  }
  check_query_nbrs_dist(nn, expected_dist, rows, rows, tolerance)
  if (check_dist_order) {
    check_nbrs_order(nn)
  }
}

check_query_nbrs <- function(
  nn,
  query,
  ref_range,
  query_range,
  k,
  expected_dist,
  tolerance = .Machine$double.eps,
  check_order = TRUE,
  absolute = FALSE
) {
  testthat::expect_shape(nn$idx, dim = c(nrow(query), k))
  check_query_nbrs_idx(nn$idx, length(ref_range))
  check_query_nbrs_dist(
    nn,
    expected_dist,
    ref_range,
    query_range,
    tolerance,
    absolute
  )
  if (check_order) {
    check_nbrs_order(nn)
  }
}

capture_everything <- function(code) {
  paste0(
    capture.output(type = "output", capture.output(type = "message", code)),
    collapse = ""
  )
}

copy <- function(m) {
  matrix(m, nrow = nrow(m), ncol = ncol(m))
}
