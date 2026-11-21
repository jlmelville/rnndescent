test_that("k-occurrence identifies invalid neighbor indices and k", {
  idx <- matrix(c(1L, 2L, 2L, 3L, 3L, 1L), ncol = 2, byrow = TRUE)
  expect_error(
    k_occur(list(idx = 1:3)),
    "Expected a matrix of neighbor indices"
  )
  for (k in c(0, NA_real_)) {
    expect_error(k_occur(idx, k = k), "'k' must be at least 1")
  }
  expect_error(
    k_occur(idx, k = 3),
    "'k' must not exceed the number of neighbor columns"
  )
  expect_error(k_occur(idx, k = c(1L, 2L)), "single value")
})

test_that("sparse k-occurrence still ignores k before dense validation", {
  graph <- Matrix::sparseMatrix(
    i = c(1L, 2L, 3L),
    j = c(1L, 1L, 2L),
    x = c(1, 1, 1),
    dims = c(4L, 4L)
  )
  expect_identical(
    k_occur(graph, k = stop("k must not be evaluated")),
    c(2L, 1L, 0L, 0L)
  )
  expect_identical(
    k_occur(graph, k = NA_real_, include_self = FALSE),
    c(1L, 1L, 0L, 0L)
  )
})
