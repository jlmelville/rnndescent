test_that("k-occurrence errors identify the failed input requirement", {
  idx <- matrix(c(1L, 2L, 2L, 3L, 3L, 1L), ncol = 2, byrow = TRUE)
  input_error <- "Expected a matrix of neighbor indices in 'idx'"
  lower_error <- "'k' must be at least 1"
  upper_error <- "'k' must not exceed the number of neighbor columns"
  cases <- list(
    list(idx = 1:3, k = 0, message = input_error),
    list(idx = list(idx = 1:3), k = 0, message = input_error),
    list(idx = idx, k = 0, message = lower_error),
    list(idx = idx, k = 3, message = upper_error),
    list(idx = idx, k = NA_real_, message = lower_error),
    list(idx = idx, k = c(1, 0), message = lower_error),
    list(idx = idx, k = c(1, 3), message = upper_error),
    list(idx = idx, k = c(0, 3), message = lower_error)
  )
  for (case in cases) {
    error <- tryCatch(k_occur(case$idx, k = case$k), error = identity)
    expect_identical(class(error), c("simpleError", "error", "condition"))
    expect_identical(conditionMessage(error), case$message)
  }
})

test_that("k-occurrence keeps the existing k comparison semantics", {
  idx <- matrix(c(1L, 2L, 2L, 3L, 3L, 1L), ncol = 2, byrow = TRUE)
  expect_error(k_occur(idx, k = integer()), "Expecting a single value")
  expect_error(k_occur(idx, k = c(1L, 2L)), "Expecting a single value")
  expect_identical(k_occur(idx, k = 1.5), k_occur(idx, k = 1))
  expect_identical(k_occur(idx, k = TRUE), k_occur(idx, k = 1))
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
