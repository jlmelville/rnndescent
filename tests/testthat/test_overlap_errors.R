library(rnndescent)

test_that("overlap column errors identify idx2 and the requested k", {
  idx1 <- matrix(1:12, nrow = 3)
  idx2 <- idx1[, 1:2]
  for (k in c(3L, 4L)) {
    error <- tryCatch(neighbor_overlap(idx1, idx2, k = k), error = identity)
    expect_identical(
      conditionMessage(error),
      paste0("Not enough columns in 'idx2' for k = ", k)
    )
    expect_s3_class(error, "simpleError")
  }
})

test_that("overlap row errors explain either direction of mismatch", {
  idx1 <- list(idx = matrix(1:6, nrow = 3))
  for (n_rows in c(2L, 4L)) {
    idx2 <- list(idx = matrix(1L, nrow = n_rows, ncol = 2))
    error <- tryCatch(neighbor_overlap(idx1, idx2), error = identity)
    expect_identical(
      conditionMessage(error),
      "'idx1' and 'idx2' must have the same number of rows"
    )
    expect_s3_class(error, "simpleError")
  }
})

test_that("overlap column errors precede row errors and ret_vec", {
  error <- tryCatch(
    neighbor_overlap(
      matrix(1L, nrow = 3, ncol = 2),
      list(idx = matrix(1L, nrow = 2, ncol = 1)),
      k = 2,
      ret_vec = stop("ret_vec was forced")
    ),
    error = identity
  )
  expect_identical(
    conditionMessage(error),
    "Not enough columns in 'idx2' for k = 2"
  )
  expect_s3_class(error, "simpleError")
})
