test_that("absolute budgets retain small-distance and rounded-reference checks", {
  expect_success(expect_equal_abs(0.020719, 0.020720, tolerance = 1e-5))
  expect_failure(expect_equal(0.020719, 0.020720, tolerance = 1e-5))
  expect_success(expect_equal_abs(-1e-7, 0, tolerance = 1e-6))
  expect_failure(expect_equal_abs(0.0207, 0.020720, tolerance = 1e-5))
  expect_failure(
    expect_equal_abs(1, 0, tolerance = 1),
    regexp = "Maximum absolute difference 1 must be strictly below 1"
  )

  # A relative comparison must not hide a real absolute-budget violation.
  expect_success(expect_equal(100.0001, 100, tolerance = 1e-5))
  expect_failure(expect_equal_abs(100.0001, 100, tolerance = 1e-5))
})

test_that("absolute budgets preserve array structure and missing-value positions", {
  reference <- matrix(c(NA, 0, 1, 2), 2)
  actual <- reference
  actual[2, 1] <- 1e-7
  expect_success(expect_equal_abs(actual, reference, tolerance = 1e-6))
  expect_failure(expect_equal_abs(
    as.vector(actual),
    reference,
    tolerance = 1e-6
  ))
  expect_failure(expect_equal_abs(t(actual), reference, tolerance = 1e-6))
  expect_failure(expect_equal_abs(
    actual[, 1, drop = FALSE],
    reference,
    tolerance = 1e-6
  ))

  named <- actual
  rownames(named) <- c("first", "second")
  expect_failure(expect_equal_abs(named, reference, tolerance = 1e-6))
  actual[1, 1] <- 0
  expect_failure(expect_equal_abs(actual, reference, tolerance = 1e-6))
  actual <- reference
  actual[2, 2] <- Inf
  expect_failure(expect_equal_abs(actual, reference, tolerance = 1e-6))
})
