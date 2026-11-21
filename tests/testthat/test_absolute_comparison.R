test_that("absolute distance budgets apply to each finite entry", {
  reference <- matrix(c(NA, 0, 1, 100), 2)
  actual <- reference
  actual[2, 1] <- 1e-7
  expect_success(expect_equal_abs(actual, reference, tolerance = 1e-6))
  actual[2, 2] <- 100.0001
  expect_failure(expect_equal_abs(actual, reference, tolerance = 1e-6))
  expect_failure(expect_equal_abs(1, 0, tolerance = 1))
  expect_failure(expect_equal_abs(as.vector(reference), reference, 1e-6))
  actual <- reference
  actual[1, 1] <- 0
  expect_failure(expect_equal_abs(actual, reference, tolerance = 1e-6))
})
