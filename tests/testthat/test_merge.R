library(rnndescent)
context("Merging")


# Errors ------------------------------------------------------------------

expect_error(validate_nn_graph(list(idx = matrix(nrow = 10, ncol = 2), dist = matrix(nrow = 11, ncol = 2))), "nn matrix has 11 rows")
expect_error(validate_nn_graph(list(idx = matrix(nrow = 10, ncol = 2), dist = matrix(nrow = 10, ncol = 3))), "nn matrix has 3 cols")
expect_error(validate_are_mergeable(
  list(idx = matrix(nrow = 10, ncol = 2), dist = matrix(nrow = 10, ncol = 2)),
  list(idx = matrix(nrow = 11, ncol = 5), dist = matrix(nrow = 11, ncol = 5))),
"must have same number of rows")
