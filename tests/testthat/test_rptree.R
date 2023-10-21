library(rnndescent)
context("RP Tree")

set.seed(1337)
res <- rp_tree_knn(ui10, k = 4, leaf_size = 3)
expect_equal(dim(res$idx), c(10, 4))
expect_equal(dim(res$dist), c(10, 4))
# knn contains missing values
expect_in(c(NA), res$dist)
expect_in(c(0), res$idx)

expected_rpknn <- list(
idx = matrix(c(
  6, 0, 0, 0,
  4, 9, 0, 0,
  7, 0, 0, 0,
  9, 2, 0, 0,
  8, 0, 0, 0,
  1, 0, 0, 0,
  3, 0, 0, 0,
  5, 0, 0, 0,
  4, 2, 0, 0,
  0, 0, 0, 0
), nrow = 10, byrow = TRUE),

dist = matrix(c(
  0.3464102, NA, NA, NA,
  0.5477225, 0.6164413, NA, NA,
  0.2236066, NA, NA, NA,
  0.3464101, 0.5477225, NA, NA,
  0.1732050, NA, NA, NA,
  0.3464102, NA, NA, NA,
  0.2236066, NA, NA, NA,
  0.1732050, NA, NA, NA,
  0.3464101, 0.6164413, NA, NA,
  NA, NA, NA, NA
), nrow = 10, byrow = TRUE)
)

expect_equal(res$idx, expected_rpknn$idx, check.attributes = FALSE)
expect_equal(res$dist, expected_rpknn$dist, check.attributes = FALSE, tol = 1e-6)

# FIXME: this will stop working eventually with parallel RNG but that's expected
set.seed(1337)
res <- rp_tree_knn(ui10, k = 4, leaf_size = 3, n_threads = 2)
expect_equal(res$idx, expected_rpknn$idx, check.attributes = FALSE)
expect_equal(res$dist, expected_rpknn$dist, check.attributes = FALSE, tol = 1e-6)
