library(rnndescent)
context("Standalone distance functions")

test_that("Euclidean distance", {
  dmat <- matrix(nrow = 10, ncol = 10)
  for (i in 1:nrow(dmat)) {
    for (j in 1:ncol(dmat)) {
      dmat[i, j] <- euclidean_distance(ui10[i, ], ui10[j, ])
    }
  }
  expect_equal(dmat, ui10_eucd, check.attributes = FALSE, tol = 1e-6)

  for (i in 1:nrow(dmat)) {
    for (j in 1:ncol(dmat)) {
      dmat[i, j] <- l2sqr_distance(ui10[i, ], ui10[j, ])
    }
  }
  expect_equal(dmat, ui10_eucd ^ 2, check.attributes = FALSE, tol = 1e-6)
})

test_that("Cosine distance", {
  dmat <- matrix(nrow = 10, ncol = 10)
  for (i in 1:nrow(dmat)) {
    for (j in 1:ncol(dmat)) {
      dmat[i, j] <- cosine_distance(ui10[i, ], ui10[j, ])
    }
  }
  expect_equal(dmat, ui10_cosd, check.attributes = FALSE, tol = 1e-6)
})

test_that("Manhattan distance", {
  dmat <- matrix(nrow = 10, ncol = 10)
  for (i in 1:nrow(dmat)) {
    for (j in 1:ncol(dmat)) {
      dmat[i, j] <- manhattan_distance(ui10[i, ], ui10[j, ])
    }
  }
  expect_equal(dmat, ui10_mand, check.attributes = FALSE, tol = 1e-6)
})

test_that("Hamming distance", {
  dmat <- matrix(nrow = 10, ncol = 10)
  for (i in 1:nrow(dmat)) {
    for (j in 1:ncol(dmat)) {
      dmat[i, j] <- hamming_distance(bitdata[i, ], bitdata[j, ])
    }
  }
  expect_equal(dmat, bit10_hamd, check.attributes = FALSE, tol = 1e-6)

  dmat <- matrix(nrow = 6, ncol = 6)
  for (i in 1:nrow(dmat)) {
    for (j in 1:ncol(dmat)) {
      dmat[i, j] <- hamming_distance(int6[i, ], int6[j, ])
    }
  }
  expect_equal(dmat, int6hd, check.attributes = FALSE, tol = 1e-6)
})

test_that("Correlation distance", {
  dmat <- matrix(nrow = 10, ncol = 10)
  for (i in 1:nrow(dmat)) {
    for (j in 1:ncol(dmat)) {
      dmat[i, j] <- correlation_distance(uirism[i, ], uirism[j, ])
    }
  }
  expect_equal(dmat, uirism10_cord, check.attributes = FALSE, tol = 1e-6)
})
