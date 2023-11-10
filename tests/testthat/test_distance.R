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

test_that("Jaccard", {
  # check corrections
  jbf <- brute_force_knn(bitdata, k = 4, metric = "jaccard")
  aj_raw <- brute_force_knn(bitdata, k = 4, metric = "alternative-jaccard")
  j_unc <- apply_dense_alt_metric_uncorrection("jaccard", jbf$dist)
  expect_equal(j_unc, aj_raw$dist)

  expect_equal(brute_force_knn(bitdata, k = 4, metric = "jaccard", use_alt_metric = FALSE), jbf, tol = 1e-7)
  expect_equal(jbf, brute_force_knn(bitdata, k = 4, metric = "bjaccard"), tol = 1e-7)
})

test_that("Hellinger", {
  bf <- brute_force_knn(bitdata, k = 4, metric = "hellinger")
  araw <- brute_force_knn(bitdata, k = 4, metric = "alternative-hellinger")
  unc <- apply_dense_alt_metric_uncorrection("hellinger", bf$dist)
  expect_equal(unc, araw$dist)
  expect_equal(brute_force_knn(bitdata, k = 4, metric = "hellinger", use_alt_metric = FALSE), bf, tol = 1e-7)
})

test_that("Matching", {
  bfm <- brute_force_knn(bitdata, k = 4, metric = "matching")
  bfmb <- brute_force_knn(bitdata, k = 4, metric = "bmatching")

  expect_equal(bfm, bfmb)
})

test_that("Kulsinski", {
  bfm <- brute_force_knn(bitdata, k = 4, metric = "kulsinski")
  bfmb <- brute_force_knn(bitdata, k = 4, metric = "bkulsinski")

  expect_equal(bfm, bfmb)
})
