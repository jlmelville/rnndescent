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
      dmat[i, j] <- squared_euclidean_distance(ui10[i, ], ui10[j, ])
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

test_that("Bray-Curtis", {
  dbf <- brute_force_knn(ui10z, k = 4, metric = "braycurtis")
  sbd <- brute_force_knn(ui10sp, k = 4, metric = "braycurtis")
  expect_equal(dbf, sbd, tol = 1e-7)
})

test_that("Canberra", {
  dbf <- brute_force_knn(ui10z, k = 4, metric = "canberra")
  sbd <- brute_force_knn(ui10sp, k = 4, metric = "canberra")
  expect_equal(dbf, sbd)
})

test_that("Chebyshev", {
  dbf <- brute_force_knn(ui10z, k = 4, metric = "chebyshev")
  sbd <- brute_force_knn(ui10sp, k = 4, metric = "chebyshev")
  expect_equal(dbf, sbd)
})

test_that("Dice", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "dice")
  bfbin <- brute_force_knn(bitdata, k = 4, metric = "bdice")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "dice")
  expect_equal(bfdense, bfsparse)
})

test_that("Dot", {
  bfdense <- brute_force_knn(ui10z, k = 4, metric = "dot")
  bfsparse <- brute_force_knn(ui10sp, k = 4, metric = "dot")
  expect_equal(bfdense, bfsparse, tol = 1e-7)
})

test_that("Hamming", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "hamming")
  bfbin <- brute_force_knn(bitdata, k = 4, metric = "bhamming")
  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "hamming")
  expect_equal(bfdense, bfsparse)
})

test_that("Hellinger", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "hellinger")
  araw <- brute_force_knn(bitdata, k = 4, metric = "alternative-hellinger")
  unc <- apply_dense_alt_metric_uncorrection("hellinger", bfdense$dist)
  expect_equal(unc, araw$dist)
  expect_equal(brute_force_knn(bitdata, k = 4, metric = "hellinger", use_alt_metric = FALSE), bfdense, tol = 1e-7)

  # sparse
  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "hellinger", use_alt_metric = FALSE)
  expect_equal(bfdense, bfsparse, tol = 1e-7)

  # check sparse uncorrection
  set.seed(1337); spunc <- nnd_knn(bitdatasp, k = 4, metric = "hellinger", n_iters = 0,
                                   init = random_knn(bitdatasp, k = 4, metric = "hellinger"))
  set.seed(1337); spnoc <- nnd_knn(bitdatasp, k = 4, metric = "hellinger", use_alt_metric = FALSE, n_iters = 0,
                                   init = random_knn(bitdatasp, k = 4, metric = "hellinger"))
  expect_equal(spunc, spnoc, tol = 1e-7)

})

test_that("Jaccard", {
  # check corrections
  jbf <- brute_force_knn(bitdata, k = 4, metric = "jaccard")
  aj_raw <- brute_force_knn(bitdata, k = 4, metric = "alternative-jaccard")
  j_unc <- apply_dense_alt_metric_uncorrection("jaccard", jbf$dist)
  expect_equal(j_unc, aj_raw$dist)

  expect_equal(brute_force_knn(bitdata, k = 4, metric = "jaccard", use_alt_metric = FALSE), jbf, tol = 1e-7)
  expect_equal(jbf, brute_force_knn(bitdata, k = 4, metric = "bjaccard"), tol = 1e-7)

  # check dense uncorrection
  set.seed(1337); dunc <- nnd_knn(bitdata, k = 4, metric = "jaccard", n_iters = 0,
                                  init = random_knn(bitdatasp, k = 4, metric = "jaccard"))
  set.seed(1337); dnoc <- nnd_knn(bitdata, k = 4, metric = "jaccard", use_alt_metric = FALSE, n_iters = 0,
                                  init = random_knn(bitdatasp, k = 4, metric = "jaccard"))
  expect_equal(dunc, dnoc, tol = 1e-7)

  # check sparse
  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "jaccard", use_alt_metric = FALSE)
  expect_equal(bfsparse, jbf, tol = 1e-7)

  # check sparse uncorrection
  set.seed(1337); spunc <- nnd_knn(bitdatasp, k = 4, metric = "jaccard", n_iters = 0,
                                  init = random_knn(bitdatasp, k = 4, metric = "jaccard"))
  set.seed(1337); spnoc <- nnd_knn(bitdatasp, k = 4, metric = "jaccard", use_alt_metric = FALSE, n_iters = 0,
                                  init = random_knn(bitdatasp, k = 4, metric = "jaccard"))
  expect_equal(spunc, spnoc, tol = 1e-7)
})

test_that("Jensen-Shannon", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "jensenshannon")

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "jensenshannon")
  expect_equal(bfdense, bfsparse)
})

test_that("Matching", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "matching")
  bfbin <- brute_force_knn(bitdata, k = 4, metric = "bmatching")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "matching")
  expect_equal(bfdense, bfsparse)
})

test_that("Kulsinski", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "kulsinski")
  bfbin <- brute_force_knn(bitdata, k = 4, metric = "bkulsinski")
  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "kulsinski")
  expect_equal(bfdense, bfsparse)
})

test_that("Rogers-Tanimoto", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "rogerstanimoto")
  bfbin <- brute_force_knn(bitdata, k = 4, metric = "brogerstanimoto")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "rogerstanimoto")
  expect_equal(bfdense, bfsparse)
})

test_that("Russell-Rao", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "russellrao")
  bfbin <- brute_force_knn(bitdata, k = 4, metric = "brussellrao")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "russellrao")
  expect_equal(bfdense, bfsparse)
})

test_that("Sokal-Michener", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "sokalmichener")
  bfbin <- brute_force_knn(bitdata, k = 4, metric = "bsokalmichener")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "sokalmichener")
  expect_equal(bfdense, bfsparse)
})

test_that("Sokal-Sneath", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "sokalsneath")
  bfbin <- brute_force_knn(bitdata, k = 4, metric = "bsokalsneath")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "sokalsneath")
  expect_equal(bfdense, bfsparse)
})

test_that("Symmetric KL", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "symmetrickl")

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "symmetrickl")
  expect_equal(bfdense, bfsparse)
})

test_that("TS-SS", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "tsss")
  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "tsss")
  expect_equal(bfdense, bfsparse)
})

test_that("Yule", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "yule")
  bfbin <- brute_force_knn(bitdata, k = 4, metric = "byule")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "yule")
  expect_equal(bfdense, bfsparse)
})
