library(rnndescent)

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

test_that("Cosine", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "cosine")
  araw <- brute_force_knn(bitdata, k = 4, metric = "alternative-cosine")
  unc <- apply_dense_alt_metric_uncorrection("cosine", bfdense$dist)
  expect_equal(unc, araw$dist)
  expect_equal(brute_force_knn(bitdata, k = 4, metric = "cosine", use_alt_metric = FALSE), bfdense, tol = 1e-7)

  # sparse
  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "cosine", use_alt_metric = FALSE)
  expect_equal(bfdense, bfsparse, tol = 1e-7)

  # check sparse uncorrection
  set.seed(1337)
  spunc <- nnd_knn(bitdatasp,
    k = 4, metric = "cosine", n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "cosine")
  )
  set.seed(1337)
  spnoc <- nnd_knn(bitdatasp,
    k = 4, metric = "cosine", use_alt_metric = FALSE, n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "cosine")
  )
  expect_equal(spunc, spnoc, tol = 1e-7)
})

test_that("Dice", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "dice")
  bfbin <- brute_force_knn(lbitdata, k = 4, metric = "dice")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "dice")
  expect_equal(bfdense, bfsparse)
})

test_that("Dot", {
  bfdense <- brute_force_knn(ui10z, k = 4, metric = "dot")
  bfsparse <- brute_force_knn(ui10sp, k = 4, metric = "dot")
  expect_equal(bfdense, bfsparse, tol = 1e-5)

  bfdense <- brute_force_knn(bitdata, k = 4, metric = "dot")
  araw <- brute_force_knn(bitdata, k = 4, metric = "alternative-dot")
  unc <- apply_dense_alt_metric_uncorrection("dot", bfdense$dist)
  expect_equal(unc, araw$dist)
  expect_equal(brute_force_knn(bitdata, k = 4, metric = "dot", use_alt_metric = FALSE), bfdense, tol = 1e-7)

  # check dense uncorrection
  set.seed(1337)
  dunc <- nnd_knn(bitdata,
    k = 4, metric = "dot", n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "dot")
  )
  set.seed(1337)
  dnoc <- nnd_knn(bitdata,
    k = 4, metric = "dot", use_alt_metric = FALSE, n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "dot")
  )
  expect_equal(dunc, dnoc, tol = 1e-7)

  # sparse
  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "dot", use_alt_metric = FALSE)
  expect_equal(bfdense, bfsparse, tol = 1e-6)
  bfsparsec <- brute_force_knn(bitdatasp, k = 4, metric = "dot")
  expect_equal(bfdense, bfsparsec, tol = 1e-5)

  # check sparse uncorrection
  set.seed(1337)
  spunc <- nnd_knn(bitdatasp,
    k = 4, metric = "dot", n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "dot")
  )
  set.seed(1337)
  spnoc <- nnd_knn(bitdatasp,
    k = 4, metric = "dot", use_alt_metric = FALSE, n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "dot")
  )
  expect_equal(spunc, spnoc, tol = 1e-7)

  # query
  bfsparse_query <- brute_force_knn_query(bitdatasp, bitdatasp, k = 4, metric = "dot", use_alt_metric = FALSE)
})

test_that("Hamming", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "hamming")
  bfbin <- brute_force_knn(lbitdata, k = 4, metric = "hamming")
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

  # check dense uncorrection
  set.seed(1337)
  dunc <- nnd_knn(bitdata,
    k = 4, metric = "hellinger", n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "hellinger")
  )
  set.seed(1337)
  dnoc <- nnd_knn(bitdata,
    k = 4, metric = "hellinger", use_alt_metric = FALSE, n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "hellinger")
  )
  expect_equal(dunc, dnoc, tol = 1e-7)

  # sparse
  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "hellinger", use_alt_metric = FALSE)
  expect_equal(bfdense, bfsparse, tol = 1e-7)
  bfsparsec <- brute_force_knn(bitdatasp, k = 4, metric = "hellinger")
  expect_equal(bfdense, bfsparsec, tol = 1e-7)

  # check sparse uncorrection
  set.seed(1337)
  spunc <- nnd_knn(bitdatasp,
    k = 4, metric = "hellinger", n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "hellinger")
  )
  set.seed(1337)
  spnoc <- nnd_knn(bitdatasp,
    k = 4, metric = "hellinger", use_alt_metric = FALSE, n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "hellinger")
  )
  expect_equal(spunc, spnoc, tol = 1e-7)
})

test_that("Jaccard", {
  # check corrections
  jbf <- brute_force_knn(bitdata, k = 4, metric = "jaccard")
  aj_raw <- brute_force_knn(bitdata, k = 4, metric = "alternative-jaccard")
  j_unc <- apply_dense_alt_metric_uncorrection("jaccard", jbf$dist)
  expect_equal(j_unc, aj_raw$dist)

  expect_equal(brute_force_knn(bitdata, k = 4, metric = "jaccard", use_alt_metric = FALSE), jbf, tol = 1e-7)
  expect_equal(jbf, brute_force_knn(lbitdata, k = 4, metric = "jaccard"), tol = 1e-7)

  bfbin <- brute_force_knn(lbitdata, k = 4, metric = "jaccard")
  expect_equal(bfbin, jbf)

  # check dense uncorrection
  set.seed(1337)
  dunc <- nnd_knn(bitdata,
    k = 4, metric = "jaccard", n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "jaccard")
  )
  set.seed(1337)
  dnoc <- nnd_knn(bitdata,
    k = 4, metric = "jaccard", use_alt_metric = FALSE, n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "jaccard")
  )
  expect_equal(dunc, dnoc, tol = 1e-7)

  # check sparse
  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "jaccard", use_alt_metric = FALSE)
  expect_equal(bfsparse, jbf, tol = 1e-7)

  # check sparse uncorrection
  set.seed(1337)
  spunc <- nnd_knn(bitdatasp,
    k = 4, metric = "jaccard", n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "jaccard")
  )
  set.seed(1337)
  spnoc <- nnd_knn(bitdatasp,
    k = 4, metric = "jaccard", use_alt_metric = FALSE, n_iters = 0,
    init = random_knn(bitdatasp, k = 4, metric = "jaccard")
  )
  expect_equal(spunc, spnoc, tol = 1e-7)
})

test_that("Jensen-Shannon", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "jensenshannon")

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "jensenshannon")
  expect_equal(bfdense, bfsparse)
})

test_that("Matching", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "matching")
  bfbin <- brute_force_knn(lbitdata, k = 4, metric = "matching")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "matching")
  expect_equal(bfdense, bfsparse)
})

test_that("Kulsinski", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "kulsinski")
  bfbin <- brute_force_knn(lbitdata, k = 4, metric = "kulsinski")
  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "kulsinski")
  expect_equal(bfdense, bfsparse)
})

test_that("Rogers-Tanimoto", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "rogerstanimoto")
  bfbin <- brute_force_knn(lbitdata, k = 4, metric = "rogerstanimoto")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "rogerstanimoto")
  expect_equal(bfdense, bfsparse)
})

test_that("Russell-Rao", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "russellrao")
  bfbin <- brute_force_knn(lbitdata, k = 4, metric = "russellrao")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "russellrao")
  expect_equal(bfdense, bfsparse)
})

test_that("Sokal-Michener", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "sokalmichener")
  bfbin <- brute_force_knn(lbitdata, k = 4, metric = "sokalmichener")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "sokalmichener")
  expect_equal(bfdense, bfsparse)
})

test_that("Sokal-Sneath", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "sokalsneath")
  bfbin <- brute_force_knn(lbitdata, k = 4, metric = "sokalsneath")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "sokalsneath")
  expect_equal(bfdense, bfsparse)
})

test_that("Spearman Rank", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "spearmanr")

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "spearmanr")
  expect_equal(bfdense, bfsparse, tol = 1e-6)

  # test with negative values
  set.seed(42)
  m <- matrix(rnorm(10 * 10), nrow = 10)
  mz <- m
  mz[sample(length(m), 20)] <- 0
  msp <- Matrix::drop0(mz)
  expect_equal(brute_force_knn(mz, metric = "spearmanr", k = 4),
    brute_force_knn(msp, metric = "spearmanr", k = 4),
    tol = 1e-7
  )
})

test_that("Symmetric KL", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "symmetrickl")

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "symmetrickl")
  expect_equal(bfdense, bfsparse)
})

test_that("true angular", {
  # true angular is weird in that it's a similarity not a distance
  # it behaves very strangely, so this is just to characterize its strange
  # behavior rather than to assert it's behaving in a semantically "correct"
  # manner

  # alt metric uses cosine so ordering is based on that, but the conversion
  # turns it back into a similarity so the distances are in descending order
  bfdense <- brute_force_knn(bitdata, k = 10, metric = "trueangular")
  bfsparse <- brute_force_knn(bitdatasp, k = 10, metric = "trueangular")
  expect_equal(bfdense, bfsparse, tol = 1e-4)

  # turn off alt metric so we get the true angular ordering
  # now self "distance" is the furthest distance but the distances are increasing
  # do all 10 neighbors so we can ensure self-distance is 1 and not subject
  # to acos rounding error causing NaN
  bfdensena <- brute_force_knn(bitdata, k = 10, metric = "trueangular", use_alt_metric = FALSE)
  bfsparsena <- brute_force_knn(bitdatasp, k = 10, metric = "trueangular", use_alt_metric = FALSE)
  expect_equal(bfdensena, bfsparsena, tol = 1e-5)
  expect_equal(bfdensena$idx[, 10], 1:10)
  expect_equal(bfdensena$dist[, 10], rep(1.0, 10), tol = 1e-3)

  # more uncorrections
  set.seed(42)
  true_nnd <-
    nnd_knn(
      bitdata,
      k = 4,
      init = random_knn(bitdata, k = 4, metric = "trueangular"),
      metric = "trueangular",
      max_candidates = 8
    )
  true_bf <- brute_force_knn(bitdata, k = 4, metric = "trueangular")
  expect_equal(true_nnd, true_bf)
  set.seed(42)
  spinit <- random_knn(bitdatasp, k = 4, metric = "trueangular")
  spinit$idx[1, 2] <- 0
  spinit$dist[1, 2] <- NA
  true_nnd <-
    nnd_knn(
      bitdatasp,
      k = 4,
      init = spinit,
      metric = "trueangular",
      max_candidates = 8
    )
  expect_equal(true_nnd, true_bf, tol = 1e-4)
})

test_that("TS-SS", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "tsss")
  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "tsss")
  expect_equal(bfdense, bfsparse)

  #8: TS-SS distances seem to be off by a factor of 2 compared to the definiton
  # in the paper. rnndescent was designed to reproduce the PyNNDescent results
  # which we will test against below and divide by 2
  python_distances <-
    matrix(c(0, 0.29106654, 1.50010305, 1.76843838,
           0, 0.20153333, 0.34656114, 0.369184,
           0, 0.09994711, 0.27364606, 0.49758044,
           0, 0.15930582, 0.44352352, 0.78938532,
           0, 0.06631046, 0.19698709, 0.27364606,
           0, 0.29106654, 0.75232326, 1.30880218,
           0, 0.09994711, 0.19698709, 0.20153333,
           0, 0.06631046, 0.34656114, 0.44352352,
           0, 0.15930582, 0.9577528, 1.0852265,
           0, 1.30880218, 1.50010305, 5.27228563),
         nrow = 10, ncol = 4, byrow = TRUE) / 2
  ui10_tsss <- brute_force_knn(ui10, k = 4, metric = "tsss")
  expect_equal(ui10_tsss$dist, python_distances, tol = 1e-5)
})

test_that("Yule", {
  bfdense <- brute_force_knn(bitdata, k = 4, metric = "yule")
  bfbin <- brute_force_knn(lbitdata, k = 4, metric = "yule")

  expect_equal(bfdense, bfbin)

  bfsparse <- brute_force_knn(bitdatasp, k = 4, metric = "yule")
  expect_equal(bfdense, bfsparse)
})
