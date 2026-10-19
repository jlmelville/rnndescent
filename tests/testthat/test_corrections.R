library(rnndescent)
context("Corrections")

# 14: bad uncorrection broke cosine, jaccard, hellinger default search

float32_max <- 3.4028235e+38

test_that("dot inverse", {
  expect_equal(uncorrect_alternative_cosine(0), 0)
  expect_equal(uncorrect_alternative_cosine(0.5), 1.0)
  expect_equal(uncorrect_alternative_cosine(1), float32_max)
})

test_that("jaccard", {
  expect_equal(correct_alternative_jaccard(0), 0)
  expect_equal(correct_alternative_jaccard(1), 0.5)
  expect_equal(correct_alternative_jaccard(float32_max), 1.0)

  expect_equal(uncorrect_alternative_jaccard(0), 0)
  expect_equal(uncorrect_alternative_jaccard(0.5), 1.0)
  expect_equal(uncorrect_alternative_jaccard(1), float32_max)
})

test_that("hellinger", {
  expect_equal(correct_alternative_hellinger(0), 0)
  expect_equal(correct_alternative_hellinger(1), 1 / sqrt(2))
  expect_equal(correct_alternative_hellinger(float32_max), 1.0)

  expect_equal(uncorrect_alternative_hellinger(0), 0)
  expect_equal(uncorrect_alternative_hellinger(1 / sqrt(2)), 1.0)
  expect_equal(uncorrect_alternative_hellinger(1), float32_max)
})

test_that("dot", {
  expect_equal(correct_alternative_dot(0), 0)
  expect_equal(correct_alternative_dot(1), 0.5)
  expect_equal(correct_alternative_dot(float32_max), 1.0)
})

test_that("alternative corrections retain mixed values and named shapes", {
  values <- c(-Inf, -1, 1e-8, 1, Inf, NA_real_, NaN, 1e-7)
  expected <- list(
    dot = c(-Inf, -1, 1 - 2^-1e-8, 0.5, 1, NA, NA, 1 - 2^-1e-7),
    jaccard = c(0, 0, 0, 0.5, 0, NA, NA, 0),
    hellinger = c(0, 0, 0, sqrt(0.5), 0, NA, NA, 0)
  )
  for (as_matrix in c(FALSE, TRUE)) {
    shape <- function(x) {
      if (as_matrix) {
        matrix(x, 2, dimnames = list(c("a", "b"), letters[1:4]))
      } else {
        stats::setNames(x, letters[1:8])
      }
    }
    for (metric in names(expected)) {
      for (sparse in c(FALSE, TRUE)) {
        expect_identical(
          apply_alt_metric_correction(metric, shape(values), sparse),
          shape(expected[[metric]])
        )
      }
    }
  }
})

test_that("alternative inverses skip unused invalid logarithms", {
  values <- c(0, 0.5, 1 - 2e-10, 1 - 1e-10, 1, 2, Inf, -0.5, NA, NaN)
  for (metric in c("dot", "jaccard", "hellinger")) {
    expected <- c(
      0,
      1,
      -log2(1 - values[3]),
      rep(float32_max, 4),
      -log2(1.5),
      NA,
      NA
    )
    if (metric == "hellinger") {
      expected[c(2, 3, 8)] <- c(
        -log2(0.75),
        -log2(1 - values[3]^2),
        -log2(0.75)
      )
    }
    for (as_matrix in c(FALSE, TRUE)) {
      shape <- function(x) {
        if (as_matrix) {
          matrix(x, 2, dimnames = list(c("a", "b"), letters[1:5]))
        } else {
          stats::setNames(x, letters[1:10])
        }
      }
      for (sparse in c(FALSE, TRUE)) {
        expect_warning(
          actual <- apply_alt_metric_uncorrection(
            metric,
            shape(values),
            sparse
          ),
          NA
        )
        expect_identical(actual, shape(expected))
      }
    }
  }
})

# These private wrappers also convert graph edge vectors. Public queries do not
# expose named, entirely missing or zero-length vectors for this contract.
test_that("alternative conversions retain missing and empty simplification", {
  for (metric in c("dot", "jaccard", "hellinger")) {
    for (sparse in c(FALSE, TRUE)) {
      for (inverse in c(FALSE, TRUE)) {
        convert <- if (inverse) {
          apply_alt_metric_uncorrection
        } else {
          apply_alt_metric_correction
        }
        expected <- c(a = NA, b = NA)
        if (!inverse && metric == "hellinger") {
          expected[] <- NA_real_
        }
        expect_identical(
          convert(metric, c(a = NA_real_, b = NaN), sparse),
          expected
        )
        expect_identical(convert(metric, numeric(), sparse), list())
        expect_identical(
          convert(metric, matrix(integer(), 0, 2), sparse),
          matrix(numeric(), 0, 2)
        )
      }
    }
  }
})

test_that("supplied initial distances survive alternative conversion", {
  data <- rbind(
    c(1, 0, 0, 1),
    c(0, 1, 0, 0),
    c(1, 1, 1, 0),
    c(0, 0, 1, 1),
    c(1, 0, 1, 0)
  )
  data <- data / sqrt(rowSums(data^2))
  for (metric in c("dot", "hellinger", "jaccard")) {
    for (input in list(data, methods::as(data, "dgCMatrix"))) {
      init <- brute_force_knn(input, k = 3, metric = metric, n_threads = 0)
      actual <- nnd_knn(
        input,
        init = init,
        metric = metric,
        n_iters = 0,
        n_threads = 0
      )
      for (i in seq_len(nrow(data))) {
        expect_setequal(actual$idx[i, ], init$idx[i, ])
        expect_equal(
          actual$dist[i, ],
          init$dist[i, match(actual$idx[i, ], init$idx[i, ])],
          tolerance = 1e-6
        )
      }
    }
  }
})
