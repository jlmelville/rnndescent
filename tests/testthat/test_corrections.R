library(rnndescent)

# 14: bad uncorrection broke cosine, jaccard, hellinger default search

float32_max <- 3.4028235e+38

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
      expect_identical(
        apply_alt_metric_correction(metric, shape(values)),
        shape(expected[[metric]])
      )
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
      expect_warning(
        actual <- apply_alt_metric_uncorrection(
          metric,
          shape(values)
        ),
        NA
      )
      expect_identical(actual, shape(expected))
    }
  }
})

test_that("alternative metrics preserve edgeless search graphs", {
  graph <- Matrix::sparseMatrix(
    i = integer(),
    j = integer(),
    x = numeric(),
    dims = c(3L, 3L)
  )
  for (metric in c("dot", "jaccard", "hellinger")) {
    expect_identical(
      prepare_search_graph(diag(3), graph, metric = metric),
      graph
    )
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
