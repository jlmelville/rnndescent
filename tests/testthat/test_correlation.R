library(rnndescent)
context("Correlation distance")

# fmt: skip
cor_index <- matrix(
  c(
    1, 3, 6, 8, 5, 7, 4, 9, 10, 2,
    2, 10, 9, 4, 8, 3, 1, 6, 5, 7,
    3, 1, 8, 6, 5, 7, 9, 4, 10, 2,
    4, 9, 8, 10, 3, 1, 6, 2, 5, 7,
    5, 7, 6, 1, 3, 8, 4, 9, 10, 2,
    6, 5, 7, 1, 3, 8, 4, 9, 10, 2,
    7, 5, 6, 1, 3, 8, 4, 9, 10, 2,
    8, 3, 4, 1, 9, 6, 5, 10, 7, 2,
    9, 4, 10, 8, 2, 3, 1, 6, 5, 7,
    10, 9, 4, 2, 8, 3, 1, 6, 5, 7
  ),
  byrow = TRUE, ncol = 10, nrow = 10
)
# fmt: skip
cor_dist <- matrix(
  c(
    0.00000000e+00, 2.60889537e-05, 4.13946853e-04, 4.61852891e-04,
    6.52668500e-04, 1.18880691e-03, 1.83154822e-03, 1.92335771e-03,
    3.44803882e-03, 4.00133876e-03,
    0.00000000e+00, 9.67144550e-04, 1.45365862e-03, 2.60336976e-03,
    2.88194472e-03, 3.39291433e-03, 4.00133876e-03, 6.40810993e-03,
    7.76732119e-03, 9.27944638e-03,
    0.00000000e+00, 2.60889537e-05, 3.95491414e-04, 6.22703492e-04,
    9.38868047e-04, 1.56232107e-03, 1.64390757e-03, 1.66652079e-03,
    3.01440442e-03, 3.39291433e-03,
    0.00000000e+00, 1.66690505e-04, 4.54440488e-04, 6.93152364e-04,
    1.66652079e-03, 1.83154822e-03, 2.16742062e-03, 2.60336976e-03,
    3.28117923e-03, 3.86063303e-03,
    0.00000000e+00, 8.60443273e-05, 1.16680316e-04, 6.52668500e-04,
    9.38868047e-04, 1.49684289e-03, 3.28117923e-03, 3.96913822e-03,
    6.23882524e-03, 7.76732119e-03,
    0.00000000e+00, 1.16680316e-04, 2.77394147e-04, 4.13946853e-04,
    6.22703492e-04, 8.21174669e-04, 2.16742062e-03, 2.78434617e-03,
    4.73942264e-03, 6.40810993e-03,
    1.11022302e-16, 8.60443273e-05, 2.77394147e-04, 1.18880691e-03,
    1.56232107e-03, 2.04787836e-03, 3.86063303e-03, 4.78602797e-03,
    7.27277830e-03, 9.27944638e-03,
    0.00000000e+00, 3.95491414e-04, 4.54440488e-04, 4.61852891e-04,
    5.93789371e-04, 8.21174669e-04, 1.49684289e-03, 1.62634825e-03,
    2.04787836e-03, 2.88194472e-03,
    0.00000000e+00, 1.66690505e-04, 2.60225275e-04, 5.93789371e-04,
    1.45365862e-03, 1.64390757e-03, 1.92335771e-03, 2.78434617e-03,
    3.96913822e-03, 4.78602797e-03,
    0.00000000e+00, 2.60225275e-04, 6.93152364e-04, 9.67144550e-04,
    1.62634825e-03, 3.01440442e-03, 3.44803882e-03, 4.73942264e-03,
    6.23882524e-03, 7.27277830e-03
  ),
  byrow = TRUE, ncol = 10, nrow = 10
)

test_that("correlation neighbors match the expected indices and distances", {
  res <- brute_force_knn(uirism[1:10, ], k = 10, metric = "correlation")
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)

  res <- random_knn(uirism[1:10, ], k = 10, metric = "correlation")
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)

  res <- nnd_knn(uirism[1:10, ], k = 10, metric = "correlation")
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)

  res <- graph_knn_query(
    reference = uirism[1:10, ],
    query = uirism[1:10, ],
    reference_graph = res,
    k = 10,
    metric = "correlation"
  )
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)

  res <- brute_force_knn_query(
    reference = uirism[1:10, ],
    query = uirism[1:10, ],
    k = 10,
    metric = "correlation"
  )
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)

  res <- random_knn_query(
    reference = uirism[1:10, ],
    query = uirism[1:10, ],
    k = 10,
    metric = "correlation"
  )
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)
})


test_that("correlation vector conventions agree across dense and sparse data", {
  query <- rbind(
    zero = c(0, 0, 0),
    constants = c(1, 1, 1),
    zero_constant = c(0, 0, 0),
    zero_variable = c(0, 0, 0),
    constant_variable = c(1, 1, 1),
    identical = c(-1, 0, 1),
    opposite = c(-1, 0, 1),
    orthogonal = c(-1, 0, 1)
  )
  reference <- rbind(
    c(0, 0, 0),
    c(2, 2, 2),
    c(2, 2, 2),
    c(-1, 0, 1),
    c(-1, 0, 1),
    c(-1, 0, 1),
    c(1, 0, -1),
    c(1, -2, 1)
  )
  expected <- c(0, 0, 0, 1, 1, 0, 2, 1)
  for (sparse in c(FALSE, TRUE)) {
    for (i in seq_len(nrow(query))) {
      q <- query[i, , drop = FALSE]
      r <- reference[i, , drop = FALSE]
      if (sparse) {
        q <- Matrix::drop0(q)
        r <- Matrix::drop0(r)
      }
      result <- brute_force_knn_query(
        q,
        r,
        k = 1,
        metric = "correlation",
        n_threads = 0
      )
      info <- paste(rownames(query)[i], "sparse =", sparse)
      expect_equal(result$idx, matrix(1L), info = info)
      expect_equal(
        as.vector(result$dist),
        expected[i],
        tolerance = 1e-6,
        info = info
      )
    }
  }

  stored_zero <- Matrix::sparseMatrix(i = 1, j = 1, x = 0, dims = c(1, 3))
  result <- brute_force_knn_query(
    stored_zero,
    stored_zero,
    k = 1,
    metric = "correlation",
    n_threads = 0
  )
  expect_equal(result$idx, matrix(1L))
  expect_equal(result$dist, matrix(0))
})


test_that("correlation distances and neighbors survive positive rescaling", {
  reference <- rbind(c(1, -2, 1), c(-1, 0, 1), c(1, 0, -1))
  query <- matrix(c(-1, 0, 1), nrow = 1)
  for (scale in c(1, 1e10, 1e-15)) {
    for (sparse in c(FALSE, TRUE)) {
      ref <- reference * scale
      qry <- query * scale
      if (sparse) {
        ref <- Matrix::sparseMatrix(
          i = row(ref),
          j = col(ref),
          x = c(ref),
          dims = dim(ref)
        )
        qry <- Matrix::sparseMatrix(
          i = row(qry),
          j = col(qry),
          x = c(qry),
          dims = dim(qry)
        )
      }
      info <- paste("scale =", scale, "sparse =", sparse)
      result <- brute_force_knn_query(
        qry,
        ref,
        k = 3,
        metric = "correlation",
        n_threads = 0
      )
      expect_equal(result$idx, matrix(c(2L, 1L, 3L), nrow = 1), info = info)
      expect_equal(
        result$dist,
        matrix(c(0, 1, 2), nrow = 1),
        tolerance = 1e-6,
        info = info
      )
      nearest <- brute_force_knn_query(
        qry,
        ref,
        k = 1,
        metric = "correlation",
        n_threads = 0
      )
      expect_equal(nearest$idx, matrix(2L), info = info)
      expect_equal(nearest$dist, matrix(0), tolerance = 1e-6, info = info)
    }
  }
})

test_that("sparse correlation preserves zero-centered-product membership", {
  x <- c(1, 2, 2, 0, 0, 1, 1, 1)
  y <- c(2, 1, 2, 1, 0, 0, 1, 1)
  # Shared columns with zero centered products are absent from product indices.
  # Preserve the established sparse result, including when the inputs swap.
  for (pair in list(list(x, y), list(y, x), list(x, x))) {
    expected <- if (identical(pair[[1]], pair[[2]])) 0 else 1
    for (obs in c("R", "C")) {
      q <- Matrix::drop0(matrix(pair[[1]], nrow = 1))
      r <- Matrix::drop0(matrix(pair[[2]], nrow = 1))
      if (obs == "C") {
        q <- Matrix::t(q)
        r <- Matrix::t(r)
      }
      for (n_threads in c(0L, 2L)) {
        result <- brute_force_knn_query(
          q,
          r,
          k = 1,
          metric = "correlation",
          obs = obs,
          n_threads = n_threads
        )
        expect_identical(result$dist, matrix(expected))
        expect_equal(result$idx, matrix(1L))
      }
    }
  }
})

# correlation-preprocess -------------------------------------------------------

context("Correlation-Preprocess distance")

test_that("preprocessed correlation preserves neighbors and distances", {
  res <- brute_force_knn(
    uirism[1:10, ],
    k = 10,
    metric = "correlation-preprocess"
  )
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)

  res <- random_knn(uirism[1:10, ], k = 10, metric = "correlation-preprocess")
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)

  res <- nnd_knn(uirism[1:10, ], k = 10, metric = "correlation-preprocess")
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)

  res <- graph_knn_query(
    reference = uirism[1:10, ],
    query = uirism[1:10, ],
    reference_graph = res,
    k = 10,
    metric = "correlation-preprocess"
  )
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)

  res <- brute_force_knn_query(
    reference = uirism[1:10, ],
    query = uirism[1:10, ],
    k = 10,
    metric = "correlation-preprocess"
  )
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)

  res <- random_knn_query(
    reference = uirism[1:10, ],
    query = uirism[1:10, ],
    k = 10,
    metric = "correlation-preprocess"
  )
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)
  expect_equal(res$idx, cor_index, check.attributes = FALSE)
  expect_equal(res$dist, cor_dist, check.attributes = FALSE, tolerance = 1e-6)
})
