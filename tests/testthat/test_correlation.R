library(rnndescent)
context("Correlation distance")

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

res <- brute_force_knn(uirism[1:10, ], k = 10, metric = "correlation")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)

res <- random_knn(uirism[1:10, ], k = 10, metric = "correlation")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)

res <- nnd_knn(uirism[1:10, ], k = 10, metric = "correlation")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)

res <- graph_knn_query(reference = uirism[1:10, ], query = uirism[1:10, ], reference_graph = res, k = 10, metric = "correlation")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)

res <- brute_force_knn_query(reference = uirism[1:10, ], query = uirism[1:10, ], k = 10, metric = "correlation")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)

res <- random_knn_query(reference = uirism[1:10, ], query = uirism[1:10, ], k = 10, metric = "correlation")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)


# correlation-preprocess -------------------------------------------------------

context("Correlation-Preprocess distance")

res <- brute_force_knn(uirism[1:10, ], k = 10, metric = "correlation-preprocess")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)

res <- random_knn(uirism[1:10, ], k = 10, metric = "correlation-preprocess")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)

res <- nnd_knn(uirism[1:10, ], k = 10, metric = "correlation-preprocess")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)

res <- graph_knn_query(reference = uirism[1:10, ], query = uirism[1:10, ], reference_graph = res, k = 10, metric = "correlation-preprocess")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)

res <- brute_force_knn_query(reference = uirism[1:10, ], query = uirism[1:10, ], k = 10, metric = "correlation-preprocess")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)

res <- random_knn_query(reference = uirism[1:10, ], query = uirism[1:10, ], k = 10, metric = "correlation-preprocess")
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)
expect_equal(res$idx, cor_index, check.attributes = FALSE)
expect_equal(res$dist, cor_dist, check.attributes = FALSE, tol = 1e-6)
