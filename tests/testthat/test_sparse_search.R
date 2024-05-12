library(rnndescent)
context("Sparse search/build")

#14: Alt metric gave the wrong results (because of a bad uncorrection)

# [1,] . . 1 . . . . . 1 1 . 1 . 1 . 1 1 2 1
# [2,] 1 1 . 1 1 1 1 1 . . 1 . 1 . 1 . . . .
# [3,] . . . . . . . . . . . . . . . . . . 1

xi <- c(
  1L, 1L, 0L, 1L, 1L, 1L, 1L, 1L, 0L, 0L, 1L, 0L, 1L, 0L, 1L,
  0L, 0L, 0L, 0L, 2L
)
xj <- c(
  0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L, 13L,
  14L, 15L, 16L, 17L, 18L, 18L
)
xx <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1)
xsp <- Matrix::sparseMatrix(
  i = xi + 1,
  j = xj + 1,
  x = xx,
  dims = c(3, 19),
)
xd <- as.matrix(xsp)

# [1,] 1 1 . . 1 1 1 1 . . 1 . 1 . 1 . . . .
# [2,] 1 1 . 1 1 1 1 1 . . 1 . 1 . 1 . . . .
# [3,] 1 1 . . 1 1 1 1 . . 1 . 1 . 1 . . . .
# [4,] 1 1 . . 1 . 1 . . . 1 . . . 1 . . . .
# [5,] . . 1 . . . . . 1 1 . 1 . 1 . 1 1 2 1
# [6,] . . 1 . . . . . 1 1 . 1 . 1 . . 1 2 1
# [7,] . . 1 . . . . . 1 1 . 1 . 1 . 1 1 2 1
# [8,] . . . . . . . . 1 1 . . . 1 . . . 1 .

yi <- c(
  0L, 1L, 2L, 3L, 0L, 1L, 2L, 3L, 4L, 5L, 6L, 1L, 0L, 1L, 2L,
  3L, 0L, 1L, 2L, 0L, 1L, 2L, 3L, 0L, 1L, 2L, 4L, 5L, 6L, 7L, 4L,
  5L, 6L, 7L, 0L, 1L, 2L, 3L, 4L, 5L, 6L, 0L, 1L, 2L, 4L, 5L, 6L,
  7L, 0L, 1L, 2L, 3L, 4L, 6L, 4L, 5L, 6L, 4L, 5L, 6L, 7L, 4L, 5L,
  6L
)
yj <- c(
  0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L, 2L, 2L, 2L, 3L, 4L, 4L, 4L,
  4L, 5L, 5L, 5L, 6L, 6L, 6L, 6L, 7L, 7L, 7L, 8L, 8L, 8L, 8L, 9L,
  9L, 9L, 9L, 10L, 10L, 10L, 10L, 11L, 11L, 11L, 12L, 12L, 12L,
  13L, 13L, 13L, 13L, 14L, 14L, 14L, 14L, 15L, 15L, 16L, 16L, 16L,
  17L, 17L, 17L, 17L, 18L, 18L, 18L
)
yx <- c(
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1,
  1, 1
)
ysp <- Matrix::sparseMatrix(
  i = yi + 1,
  j = yj + 1,
  x = yx,
  dims = c(8, 19),
)
yd <- as.matrix(ysp)

index_graph_idx <- matrix(c(
  1, 3, 2,
  2, 3, 1,
  3, 1, 2
), nrow = 3, byrow = TRUE)

index_graph_dist <- matrix(c(
  0, 0.711325, 1,
  0, 1, 1,
  0, 0.711325, 1
), nrow = 3, byrow = TRUE)

search_graph <- Matrix::sparseMatrix(
  i = c(1L, 2L, 0L, 0L, 1L) + 1,
  j = c(0L, 0L, 1L, 2L, 2L) + 1,
  x = c(1, 0.7113249, 1, 0.7113249, 1),
  dims = c(3, 3),
)

k3_idx <- matrix(c(
  2, 1, 3,
  2, 1, 3,
  2, 1, 3,
  2, 1, 3,
  1, 3, 2,
  1, 3, 2,
  1, 3, 2,
  1, 2, 3
), nrow = 8, byrow = TRUE)

k3_dist <- matrix(c(
  5.131668e-02, 1.0000000, 1,
  -1.192093e-07, 1.0000000, 1,
  5.131668e-02, 1.0000000, 1,
  2.254034e-01, 1.0000000, 1,
  1.192093e-07, 0.7113249, 1,
  4.257292e-02, 0.6984887, 1,
  1.192093e-07, 0.7113249, 1,
  2.783122e-01, 1.0000000, 1
), nrow = 8, byrow = TRUE)

test_that("sparse no alt metric", {
  set.seed(2024)
  nndes_index <- rnnd_build(data = xsp, k = 3, metric = "cosine", use_alt_metric = FALSE)
  expect_equal(nndes_index$graph$idx, index_graph_idx)
  expect_equal(nndes_index$graph$dist, index_graph_dist, tol = 1e-6)
  expect_equal(nndes_index$search_graph, search_graph, tol = 1e-7)

  k1_query <- rnnd_query(nndes_index, query = ysp, k = 1)
  expect_equal(k1_query$idx, k3_idx[, 1, drop = FALSE])
  expect_equal(k1_query$dist, k3_dist[, 1, drop = FALSE], tol = 1e-6)

  k2_query <- rnnd_query(nndes_index, query = ysp, k = 2)
  expect_equal(k2_query$idx, k3_idx[, 1:2])
  expect_equal(k2_query$dist, k3_dist[, 1:2], tol = 1e-6)

  k3_query <- rnnd_query(nndes_index, query = ysp, k = 3)
  expect_equal(k3_query$idx, k3_idx)
  expect_equal(k3_query$dist, k3_dist, tol = 1e-6)
})

test_that("dense no alt metric", {
  set.seed(2024)
  nndes_index <- rnnd_build(data = xd, k = 3, metric = "cosine", use_alt_metric = FALSE)
  expect_equal(nndes_index$graph$idx, index_graph_idx)
  expect_equal(nndes_index$graph$dist, index_graph_dist, tol = 1e-6)
  expect_equal(nndes_index$search_graph, search_graph, tol = 1e-7)

  k1_query <- rnnd_query(nndes_index, query = yd, k = 1)
  expect_equal(k1_query$idx, k3_idx[, 1, drop = FALSE])
  expect_equal(k1_query$dist, k3_dist[, 1, drop = FALSE], tol = 1e-6)

  k2_query <- rnnd_query(nndes_index, query = yd, k = 2)
  expect_equal(k2_query$idx, k3_idx[, 1:2])
  expect_equal(k2_query$dist, k3_dist[, 1:2], tol = 1e-6)

  k3_query <- rnnd_query(nndes_index, query = yd, k = 3)
  expect_equal(k3_query$idx, k3_idx)
  expect_equal(k3_query$dist, k3_dist, tol = 1e-6)
})

test_that("sparse alt metric", {
  set.seed(2024)
  nndes_index <- rnnd_build(data = xsp, k = 3, metric = "cosine", use_alt_metric = TRUE)
  expect_equal(nndes_index$graph$idx, index_graph_idx)
  expect_equal(nndes_index$graph$dist, index_graph_dist, tol = 1e-6)
  expect_equal(nndes_index$search_graph, search_graph, tol = 1e-7)

  k1_query <- rnnd_query(nndes_index, query = ysp, k = 1)
  expect_equal(k1_query$idx, k3_idx[, 1, drop = FALSE])
  expect_equal(k1_query$dist, k3_dist[, 1, drop = FALSE], tol = 1e-6)

  k2_query <- rnnd_query(nndes_index, query = ysp, k = 2)
  expect_equal(k2_query$idx, k3_idx[, 1:2])
  expect_equal(k2_query$dist, k3_dist[, 1:2], tol = 1e-6)

  k3_query <- rnnd_query(nndes_index, query = ysp, k = 3)
  expect_equal(k3_query$idx, k3_idx)
  expect_equal(k3_query$dist, k3_dist, tol = 1e-6)
})

test_that("dense alt metric", {
  set.seed(2024)
  nndes_index <- rnnd_build(data = xd, k = 3, metric = "cosine", use_alt_metric = TRUE)
  expect_equal(nndes_index$graph$idx, index_graph_idx)
  expect_equal(nndes_index$graph$dist, index_graph_dist, tol = 1e-6)
  expect_equal(nndes_index$search_graph, search_graph, tol = 1e-7)

  k1_query <- rnnd_query(nndes_index, query = yd, k = 1)
  expect_equal(k1_query$idx, k3_idx[, 1, drop = FALSE])
  expect_equal(k1_query$dist, k3_dist[, 1, drop = FALSE], tol = 1e-6)

  k2_query <- rnnd_query(nndes_index, query = yd, k = 2)
  expect_equal(k2_query$idx, k3_idx[, 1:2])
  expect_equal(k2_query$dist, k3_dist[, 1:2], tol = 1e-6)

  k3_query <- rnnd_query(nndes_index, query = yd, k = 3)
  expect_equal(k3_query$idx, k3_idx)
  expect_equal(k3_query$dist, k3_dist, tol = 1e-6)
})

test_that("k=2 builds", {
  set.seed(2024)
  k2dtg <- rnnd_build(data = xd[1:2, ], k = 2, metric = "cosine", use_alt_metric = TRUE)$graph
  k2stg <- rnnd_build(data = xsp[1:2, ], k = 2, metric = "cosine", use_alt_metric = TRUE)$graph
  expect_equal(k2dtg$idx, k2stg$idx)
  expect_equal(k2dtg$dist, k2stg$dist)

  k2dfg <- rnnd_build(data = xd[1:2, ], k = 2, metric = "cosine", use_alt_metric = FALSE)$graph
  k2sfg <- rnnd_build(data = xsp[1:2, ], k = 2, metric = "cosine", use_alt_metric = FALSE)$graph
  expect_equal(k2dfg$idx, k2sfg$idx)
  expect_equal(k2dfg$dist, k2sfg$dist)
  expect_equal(k2dfg$idx, k2dtg$idx)
})
