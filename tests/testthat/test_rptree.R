library(rnndescent)
context("RP Tree")

# just one tree
expected_rpt_knn <- list(
  idx = matrix(
    c(
      1, 6, 10, 0,
      2, 7, 4, 9,
      3, 5, 8, 0,
      4, 9, 2, 7,
      5, 8, 3, 0,
      6, 1, 10, 0,
      7, 2, 4, 9,
      8, 5, 3, 0,
      9, 4, 2, 7,
      10, 6, 1, 0
    ),
    nrow = 10,
    byrow = TRUE
  ),
  dist = matrix(
    c(
      0, 0.3464102, 0.6782330, NA,
      0, 0.3000002, 0.5477225, 0.6164413,
      0, 0.3316626, 0.4690416, NA,
      0, 0.3464101, 0.5477225, 0.6708205,
      0, 0.1732050, 0.3316626, NA,
      0, 0.3464102, 0.5830952, NA,
      0, 0.3000002, 0.6708205, 0.8185353,
      0, 0.1732050, 0.4690416, NA,
      0, 0.3464101, 0.6164413, 0.8185353,
      0, 0.5830952, 0.6782330, NA
    ),
    nrow = 10,
    byrow = TRUE
  )
)

set.seed(1337)
res <- rpf_knn(ui10, k = 4, leaf_size = 4, n_trees = 1)
expect_equal(res, expected_rpt_knn, tol = 1e-7)

set.seed(1337)
res <- rpf_knn(ui10, k = 4)
expect_equal(res$idx, ui10_nn4$idx, check.attributes = FALSE)
expect_equal(res$dist, ui10_nn4$dist, check.attributes = FALSE, tol = 1e-4)

set.seed(1337)
res <- rpf_knn(ui10, k = 4, include_self = FALSE)
expect_equal(res$idx[, 1:3], ui10_nn4$idx[, 2:4], check.attributes = FALSE)
expect_equal(res$dist[, 1:3], ui10_nn4$dist[, 2:4], check.attributes = FALSE, tol = 1e-4)

# euclidean
set.seed(1337)
res <- rpf_knn(ui10, k = 4)
expect_equal(res$idx, ui10_nn4$idx, check.attributes = FALSE)
expect_equal(res$dist, ui10_nn4$dist, check.attributes = FALSE, tol = 1e-4)

# cosine
set.seed(1337)
uiris_rnn <- rpf_knn(uirism, 15, metric = "cosine", n_trees = 40)
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)


# multi-threading
set.seed(1337)
res <- rpf_knn(ui10, k = 4, leaf_size = 3, n_threads = 2, n_trees = 1)
expect_in(c(NA), res$dist)
expect_in(c(0), res$idx)

set.seed(1337)
res <- rpf_knn(ui10, k = 4, n_threads = 2)
expect_equal(res$idx, ui10_nn4$idx, check.attributes = FALSE)
expect_equal(res$dist, ui10_nn4$dist, check.attributes = FALSE, tol = 1e-4)

# euclidean converges
set.seed(1337)
uiris_rnn <- rpf_knn(uiris, 15, n_trees = 40, n_threads = 2)
expect_equal(sum(uiris_rnn$dist), ui_edsum, tol = 1e-3)

# cosine
set.seed(1337)
uiris_rnn <- rpf_knn(uirism, 15, metric = "cosine", n_trees = 40, n_threads = 2)
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)

# R index
expected_rpf_index <- list(
    list(
    hyperplanes = matrix(c(
      -0.5000000, -0.8000002, -0.2, -0.3,
      0.3000002, -0.3000002,  0.1, -0.2,
      0.0000000,  0.0000000,  0.0,  0.0,
      0.0000000,  0.0000000,  0.0,  0.0,
      0.0000000,  0.0000000,  0.0,  0.0
    ), nrow = 5, byrow = TRUE),
    offsets = c(5.7700009, -0.5550003, NA, NA, NA),
    children = matrix(c(
      1, 4,
      2, 3,
      0, 3,
      3, 7,
      7, 10
    ), nrow = 5, byrow = TRUE),
    indices = c(2, 4, 7, 1, 3, 6, 8, 0, 5, 9),
    leaf_size = 4
  )
)

set.seed(1337)
rpf_index <- rpf_build(ui10, "euclidean", 1, leaf_size = 4)
expect_equal(rpf_index, expected_rpf_index, tol = 1e-7)

# query data against itself to reproduce knn (just more slowly)
set.seed(1337)
rpf_query_res <-
  rpf_knn_query(
    ui10,
    ui10,
    rpf_index,
    k = 4,
    metric = "euclidean",
    n_threads = 0,
    cache = TRUE
  )

expect_equal(rpf_query_res, expected_rpt_knn, tol = 1e-7)

set.seed(1337)
rpf_query_res <-
  rpf_knn_query(
    ui10,
    ui10,
    rpf_index,
    k = 4,
    metric = "euclidean",
    n_threads = 0,
    cache = FALSE
  )
expect_equal(rpf_query_res, expected_rpt_knn, tol = 1e-7)

