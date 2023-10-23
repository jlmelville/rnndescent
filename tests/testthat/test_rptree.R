library(rnndescent)
context("RP Tree")

set.seed(1337)
res <- rp_tree_knn(ui10, k = 4, leaf_size = 3, n_trees = 1)
expect_equal(dim(res$idx), c(10, 4))
expect_equal(dim(res$dist), c(10, 4))
# knn will (likely) contains missing values with 1 tree
expect_in(c(NA), res$dist)
expect_in(c(0), res$idx)

set.seed(1337)
res <- rp_tree_knn(ui10, k = 4)
expect_equal(res$idx, ui10_nn4$idx, check.attributes = FALSE)
expect_equal(res$dist, ui10_nn4$dist, check.attributes = FALSE, tol = 1e-4)

set.seed(1337)
res <- rp_tree_knn(ui10, k = 4, include_self = FALSE)
expect_equal(res$idx[, 1:3], ui10_nn4$idx[, 2:4], check.attributes = FALSE)
expect_equal(res$dist[, 1:3], ui10_nn4$dist[, 2:4], check.attributes = FALSE, tol = 1e-4)

# euclidean
set.seed(1337)
res <- rp_tree_knn(ui10, k = 4)
expect_equal(res$idx, ui10_nn4$idx, check.attributes = FALSE)
expect_equal(res$dist, ui10_nn4$dist, check.attributes = FALSE, tol = 1e-4)

# cosine
set.seed(1337)
uiris_rnn <- rp_tree_knn(uirism, 15, metric = "cosine", n_trees = 40)
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)


# multi-threading
set.seed(1337)
res <- rp_tree_knn(ui10, k = 4, leaf_size = 3, n_threads = 2, n_trees = 1)
expect_in(c(NA), res$dist)
expect_in(c(0), res$idx)

# euclidean converges
set.seed(1337)
res <- rp_tree_knn(ui10, k = 4, n_threads = 2)
expect_equal(res$idx, ui10_nn4$idx, check.attributes = FALSE)
expect_equal(res$dist, ui10_nn4$dist, check.attributes = FALSE, tol = 1e-4)

set.seed(1337)
uiris_rnn <- rp_tree_knn(uiris, 15, n_trees = 40, n_threads = 2)
expect_equal(sum(uiris_rnn$dist), ui_edsum, tol = 1e-3)

# cosine
set.seed(1337)
uiris_rnn <- rp_tree_knn(uirism, 15, metric = "cosine", n_trees = 40, n_threads = 2)
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)


# R index
expected_rp_tree_index <- list(
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

set.seed(1337)
rp_tree_index <- rnn_build_search_forest(t(ui10), "euclidean", 1, leaf_size = 4)[[1]]
expect_equal(rp_tree_index, expected_rp_tree_index, tol = 1e-7)

set.seed(1337)
tree_search_res <- rnn_tree_search(t(ui10), 4, "euclidean", idx=1, leaf_size = 4)
expected_tree_search_res <- list(
  c(1, 6, 10),
  c(2, 4, 7, 9),
  c(3, 5, 8),
  c(2, 4, 7, 9),
  c(3, 5, 8),
  c(1, 6, 10),
  c(2, 4, 7, 9),
  c(3, 5, 8),
  c(2, 4, 7, 9),
  c(1, 6, 10)
)
for (i in 1:length(tree_search_res)) {
  expect_in(c(i), tree_search_res[[i]])
}
expect_equal(tree_search_res, expected_tree_search_res)
