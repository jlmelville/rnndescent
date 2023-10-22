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
