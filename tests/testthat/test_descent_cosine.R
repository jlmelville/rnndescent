library(rnndescent)
context("NN descent Cosine")

# Cosine distance
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, metric = "cosine")
# expected sum from RcppHNSW
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)

# Cosine distance
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, low_memory = FALSE, metric = "cosine")
# expected sum from RcppHNSW
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)

# multi-threading
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, metric = "cosine", n_threads = 1)
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)

# with caching
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, metric = "cosine", low_memory = FALSE, n_threads = 1)
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)

# Query

context("NN descent Cosine queries")

set.seed(1337)
ui6_nnd <- nnd_knn(ui6, k = 4, metric = "cosine")
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, k = 4, metric = "cosine")
check_query_nbrs_idx(qnbrs4$idx, nref = nrow(ui6))
expect_equal(sum(qnbrs4$dist), ui4q_cdsum, tol = 1e-5)

set.seed(1337)
ui4_nnd <- nnd_knn(ui4, k = 4, metric = "cosine")
qnbrs6 <- graph_knn_query(reference = ui4, reference_graph = ui4_nnd, query = ui6, k = 4, metric = "cosine")
check_query_nbrs_idx(qnbrs6$idx, nref = nrow(ui4))
expect_equal(sum(qnbrs6$dist), ui6q_cdsum, tol = 1e-5)
