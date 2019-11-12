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

# NB Annoy and HNSW don't agree to more than this # of decimal places
# sum(RcppHNSW::hnsw_search(ui4, RcppHNSW::hnsw_build(ui6, distance = "cosine"), k = 4)$dist)
ui4q_dsum <- 0.02072
# sum(RcppHNSW::hnsw_search(ui6, RcppHNSW::hnsw_build(ui4, distance = "cosine"), k = 4)$dist)
ui6q_dsum <-  0.04220

set.seed(1337)
ui6_nnd <- nnd_knn(ui6, k = 4, metric = "cosine")
qnbrs4 <- nnd_knn_query(reference = ui6, reference_idx = ui6_nnd$idx, query = ui4, k = 4, metric = "cosine")
expect_equal(sum(qnbrs4$dist), ui4q_dsum, tol = 1e-5)

set.seed(1337)
ui4_nnd <- nnd_knn(ui4, k = 4, metric = "cosine")
qnbrs6 <- nnd_knn_query(reference = ui4, reference_idx = ui4_nnd$idx, query = ui6, k = 4, metric = "cosine")
expect_equal(sum(qnbrs6$dist), ui6q_dsum, tol = 1e-5)
