library(rnndescent)
context("NN descent cosine")

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
