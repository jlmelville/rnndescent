library(rnndescent)
context("NN descent cosine")

# Cosine distance
set.seed(1337); uiris_rnn <- nnd_knn(uirism, 15, use_cpp = TRUE, metric = "cosine")
# expected sum from RcppHNSW
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)

# Cosine distance
set.seed(1337); uiris_rnn <- nnd_knn(uirism, 15, use_cpp = TRUE, use_set = TRUE, metric = "cosine")
# expected sum from RcppHNSW
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)
