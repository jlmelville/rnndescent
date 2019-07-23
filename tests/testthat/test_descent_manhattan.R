library(rnndescent)
context("NN descent Manhattan")

# Manhattan
set.seed(1337)
juirism <- jitter(uirism)
set.seed(1337); juiris_rnn <- nnd_knn(juirism, 15, use_cpp = TRUE, metric = "manhattan")
# expected sum from Annoy
expect_equal(sum(juiris_rnn$dist), 1674.102, tol = 1e-3)


# Manhattan
set.seed(1337); juiris_rnn <- nnd_knn(juirism, 15, use_cpp = TRUE, use_set = TRUE, metric = "manhattan")
# expected sum from Annoy
expect_equal(sum(juiris_rnn$dist), 1674.102, tol = 1e-3)


