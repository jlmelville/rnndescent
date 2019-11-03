library(rnndescent)
context("NN descent Manhattan")

expected_sum <- 1674.102

# Manhattan
set.seed(1337)
juirism <- jitter(uirism)
set.seed(1337)
juiris_rnn <- nnd_knn(juirism, 15, use_cpp = TRUE, metric = "manhattan")
# expected sum from Annoy
expect_equal(sum(juiris_rnn$dist), expected_sum, tol = 1e-3)


# Manhattan
set.seed(1337)
juiris_rnn <- nnd_knn(juirism, 15, use_cpp = TRUE, low_memory = FALSE, metric = "manhattan")
# expected sum from Annoy
expect_equal(sum(juiris_rnn$dist), expected_sum, tol = 1e-3)


# fast rand
set.seed(1337)
juiris_rnn <- nnd_knn(juirism, 15, use_cpp = TRUE, metric = "manhattan", fast_rand = TRUE)
expect_equal(sum(juiris_rnn$dist), expected_sum, tol = 1e-3)

# multi-threading
set.seed(1337)
juiris_rnn <- nnd_knn(juirism, 15, use_cpp = TRUE, metric = "manhattan", n_threads = 1)
expect_equal(sum(juiris_rnn$dist), expected_sum, tol = 1e-3)

# high memory
set.seed(1337)
juiris_rnn <- nnd_knn(juirism, 15, use_cpp = TRUE, metric = "manhattan", n_threads = 1, low_memory = FALSE)
expect_equal(sum(juiris_rnn$dist), expected_sum, tol = 1e-3)
