library(rnndescent)
context("NN descent Manhattan")

expected_sum <- 1674.102

# Manhattan
set.seed(1337)
juirism <- jitter(uirism)
set.seed(1337)
juiris_rnn <- nnd_knn(juirism, 15, metric = "manhattan")
# expected sum from Annoy
expect_equal(sum(juiris_rnn$dist), expected_sum, tol = 1e-3)

# high memory
set.seed(1337)
juiris_rnn <- nnd_knn(juirism, 15, low_memory = FALSE, metric = "manhattan")
# expected sum from Annoy
expect_equal(sum(juiris_rnn$dist), expected_sum, tol = 1e-3)

# multi-threading
set.seed(1337)
juiris_rnn <- nnd_knn(juirism, 15, metric = "manhattan", n_threads = 1)
expect_equal(sum(juiris_rnn$dist), expected_sum, tol = 1e-3)

# high memory + multi-threading
set.seed(1337)
juiris_rnn <- nnd_knn(juirism, 15,
  metric = "manhattan", n_threads = 1,
  low_memory = FALSE
)
expect_equal(sum(juiris_rnn$dist), expected_sum, tol = 1e-3)

# queries

context("NN descent Manhattan queries")

set.seed(1337)
ui6_nnd <- nnd_knn(ui6, k = 4, metric = "manhattan")
qnbrs4 <- nnd_knn_query(reference = ui6, reference_idx = ui6_nnd$idx, query = ui4, k = 4, metric = "manhattan")
check_query_nbrs_idx(qnbrs4$idx, nref = nrow(ui6))
expect_equal(sum(qnbrs4$dist), ui4q_mdsum, tol = 1e-6)

set.seed(1337)
ui4_nnd <- nnd_knn(ui4, k = 4, metric = "manhattan")
qnbrs6 <- nnd_knn_query(reference = ui4, reference_idx = ui4_nnd$idx, query = ui6, k = 4, metric = "manhattan")
check_query_nbrs_idx(qnbrs6$idx, nref = nrow(ui4))
expect_equal(sum(qnbrs6$dist), ui6q_mdsum, tol = 1e-6)
