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

context("Query Cosine")

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


# cosine-preprocess -------------------------------------------------------------

context("NN descent Cosine-Preprocess")

# Cosine distance
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, metric = "cosine-preprocess")
# expected sum from RcppHNSW
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)

# Cosine distance
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, low_memory = FALSE, metric = "cosine-preprocess")
# expected sum from RcppHNSW
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)

# multi-threading
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, metric = "cosine-preprocess", n_threads = 1)
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)

# with caching
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, metric = "cosine-preprocess", low_memory = FALSE, n_threads = 1)
expect_equal(sum(uiris_rnn$dist), 1.347357, tol = 1e-3)

# Query

context("Cosine-Preprocess queries")

set.seed(1337)
ui6_nnd <- nnd_knn(ui6, k = 4, metric = "cosine-preprocess")
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, k = 4, metric = "cosine-preprocess")
check_query_nbrs_idx(qnbrs4$idx, nref = nrow(ui6))
expect_equal(sum(qnbrs4$dist), ui4q_cdsum, tol = 1e-5)

set.seed(1337)
ui4_nnd <- nnd_knn(ui4, k = 4, metric = "cosine-preprocess")
qnbrs6 <- graph_knn_query(reference = ui4, reference_graph = ui4_nnd, query = ui6, k = 4, metric = "cosine-preprocess")
check_query_nbrs_idx(qnbrs6$idx, nref = nrow(ui4))
expect_equal(sum(qnbrs6$dist), ui6q_cdsum, tol = 1e-5)

# metric correction

# random numbers including 0 and 1
alt_cos <- matrix(c(
 0.094763154, 0.00988037,
 0.028532648, 0.02815395,
 1.000000000, 0.46909320,
 0.001943944, 0.09845309,
 0.148561513, 0.00000000
), nrow = 5, byrow = TRUE)

# results from pynndescent.sparse.sparse_correct_alternative_cosine
cor_cos <- matrix(
  c(0.06357403, 0.00682515,
    0.01958304, 0.01932565,
    0.5       , 0.27758147,
    0.00134653, 0.06596604,
    0.09785047, 0.0),
  nrow = 5, byrow = TRUE)

expect_equal(apply_sparse_alt_metric_correction("cosine", alt_cos), cor_cos)
expect_equal(apply_sparse_alt_metric_uncorrection("cosine", cor_cos), alt_cos)

# sparse
test_that("sparse", {
  set.seed(1337); dznbrs <- nnd_knn(ui10z, k = 4, n_threads = 0, metric = "cosine")
  set.seed(1337); spnbrs <- nnd_knn(ui10sp, k = 4, n_threads = 0, metric = "cosine", use_alt_metric = TRUE)
  expect_equal(dznbrs, spnbrs, tol = 1e-5)
  set.seed(1337); spnbrs <- nnd_knn(ui10sp, k = 4, n_threads = 0, metric = "cosine", use_alt_metric = FALSE)
  expect_equal(dznbrs, spnbrs, tol = 1e-5)

  # make sure uncorrection is triggered: nnd with incorrect distances will not converge
  bfz <- brute_force_knn(ui10z, k = 4, metric = "cosine")
  set.seed(1337); cosrz <- random_knn(ui10z, k = 4, metric = "cosine")
  set.seed(1337); spannd <- nnd_knn(ui10sp, k = 4, metric = "cosine", init = cosrz, use_alt_metric = TRUE)
  set.seed(1337); spcnnd <- nnd_knn(ui10sp, k = 4, metric = "cosine", init = cosrz, use_alt_metric = FALSE)

  expect_equal(spannd, bfz, tol = 1e-3)
  expect_equal(spcnnd, bfz, tol = 1e-4)

  # make sure init graph can be prepared from sparse
  cosrz$dist <- NULL
  set.seed(1337); spannd <- nnd_knn(ui10sp, k = 4, metric = "cosine", init = cosrz, use_alt_metric = TRUE)
  set.seed(1337); spcnnd <- nnd_knn(ui10sp, k = 4, metric = "cosine", init = cosrz, use_alt_metric = FALSE)

  expect_equal(spannd, bfz, tol = 1e-5)
  expect_equal(spcnnd, bfz, tol = 1e-5)
})
