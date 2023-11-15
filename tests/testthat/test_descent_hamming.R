library(rnndescent)

context("NN descent hamming")

# For some reason, get a different set of random numbers when run via testthat
# vs the same code in the console: currently code doesn't rely on RNG as much,
# but if issues return, possibly max_candidates needs to be increased
# Update 10/27/23: yeah this seems to be a RNG issue
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4, metric = "hamming", max_candidates = 10)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE, tol = 1e-7)

# high memory
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4,
  low_memory = FALSE, metric = "hamming",
  max_candidates = 10
)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE, tol = 1e-7)

# multi-threading
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4,
  metric = "hamming", n_threads = 1,
  max_candidates = 10
)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE, tol = 1e-7)

# multi-threading high memory
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4,
  metric = "hamming", n_threads = 1, ,
  max_candidates = 10,
  low_memory = FALSE
)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE, tol = 1e-7)

# queries

context("Hamming queries")

set.seed(1337)
bit6_nnd <- nnd_knn(bit6, k = 4, metric = "hamming")
qnbrs4 <- graph_knn_query(reference = bit6, reference_graph = bit6_nnd, query = bit4, k = 4, metric = "hamming")
check_query_nbrs_idx(qnbrs4$idx, nref = nrow(bit6))
expect_equal(sum(qnbrs4$dist), bit4q_hdsum, tol = 1e-6)

set.seed(1337)
bit4_nnd <- nnd_knn(bit4, k = 4, metric = "hamming")
qnbrs6 <- graph_knn_query(reference = bit4, reference_graph = bit4_nnd, query = bit6, k = 4, metric = "hamming")
check_query_nbrs_idx(qnbrs6$idx, nref = nrow(ui4))
expect_equal(sum(qnbrs6$dist), bit6q_hdsum, tol = 1e-6)

test_that("prepare search graph with hamming and explicit zero", {
  set.seed(1337)
  bit6_nnd <- nnd_knn(bit6, k = 4, metric = "hamming")
  bit6_nnd$dist[6, 4] <- 0
  bit6_sgraph <-
    prepare_search_graph(bit6, bit6_nnd, metric = "hamming", n_threads = 1)
  qnbrs4 <-
    graph_knn_query(
      reference = bit6,
      reference_graph = bit6_sgraph,
      query = bit4,
      k = 4,
      metric = "hamming",
      n_threads = 1
    )
  check_query_nbrs_idx(qnbrs4$idx, nref = nrow(bit6))
  expect_equal(sum(qnbrs4$dist), bit4q_hdsum, tol = 1e-6)
})
