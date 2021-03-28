library(rnndescent)
context("NN descent Hamming")

# Hamming
# from Annoy
expected_hamm_idx <- matrix(
  c(
    1, 7, 4, 5,
    2, 10, 3, 9,
    3, 4, 2, 7,
    4, 3, 1, 7,
    5, 6, 7, 1,
    6, 5, 10, 3,
    7, 1, 10, 5,
    8, 9, 10, 7,
    9, 8, 10, 4,
    10, 2, 9, 7
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

expected_hamm_dist <- matrix(
  c(
    0, 72, 74, 77,
    0, 69, 78, 79,
    0, 65, 78, 79,
    0, 65, 74, 76,
    0, 67, 75, 77,
    0, 67, 80, 81,
    0, 72, 74, 75,
    0, 69, 77, 81,
    0, 69, 72, 78,
    0, 69, 72, 74
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)


expect_equal(sum(bitdata), 790)

# For some reason, get a different set of random numbers when run via testthat
# vs the same code in the console: currently code doesn't rely on RNG as much,
# but if issues return, possibly max_candidates needs to be increased.
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4, metric = "hamming")
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE)

# high memory
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4, low_memory = FALSE, metric = "hamming")
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE)

# multi-threading
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4, metric = "hamming", n_threads = 1)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE)

# multi-threading high memory
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4,
  metric = "hamming", n_threads = 1,
  low_memory = FALSE
)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE)


# queries

context("NN descent Hamming queries")

set.seed(1337)
bit6_nnd <- nnd_knn(bit6, k = 4, metric = "hamming")
qnbrs4 <- nnd_knn_query(reference = bit6, reference_nn = bit6_nnd, query = bit4, k = 4, metric = "hamming")
check_query_nbrs_idx(qnbrs4$idx, nref = nrow(bit6))
expect_equal(sum(qnbrs4$dist), bit4q_hdsum, tol = 1e-6)

set.seed(1337)
bit4_nnd <- nnd_knn(bit4, k = 4, metric = "manhattan")
qnbrs6 <- nnd_knn_query(reference = bit4, reference_nn = bit4_nnd, query = bit6, k = 4, metric = "hamming")
check_query_nbrs_idx(qnbrs6$idx, nref = nrow(ui4))
expect_equal(sum(qnbrs6$dist), bit6q_hdsum, tol = 1e-6)
