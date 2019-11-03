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

bitm <- function(nrow, ncol, prob = 0.5) {
  matrix(rbinom(n = nrow * ncol, size = 1, prob = prob), ncol = ncol)
}

set.seed(1337)
bitdata <- bitm(nrow = 10, ncol = 160)
expect_equal(sum(bitdata), 790)

set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4,
  use_cpp = FALSE,
  metric = "hamming", verbose = TRUE
)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE)

# For some reason, get a different set of random numbers when run via testthat
# vs the same code in the console: currently code doesn't rely on RNG as much,
# but if issues return, possibly max_candidates needs to be increased.
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4,
  use_cpp = TRUE,
  metric = "hamming"
)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE)

set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4,
  use_cpp = TRUE, low_memory = FALSE,
  metric = "hamming"
)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE)

# fast rand
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4,
  use_cpp = TRUE,
  metric = "hamming",
  fast_rand = TRUE,
)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE)

set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4,
  use_cpp = TRUE, low_memory = FALSE,
  metric = "hamming",
  fast_rand = TRUE
)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE)

# multi-threading
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4,
  use_cpp = TRUE,
  metric = "hamming",
  n_threads = 1
)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE)


# multi-threading
set.seed(1337)
bit_rnn <- nnd_knn(bitdata, 4,
                   use_cpp = TRUE,
                   metric = "hamming",
                   n_threads = 1, low_memory = FALSE
)
expect_equal(bit_rnn$idx, expected_hamm_idx, check.attributes = FALSE)
expect_equal(bit_rnn$dist, expected_hamm_dist, check.attributes = FALSE)
