library(rnndescent)
context("NN descent Euclidean")


i10_rinit <- list(
  dist = matrix(
    c(
      0.812403840463596, 0.346410161513775, 0.994987437106621, 1.45945195193264,
      0.479583152331272, 0.547722557505166, 0.489897948556636, 0.424264068711929,
      0.787400787401181, 0.424264068711929, 0.469041575982343, 0.223606797749979,
      0.346410161513776, 0.547722557505166, 0.424264068711928, 1.45945195193264,
      1.3114877048604, 0.728010988928052, 0.33166247903554, 0.479583152331272,
      1.36747943311773, 0.678232998312527, 0.346410161513775, 0.932737905308882,
      0.678232998312527, 0.818535277187245, 0.458257569495584, 1.2328828005938,
      0.173205080756888, 0.58309518948453, 0.458257569495584, 1.16189500386223,
      0.818535277187245, 0.616441400296897, 1.36747943311773, 0.346410161513776,
      1.80831413200251, 0.58309518948453, 1.04403065089105, 1.36014705087354
    ),
    byrow = TRUE, nrow = 10, ncol = 4
  ),
  idx = matrix(
    c(
      6, 5, 1, 3,
      4, 3, 7, 2,
      3, 1, 7, 6,
      8, 1, 7, 0,
      9, 8, 2, 1,
      8, 6, 0, 7,
      5, 8, 7, 9,
      4, 8, 6, 0,
      6, 1, 5, 3,
      8, 5, 2, 1
    ) + 1,
    byrow = TRUE, nrow = 10, ncol = 4
  )
)

expected_heap_sort_idx <- matrix(
  c(
    5, 6, 1, 7,
    2, 4, 7, 3,
    6, 4, 1, 7,
    8, 7, 1, 2,
    7, 2, 1, 8,
    0, 9, 6, 7,
    2, 7, 5, 0,
    4, 3, 6, 2,
    3, 7, 1, 4,
    5, 2, 6, 4
  ) + 1,
  byrow = TRUE, nrow = 10, ncol = 4
)

expected_heap_sort_dist <- matrix(
  c(
    0.34641016, 0.81240384, 0.99498744, 1.161895,
    0.42426407, 0.47958315, 0.48989795, 0.54772256,
    0.2236068, 0.33166248, 0.42426407, 0.46904158,
    0.34641016, 0.42426407, 0.54772256, 0.78740079,
    0.17320508, 0.33166248, 0.47958315, 0.72801099,
    0.34641016, 0.58309519, 0.678233, 0.93273791,
    0.2236068, 0.45825757, 0.678233, 0.81240384,
    0.17320508, 0.42426407, 0.45825757, 0.46904158,
    0.34641016, 0.58309519, 0.6164414, 0.72801099,
    0.58309519, 1.04403065, 1.2328828, 1.3114877
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

heap_sorted <- nnd_knn(ui10, 4, n_iters = 0, init = i10_rinit)
expect_equal(heap_sorted$idx, expected_heap_sort_idx, check.attributes = FALSE)
expect_equal(heap_sorted$dist, expected_heap_sort_dist, check.attributes = FALSE, tol = 1e-6)

expected_idx <- matrix(
  c(
    1, 6, 10, 3,
    2, 7, 3, 5,
    3, 7, 5, 2,
    4, 9, 8, 2,
    5, 8, 3, 7,
    6, 1, 3, 10,
    7, 3, 2, 5,
    8, 5, 4, 7,
    9, 4, 8, 2,
    10, 6, 1, 3
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

# distances from FNN
expected_dist <- matrix(
  c(
    0, 0.3464102, 0.6782330, 0.7000000,
    0, 0.3000000, 0.4242641, 0.4795832,
    0, 0.2236068, 0.3316625, 0.4242641,
    0, 0.3464102, 0.4242641, 0.5477226,
    0, 0.1732051, 0.3316625, 0.3464102,
    0, 0.3464102, 0.5000000, 0.5830952,
    0, 0.2236068, 0.3000000, 0.3464102,
    0, 0.1732051, 0.4242641, 0.4582576,
    0, 0.3464102, 0.5830952, 0.6164414,
    0, 0.5830952, 0.6782330, 1.0440307
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

set.seed(1337)
output <- capture_everything({
  rnn <- nnd_knn(ui10, 4, verbose = FALSE)
})
expect_equal(rnn$idx, expected_idx, check.attributes = FALSE)
expect_equal(rnn$dist, expected_dist, check.attributes = FALSE, tol = 1e-6)
expect_equal(output, "character(0)")


# default
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15)
# treat sum of distances an objective function
# expected sum from sum(FNN::get.knn(uirism, 14)$nn.dist)
expect_equal(sum(uiris_rnn$dist), 1016.834, tol = 1e-3)

# data frame can be input
set.seed(1337)
uiris_rnn <- nnd_knn(uiris, 15)
expect_equal(sum(uiris_rnn$dist), 1016.834, tol = 1e-3)

# Create external initialization
set.seed(1337)
iris_nbrs <- random_knn(uirism, 15)

# initialize from existing knn graph
set.seed(1337)
iris_nnd <- nnd_knn(uirism, init = iris_nbrs)
expect_equal(sum(iris_nnd$dist), 1016.834, tol = 1e-3)

# Use larger initialization for smaller k
set.seed(1337)
iris_nnd <- nnd_knn(uirism, init = random_knn(uirism, 20), k = 15)
expect_equal(sum(iris_nnd$dist), 1016.834, tol = 1e-3)

# high memory mode
set.seed(1337)
iris_nnd <- nnd_knn(uirism, init = iris_nbrs, low_memory = FALSE)
expect_equal(sum(iris_nnd$dist), 1016.834, tol = 1e-3)

# init default with high memory
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, low_memory = FALSE)
expect_equal(sum(uiris_rnn$dist), 1016.834, tol = 1e-3)

# max candidates
set.seed(1337)
iris_nnd <- nnd_knn(uirism, init = iris_nbrs, max_candidates = 10)
expect_equal(sum(iris_nnd$dist), 1016.834, tol = 1e-3)

# errors
expect_error(nnd_knn(ui10), "provide k")
expect_error(nnd_knn(ui10, k = 11), "k must be")
expect_error(nnd_knn(uirism, init = iris_nbrs, k = 20), "Not enough")
expect_error(nnd_knn(uirism, k = 15, metric = "not-a-real metric"), "metric")

# verbosity
expect_message(capture_everything(nnd_knn(ui10, 4, verbose = TRUE)), "Initializing")

# Multi-threading ---------------------------------------------------------

# multi-threading
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, n_threads = 1)
expect_equal(sum(uiris_rnn$dist), 1016.834, tol = 1e-3)

# with caching
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, n_threads = 1, low_memory = FALSE)
expect_equal(sum(uiris_rnn$dist), 1016.834, tol = 1e-3)

# block_size
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, n_threads = 1, block_size = 3)
expect_equal(sum(uiris_rnn$dist), 1016.834, tol = 1e-3)

# Queries -----------------------------------------------------------------

context("NN descent Euclidean queries")

# sum(RcppHNSW::hnsw_search(ui4, RcppHNSW::hnsw_build(ui6), k = 4)$dist)
ui4q_dsum <- 9.310494
# sum(RcppHNSW::hnsw_search(ui6, RcppHNSW::hnsw_build(ui4), k = 4)$dist)
ui6q_dsum <-  18.98666

set.seed(1337)
ui6_nnd <- nnd_knn(ui6, k = 4)
qnbrs4 <- nnd_knn_query(reference = ui6, reference_idx = ui6_nnd$idx, query = ui4, k = 4)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_dsum)

set.seed(1337)
ui4_nnd <- nnd_knn(ui4, k = 4)
qnbrs6 <- nnd_knn_query(reference = ui4, reference_idx = ui4_nnd$idx, query = ui6, k = 4)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_dsum, tol = 1e-6)

# max candidates
set.seed(1337)
qnbrs6 <- nnd_knn_query(reference = ui4, reference_idx = ui4_nnd$idx, query = ui6, k = 4, max_candidates = 3)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_dsum, tol = 1e-6)

# initialize separately
rnbrs4 <- random_knn_query(reference = ui6, query = ui4, k = 4)
qnbrs4 <- nnd_knn_query(reference = ui6, reference_idx = ui6_nnd$idx, query = ui4, init = rnbrs4)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_dsum)

# initialize separately and reduce graph
rnbrs5 <- random_knn_query(reference = ui6, query = ui4, k = 5)
qnbrs4 <- nnd_knn_query(reference = ui6, reference_idx = ui6_nnd$idx, query = ui4, init = rnbrs5, k = 4)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_dsum)

# chop down reference index if needed
qnbrs4 <- nnd_knn_query(reference = ui6, reference_idx = ui6_nnd$idx, query = ui4, init = rnbrs5, k = 3)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 3, expected_dist = ui10_eucd, tol = 1e-6)

# use k from reference indices
qnbrs6 <- nnd_knn_query(reference = ui4, reference_idx = ui4_nnd$idx, query = ui6)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_dsum, tol = 1e-6)

# high memory
set.seed(1337)
qnbrs4 <- nnd_knn_query(reference = ui6, reference_idx = ui6_nnd$idx, query = ui4, k = 4, low_memory = FALSE)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_dsum)

set.seed(1337)
qnbrs6 <- nnd_knn_query(reference = ui4, reference_idx = ui4_nnd$idx, query = ui6, k = 4, low_memory = FALSE)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_dsum, tol = 1e-6)

# multi-threading
set.seed(1337)
qnbrs6 <- nnd_knn_query(reference = ui4, reference_idx = ui4_nnd$idx, query = ui6, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_dsum, tol = 1e-6)

qnbrs4 <- nnd_knn_query(reference = ui6, reference_idx = ui6_nnd$idx, query = ui4, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_dsum)

# multi-threading himem
set.seed(1337)
qnbrs6 <- nnd_knn_query(reference = ui4, reference_idx = ui4_nnd$idx, query = ui6, k = 4, n_threads = 1, low_memory = FALSE)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_dsum, tol = 1e-6)

qnbrs4 <- nnd_knn_query(reference = ui6, reference_idx = ui6_nnd$idx, query = ui4, k = 4, n_threads = 1, low_memory = FALSE)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_dsum)

# block size
set.seed(1337)
qnbrs6 <- nnd_knn_query(reference = ui4, reference_idx = ui4_nnd$idx, query = ui6, k = 4, n_threads = 1, block_size = 3)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_dsum, tol = 1e-6)

# errors
expect_error(nnd_knn_query(reference = ui4, reference_idx = ui4_nnd$idx, query = ui6, k = 5), "items in the reference data")
expect_error(nnd_knn_query(reference = ui6, reference_idx = ui6_nnd$idx, query = ui4, init = rnbrs5, k = 6), "Not enough initial")
expect_error(nnd_knn_query(reference = ui6, reference_idx = ui6_nnd$idx, query = ui4, init = rnbrs5, k = 5), "Not enough reference")
expect_error(nnd_knn_query(reference = ui4, reference_idx = ui4_nnd$idx, query = ui6, k = 4, metric = "not-a-real metric"), "metric")
