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
expect_equal(sum(uiris_rnn$dist), ui_edsum, tol = 1e-3)

# data frame can be input
set.seed(1337)
uiris_rnn <- nnd_knn(uiris, 15)
expect_equal(sum(uiris_rnn$dist), ui_edsum, tol = 1e-3)

# Create external initialization
set.seed(1337)
iris_nbrs <- random_knn(uirism, 15)

# initialize from existing knn graph
set.seed(1337)
iris_nnd <- nnd_knn(uirism, init = iris_nbrs)
expect_equal(sum(iris_nnd$dist), ui_edsum, tol = 1e-3)

# initialize from rp forest
set.seed(1337)
iris_nnd <- nnd_knn(uirism, k = 15, init = "tree")
expect_equal(sum(iris_nnd$dist), ui_edsum, tol = 1e-3)

# weight by degree
set.seed(1337)
iris_nndw <- nnd_knn(uirism, k = 15, init = "tree", weight_by_degree = TRUE)
expect_equal(sum(iris_nndw$dist), ui_edsum, tol = 1e-3)

# initialize from existing knn graph with missing data
set.seed(1337)
iris_nbrs_missing <- iris_nbrs
iris_nbrs_missing$idx[1, ] <- rep(0, ncol(iris_nbrs_missing$idx))
iris_nbrs_missing$dist[1, ] <-
  rep(NA, ncol(iris_nbrs_missing$dist))
iris_nnd <- nnd_knn(uirism, init = iris_nbrs_missing)
expect_equal(sum(iris_nnd$dist), ui_edsum, tol = 1e-3)

# initialize from existing knn indices
set.seed(1337)
iris_nnd <- nnd_knn(uirism, init = list(idx = iris_nbrs$idx))
expect_equal(sum(iris_nnd$dist), ui_edsum, tol = 1e-3)

# Use larger initialization for smaller k
set.seed(1337)
iris_nnd <- nnd_knn(uirism, init = random_knn(uirism, 20), k = 15)
expect_equal(sum(iris_nnd$dist), ui_edsum, tol = 1e-3)

# high memory mode
set.seed(1337)
iris_nnd <- nnd_knn(uirism, init = iris_nbrs, low_memory = FALSE)
expect_equal(sum(iris_nnd$dist), ui_edsum, tol = 1e-3)

# init default with high memory
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, low_memory = FALSE)
expect_equal(sum(uiris_rnn$dist), ui_edsum, tol = 1e-3)

# max candidates
set.seed(1337)
iris_nnd <- nnd_knn(uirism, init = iris_nbrs, max_candidates = 10)
expect_equal(sum(iris_nnd$dist), ui_edsum, tol = 1e-3)

# turn off alt metric
set.seed(1337)
ui10_rnn <- nnd_knn(ui10, 4, use_alt_metric = FALSE)
expect_equal(sum(ui10_rnn$dist), ui10_edsum, tol = 1e-3)

# augment with random
set.seed(1337)
ui10_rnnk2 <- nnd_knn(ui10, 2, use_alt_metric = FALSE)
ui10_rnn <- nnd_knn(ui10, 4, init = ui10_rnnk2)
expect_equal(sum(ui10_rnn$dist), ui10_edsum, tol = 1e-3)

# errors
expect_error(nnd_knn(ui10), "provide k")
expect_error(nnd_knn(ui10, k = 11), "k must be")
expect_error(nnd_knn(uirism, k = 15, init = iris_nbrs, metric = "not-a-real metric"), "metric")
expect_error(nnd_knn(uirism, init = list(dist = iris_nbrs$dist, idx = iris_nbrs$idx - 2)), "Bad indexes")

# verbosity
msgs <- capture_everything(nnd_knn(ui10, 4, verbose = TRUE))
expect_match(msgs, "\\*\\*\\*")
expect_match(msgs, "Convergence")

msgs <- capture_everything(nnd_knn(ui10, 4, verbose = TRUE, progress = "dist", n_iters = 10))
expect_match(msgs, "1 / 10")
expect_match(msgs, "Convergence")

# Multi-threading ---------------------------------------------------------

# multi-threading
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, n_threads = 1)
expect_equal(sum(uiris_rnn$dist), ui_edsum, tol = 1e-3)

# with caching
set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, n_threads = 1, low_memory = FALSE)
expect_equal(sum(uiris_rnn$dist), ui_edsum, tol = 1e-3)

# initialize from existing knn indices
set.seed(1337)
iris_nnd <- nnd_knn(uirism, init = list(idx = iris_nbrs$idx), n_threads = 1)
expect_equal(sum(iris_nnd$dist), ui_edsum, tol = 1e-3)

# Queries -----------------------------------------------------------------

context("Euclidean queries")

set.seed(1337)
ui6_nnd <- nnd_knn(ui6, k = 4)
ui6_nnd_idx_copy <- copy(ui6_nnd$idx)
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, k = 4)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_edsum, tol = 1e-6)
expect_equal(ui6_nnd$idx, ui6_nnd_idx_copy)

set.seed(1337)
ui4_nnd <- nnd_knn(ui4, k = 4)
qnbrs6 <- graph_knn_query(reference = ui4, reference_graph = ui4_nnd, query = ui6, k = 4)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_edsum, tol = 1e-6)

# turn off alt metric
set.seed(1337)
qnbrs6 <- graph_knn_query(reference = ui4, reference_graph = ui4_nnd, query = ui6, k = 4, use_alt_metric = FALSE)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_edsum, tol = 1e-6)

# initialize separately
rnbrs4 <- random_knn_query(reference = ui6, query = ui4, k = 4)
rnbrs4_idx_copy <- copy(rnbrs4$idx)
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, init = rnbrs4)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_edsum, tol = 1e-6)
expect_equal(rnbrs4$idx, rnbrs4_idx_copy)

# initialize separately and reduce graph
rnbrs5 <- random_knn_query(reference = ui6, query = ui4, k = 5)
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, init = rnbrs5, k = 4)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_edsum, tol = 1e-6)

# chop down reference index if needed
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, init = rnbrs5, k = 3)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 3, expected_dist = ui10_eucd, tol = 1e-6)

# use k from reference indices
qnbrs6 <- graph_knn_query(reference = ui4, reference_graph = ui4_nnd, query = ui6)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_edsum, tol = 1e-6)

# initialize from existing knn indices
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, init = list(idx = rnbrs4$idx))
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_edsum, tol = 1e-6)
expect_equal(rnbrs4$idx, rnbrs4_idx_copy)

# multi-threading
set.seed(1337)
qnbrs6 <- graph_knn_query(reference = ui4, reference_graph = ui4_nnd, query = ui6, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_edsum, tol = 1e-6)

qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_edsum, tol = 1e-6)

# initialize from existing knn indices
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, init = list(idx = rnbrs4$idx), n_threads = 1)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_edsum, tol = 1e-6)
expect_equal(rnbrs4$idx, rnbrs4_idx_copy)

# initialize from existing matrix
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, init = rnbrs4$idx, n_threads = 1)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_edsum, tol = 1e-6)
expect_equal(rnbrs4$idx, rnbrs4_idx_copy)

# augment with random
set.seed(1337)
rnbrs2 <- random_knn_query(reference = ui6, query = ui4, k = 2)
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, init = rnbrs2, k = 4)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_edsum, tol = 1e-6)

# max_search_fraction
qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, k = 4, max_search_fraction = 0.0, init = rnbrs4)
expect_equal(qnbrs4, rnbrs4)

# errors
expect_error(graph_knn_query(
  reference = ui4, reference_graph = ui4_nnd,
  query = ui6, k = 5
), "must be <=")
expect_error(graph_knn_query(
  reference = ui4, reference_graph = ui4_nnd,
  query = ui6, k = 4, metric = "not-a-real metric"
), "metric")
expect_error(graph_knn_query(
  reference = ui6, reference_graph = ui6_nnd,
  query = ui4, init = rnbrs4, metric = "not-a-real metric"
), "metric")

test_that("column oriented", {
  set.seed(1337)
  uiris_rnn <- nnd_knn(t(uirism), 15, obs = "C", n_threads = 2)
  expect_equal(sum(uiris_rnn$dist), ui_edsum, tol = 1e-3)

  set.seed(1337)
  ui4_nnd <- nnd_knn(ui4, k = 4)
  qnbrs6 <- graph_knn_query(reference = t(ui4), reference_graph = ui4_nnd, query = t(ui6), k = 4, obs = "C", n_threads = 2)
  check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
  expect_equal(sum(qnbrs6$dist), ui6q_edsum, tol = 1e-6)
})

test_that("sparse", {
  set.seed(1337)
  dznbrs <- nnd_knn(ui10z, k = 4, n_threads = 0, metric = "euclidean")
  set.seed(1337)
  spnbrs <- nnd_knn(ui10sp, k = 4, n_threads = 0, metric = "euclidean", use_alt_metric = TRUE)
  expect_equal(dznbrs, spnbrs)
  set.seed(1337)
  spnbrs <- nnd_knn(ui10sp, k = 4, n_threads = 0, metric = "euclidean", use_alt_metric = FALSE)
  expect_equal(dznbrs, spnbrs, tol = 1e-7)

  # sparse uncorrection
  set.seed(1337)
  spnbrs <- nnd_knn(ui10sp, k = 4, n_threads = 0, metric = "euclidean", init = random_knn(ui10sp, k = 4))
  expect_equal(dznbrs, spnbrs, tol = 1e-7)

  g6 <- brute_force_knn(ui10z6, k = 4)
  set.seed(1337)
  dq4 <- graph_knn_query(reference = ui10z6, query = ui10z4, reference_graph = g6, k = 4)
  set.seed(1337)
  sq4 <- graph_knn_query(reference = ui10sp6, query = ui10sp4, reference_graph = g6, k = 4)
  expect_equal(sq4, dq4)
})


test_that("full workflow", {
  iris_ref <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  iris_query <- iris[iris$Species == "versicolor", ]

  set.seed(1337)
  iris_ref_graph <- nnd_knn(iris_ref, k = 4, init = "tree", ret_forest = TRUE)
  # keep the 5 best trees in the forest
  forest <- rpf_filter(iris_ref_graph, n_trees = 5)
  # expand the knn into a search graph
  iris_ref_search_graph <- prepare_search_graph(iris_ref, iris_ref_graph)
  # run the query with the improved graph and initialization
  iris_query_nn <- graph_knn_query(iris_query, iris_ref, iris_ref_search_graph,
    init = forest, k = 4, epsilon = 1.1
  )

  iris_qbf <- brute_force_knn_query(iris_query, iris_ref, k = 4)

  # there can be ties, so just check distances
  expect_equal(iris_query_nn$dist, iris_qbf$dist)

  # initializing from a forest means that the metric param is ignored
  iris_query_nnc <- graph_knn_query(iris_query, iris_ref, iris_ref_search_graph,
    init = forest, k = 4, epsilon = 1.1,
    metric = "cosine"
  )
  expect_equal(iris_query_nn$dist, iris_qbf$dist)
})
