library(rnndescent)
context("Random neighbors")

set.seed(1337)
rnbrs <- random_knn(ui10, k = 4, n_threads = 0)
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

set.seed(1337)
rnbrs <- random_knn(ui10, k = 4, n_threads = 1)
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

# turn off alt metric
set.seed(1337)
rnbrs <- random_knn(ui10, k = 4, n_threads = 1, use_alt_metric = FALSE)
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

# turn off ordering
rnbrs <- random_knn(ui10, k = 4, order_by_distance = FALSE)
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6, check_idx_order = FALSE)
rnbrs <- random_knn(ui10, k = 4, order_by_distance = FALSE, n_threads = 1)
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6, check_idx_order = FALSE)

# large sample code path
res <- random_knn(matrix(rnorm(6000), nrow = 3000), k = 3)
check_nbrs_order(res)

# errors
expect_error(random_knn(ui10, k = 11), "k must be")
expect_error(random_knn(ui10, k = 4, metric = "not-a-real metric"), "metric")

# Queries -----------------------------------------------------------------

context("Random neighbor queries")

set.seed(1337)
qnbrs4 <- random_knn_query(reference = ui6, query = ui4, k = 4)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

set.seed(1337)
qnbrs6 <- random_knn_query(reference = ui4, query = ui6, k = 4)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

# turn off alt_metric
set.seed(1337)
qnbrs6 <- random_knn_query(reference = ui4, query = ui6, k = 4, use_alt_metric = FALSE)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

# no reordering
set.seed(1337)
qnbrs6 <- random_knn_query(reference = ui4, query = ui6, k = 4, order_by_distance = FALSE)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6, check_order = FALSE)

# threads
set.seed(1337)
qnbrs4 <- random_knn_query(reference = ui6, query = ui4, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

set.seed(1337)
qnbrs6 <- random_knn_query(reference = ui4, query = ui6, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

# no re-ordering
set.seed(1337)
qnbrs6 <- random_knn_query(reference = ui4, query = ui6, k = 4, n_threads = 1, order_by_distance = FALSE)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6, check_order = FALSE)

# Errors
expect_error(random_knn_query(reference = ui4, query = ui6, k = 7), "must be <=")
expect_error(random_knn_query(reference = ui4, query = ui6, k = 4, metric = "not-a-real metric"), "metric")

# Other metrics

set.seed(1337)
qnbrs4 <- random_knn_query(reference = ui6, query = ui4, k = 4, metric = "cosine")
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_cosd, tol = 1e-6)

set.seed(1337)
qnbrs6 <- random_knn_query(reference = ui4, query = ui6, k = 4, metric = "cosine")
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_cosd, tol = 1e-6)

set.seed(1337)
qnbrs4 <- random_knn_query(reference = ui6, query = ui4, k = 4, metric = "manhattan")
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_mand, tol = 1e-6)

set.seed(1337)
qnbrs6 <- random_knn_query(reference = ui4, query = ui6, k = 4, metric = "manhattan")
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_mand, tol = 1e-6)

set.seed(1337)
qnbrs4 <- random_knn_query(reference = bit6, query = bit4, k = 4, metric = "hamming")
check_query_nbrs(nn = qnbrs4, query = bit4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = bit10_hamd, tol = 1e-6)

set.seed(1337)
qnbrs6 <- random_knn_query(reference = bit4, query = bit6, k = 4, metric = "hamming")
check_query_nbrs(nn = qnbrs6, query = bit6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = bit10_hamd, tol = 1e-6)


set.seed(1337)
qnbrs4 <- random_knn_query(reference = bit6, query = bit4, k = 4, metric = "bhamming")
check_query_nbrs(nn = qnbrs4, query = bit4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = bit10_hamd, tol = 1e-6)

set.seed(1337)
qnbrs6 <- random_knn_query(reference = bit4, query = bit6, k = 4, metric = "bhamming")
check_query_nbrs(nn = qnbrs6, query = bit6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = bit10_hamd, tol = 1e-6)

test_that("column orientation", {
  set.seed(1337)
  rnbrs <- random_knn(t(ui10), k = 4, n_threads = 0, obs = "C")
  check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

  set.seed(1337)
  qnbrs6 <- random_knn_query(reference = t(bit4), query = t(bit6), k = 4, metric = "bhamming", obs = "C")
  check_query_nbrs(nn = qnbrs6, query = bit6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = bit10_hamd, tol = 1e-6)
})

test_that("one neighbor code path is ok", {
  set.seed(1337)
  rnbrs1 <- random_knn(ui10, k = 1)

  expect_equal(ncol(rnbrs1$idx), 1)
  expect_equal(nrow(rnbrs1$idx), 10)
  expect_equal(ncol(rnbrs1$dist), 1)
  expect_equal(nrow(rnbrs1$dist), 10)

  expect_in(rnbrs1$idx, 1:10)
})
