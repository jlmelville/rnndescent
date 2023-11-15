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
qnbrs4 <- random_knn_query(reference = lbit6, query = lbit4, k = 4, metric = "hamming")
check_query_nbrs(nn = qnbrs4, query = bit4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = bit10_hamd, tol = 1e-6)

set.seed(1337)
qnbrs6 <- random_knn_query(reference = lbit4, query = lbit6, k = 4, metric = "hamming")
check_query_nbrs(nn = qnbrs6, query = bit6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = bit10_hamd, tol = 1e-6)

test_that("column orientation", {
  set.seed(1337)
  rnbrs <- random_knn(t(ui10), k = 4, n_threads = 0, obs = "C")
  check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

  set.seed(1337)
  qnbrs6 <- random_knn_query(reference = t(lbit4), query = t(lbit6), k = 4, metric = "hamming", obs = "C")
  check_query_nbrs(nn = qnbrs6, query = lbit6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = bit10_hamd, tol = 1e-6)
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

test_that("sparse euclidean", {
  set.seed(1337)
  dzrnbrs <- random_knn(ui10z, k = 4, n_threads = 0, use_alt_metric = FALSE, metric = "euclidean")
  set.seed(1337)
  sprnbrs <- random_knn(ui10sp, k = 4, n_threads = 0, use_alt_metric = FALSE, metric = "euclidean")
  expect_equal(dzrnbrs, sprnbrs)
  set.seed(1337)
  sprnbrs <- random_knn(ui10sp, k = 4, n_threads = 0, use_alt_metric = TRUE, metric = "euclidean")
  expect_equal(dzrnbrs, sprnbrs, tol = 1e-7)
})

test_that("sparse cosine", {
  set.seed(1337)
  dzrnbrs <- random_knn(ui10z, k = 4, n_threads = 0, use_alt_metric = FALSE, metric = "cosine")
  set.seed(1337)
  sprnbrs <- random_knn(ui10sp, k = 4, n_threads = 0, use_alt_metric = FALSE, metric = "cosine")
  expect_equal(dzrnbrs, sprnbrs, tol = 1e-6)
  set.seed(1337)
  sprnbrs <- random_knn(ui10sp, k = 4, n_threads = 0, use_alt_metric = TRUE, metric = "cosine")
  expect_equal(dzrnbrs, sprnbrs, tol = 1e-6)
})

test_that("sparse query", {
  set.seed(1337)
  sp6_4 <- random_knn_query(ui10sp6, ui10sp4, k = 4)
  set.seed(1337)
  dz6_4 <- random_knn_query(ui10z6, ui10z4, k = 4)
  expect_equal(sp6_4, dz6_4)

  set.seed(1337)
  sp4_6 <- random_knn_query(ui10sp4, ui10sp6, k = 4)
  set.seed(1337)
  dz4_6 <- random_knn_query(ui10z4, ui10z6, k = 4)
  expect_equal(sp4_6, dz4_6)
})
