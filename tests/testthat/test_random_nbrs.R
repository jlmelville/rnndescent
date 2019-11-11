library(rnndescent)
context("Random neighbors")

set.seed(1337)
rnbrs <- random_knn(ui10, k = 4, n_threads = 0)
check_nbrs_idx(rnbrs$idx)
check_nbrs_dist(rnbrs, ui10_eucd, tol = 1e-6)

set.seed(1337)
rnbrs <- random_knn(ui10, k = 4, n_threads = 1)
check_nbrs_idx(rnbrs$idx)
check_nbrs_dist(rnbrs, ui10_eucd, tol = 1e-6)

# errors
expect_error(random_knn(ui10, k = 11), "k must be")
expect_error(random_knn(ui10, k = 4, metric = "not-a-real metric"), "metric")

# Queries -----------------------------------------------------------------

context("Random neighbor queries")

ui6 <- ui10[1:6, ]
ui4 <- ui10[7:10, ]

set.seed(1337)
qnbrs4 <- random_knn_query(reference = ui6, query = ui4, k = 4)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

set.seed(1337)
qnbrs6 <- random_knn_query(reference = ui4, query = ui6, k = 4)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

set.seed(1337)
qnbrs4 <- random_knn_query(reference = ui6, query = ui4, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

set.seed(1337)
qnbrs6 <- random_knn_query(reference = ui4, query = ui6, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

# Errors

expect_error(random_knn_query(reference = ui4, query = ui6, k = 7), "items in the reference data")
