library(rnndescent)
context("Brute force construction")

rnbrs <- brute_force_knn(ui10, k = 4, n_threads = 0)
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

rnbrs <- brute_force_knn(ui10, k = 4, n_threads = 1)
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

# Error
expect_error(brute_force_knn(ui10, k = 11), "k must be")
expect_error(brute_force_knn(ui10, k = 4, metric = "not-a-real metric"), "metric")

# Queries -----------------------------------------------------------------

context("Brute force queries")

qnbrs4 <- brute_force_knn_query(reference = ui6, query = ui4, k = 4)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

qnbrs6 <- brute_force_knn_query(reference = ui4, query = ui6, k = 4)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

# Errors

expect_error(brute_force_knn_query(reference = ui4, query = ui6, k = 7), "items in the reference data")

# threads

qnbrs4 <- brute_force_knn_query(reference = ui6, query = ui4, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

qnbrs6 <- brute_force_knn_query(reference = ui4, query = ui6, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

# other metrics

ui6_nnd <- brute_force_knn(ui6, k = 4, metric = "cosine")
qnbrs4 <- brute_force_knn_query(reference = ui6, query = ui4, k = 4, metric = "cosine")
check_query_nbrs_idx(qnbrs4$idx, nref = nrow(ui6))
expect_equal(sum(qnbrs4$dist), ui4q_cdsum, tol = 1e-5)

ui4_nnd <- brute_force_knn(ui4, k = 4, metric = "cosine")
qnbrs6 <- brute_force_knn_query(reference = ui4, query = ui6, k = 4, metric = "cosine")
check_query_nbrs_idx(qnbrs6$idx, nref = nrow(ui4))
expect_equal(sum(qnbrs6$dist), ui6q_cdsum, tol = 1e-5)

ui6_nnd <- brute_force_knn(ui6, k = 4, metric = "manhattan")
qnbrs4 <- brute_force_knn_query(reference = ui6, query = ui4, k = 4, metric = "manhattan")
check_query_nbrs_idx(qnbrs4$idx, nref = nrow(ui6))
expect_equal(sum(qnbrs4$dist), ui4q_mdsum, tol = 1e-5)

ui4_nnd <- brute_force_knn(ui4, k = 4, metric = "manhattan")
qnbrs6 <- brute_force_knn_query(reference = ui4, query = ui6, k = 4, metric = "manhattan")
check_query_nbrs_idx(qnbrs6$idx, nref = nrow(ui4))
expect_equal(sum(qnbrs6$dist), ui6q_mdsum, tol = 1e-5)

ui6_nnd <- brute_force_knn(bit6, k = 4, metric = "hamming")
qnbrs4 <- brute_force_knn_query(reference = bit6, query = bit4, k = 4, metric = "hamming")
check_query_nbrs_idx(qnbrs4$idx, nref = nrow(bit6))
expect_equal(sum(qnbrs4$dist), bit4q_hdsum)

ui4_nnd <- brute_force_knn(bit4, k = 4, metric = "hamming")
qnbrs6 <- brute_force_knn_query(reference = bit4, query = bit6, k = 4, metric = "hamming")
check_query_nbrs_idx(qnbrs6$idx, nref = nrow(bit4))
expect_equal(sum(qnbrs6$dist), bit6q_hdsum)
