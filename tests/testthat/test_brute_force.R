library(rnndescent)
context("Brute force construction")

rnbrs <- brute_force_knn(ui10, k = 4, n_threads = 0)
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

rnbrs <- brute_force_knn(ui10, k = 4, n_threads = 1)
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

# turn off alt metric
rnbrs <- brute_force_knn(ui10, k = 4, n_threads = 1, use_alt_metric = FALSE)
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

# Error
expect_error(brute_force_knn(ui10, k = 11), "k must be")
expect_error(brute_force_knn(ui10, k = 4, metric = "not-a-real metric"), "metric")

# Queries -----------------------------------------------------------------

context("Brute force queries")

qnbrs4 <- brute_force_knn_query(reference = ui6, query = ui4, k = 4)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_edsum, tol = 1e-5)

qnbrs6 <- brute_force_knn_query(reference = ui4, query = ui6, k = 4)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_edsum, tol = 1e-5)

# turn off alt metric
qnbrs6 <- brute_force_knn_query(reference = ui4, query = ui6, k = 4, use_alt_metric = FALSE)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_edsum, tol = 1e-5)

# Errors

expect_error(brute_force_knn_query(reference = ui4, query = ui6, k = 7), "k must be")
expect_error(brute_force_knn_query(reference = ui4, query = ui6, k = 4, metric = "not-a-real metric"), "metric")

# threads

qnbrs4 <- brute_force_knn_query(reference = ui6, query = ui4, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs4, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs4$dist), ui4q_edsum, tol = 1e-5)

qnbrs6 <- brute_force_knn_query(reference = ui4, query = ui6, k = 4, n_threads = 1)
check_query_nbrs(nn = qnbrs6, query = ui6, ref_range = 7:10, query_range = 1:6, k = 4, expected_dist = ui10_eucd, tol = 1e-6)
expect_equal(sum(qnbrs6$dist), ui6q_edsum, tol = 1e-5)

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

ui6_nnd <- brute_force_knn(bit6, k = 4, metric = "bhamming")
qnbrs4 <- brute_force_knn_query(reference = bit6, query = bit4, k = 4, metric = "bhamming")
check_query_nbrs_idx(qnbrs4$idx, nref = nrow(bit6))
expect_equal(sum(qnbrs4$dist), bit4q_hdsum)

ui4_nnd <- brute_force_knn(bit4, k = 4, metric = "bhamming")
qnbrs6 <- brute_force_knn_query(reference = bit4, query = bit6, k = 4, metric = "bhamming")
check_query_nbrs_idx(qnbrs6$idx, nref = nrow(bit4))
expect_equal(sum(qnbrs6$dist), bit6q_hdsum)

ui6_nnd <- brute_force_knn(int6, k = 6, metric = "hamming")
check_nbrs(ui6_nnd, int6hd, tol = 1e-6, check_idx_order = FALSE, check_dist_order = TRUE)

h46d <- matrix(c(
  1, 2, 2, 3,
  3, 3, 3, 3,
  3, 3, 4, 4,
  2, 2, 3, 3),
  byrow = TRUE, nrow = 4)
qnbrs4 <- brute_force_knn_query(int4, int6, k = 4, metric = "hamming")
check_query_nbrs_idx(qnbrs4$idx, nref = nrow(int6))
expect_equal(qnbrs4$dist, h46d)

h64d <- matrix(c(
  2, 3, 3, 4,
  1, 2, 3, 4,
  3, 3, 3, 4,
  2, 2, 3, 4,
  3, 3, 4, 4,
  3, 4, 4, 4),
  byrow = TRUE, nrow = 6)
qnbrs6 <- brute_force_knn_query(int6, int4, k = 4, metric = "hamming")
check_query_nbrs_idx(qnbrs6$idx, nref = nrow(int6))
expect_equal(qnbrs6$dist, h64d)

# orientation

rnbrs <- brute_force_knn(t(ui10), k = 4, n_threads = 2, obs = "C")
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

qnbrs6 <- brute_force_knn_query(t(int6), t(int4), k = 4, metric = "hamming", n_threads = 2, obs = "C")
check_query_nbrs_idx(qnbrs6$idx, nref = nrow(int6))
expect_equal(qnbrs6$dist, h64d)

# sparse
rnbrs <- brute_force_knn(ui10sp_full, k = 4, n_threads = 0)
check_nbrs(rnbrs, ui10_eucd, tol = 1e-6)

nbrs_zdense <- brute_force_knn(ui10z, k = 4, n_threads = 0)
nbrs_zsparse <- brute_force_knn(ui10sp, k = 4, n_threads = 0)
expect_equal(nbrs_zsparse, nbrs_zdense)

expect_equal(
  brute_force_knn(
    ui10sp,
    k = 4,
    n_threads = 0,
    metric = "euclidean",
    use_alt_metric = FALSE
  ),
  brute_force_knn(
    ui10z,
    k = 4,
    n_threads = 0,
    metric = "euclidean",
    use_alt_metric = FALSE
  )
)

expect_equal(
  brute_force_knn(
    ui10sp,
    k = 4,
    n_threads = 0,
    metric = "manhattan"
  ),
  brute_force_knn(
    ui10z,
    k = 4,
    n_threads = 0,
    metric = "manhattan"
  )
)


expect_equal(
  brute_force_knn(
    bitdata,
    k = 4,
    n_threads = 0,
    metric = "hamming"
  ),
  brute_force_knn(
    Matrix::drop0(bitdata),
    k = 4,
    n_threads = 0,
    metric = "hamming"
  )
)

expect_equal(
  brute_force_knn(
    bitdata,
    k = 4,
    n_threads = 0,
    metric = "cosine"
  ),
  brute_force_knn(
    Matrix::drop0(bitdata),
    k = 4,
    n_threads = 0,
    metric = "cosine",
    use_alt_metric = FALSE
  ),
  tol = 1e-6
)

expect_equal(
  brute_force_knn(
    bitdata,
    k = 4,
    n_threads = 0,
    metric = "cosine"
  ),
  brute_force_knn(
    Matrix::drop0(bitdata),
    k = 4,
    n_threads = 0,
    metric = "cosine",
    use_alt_metric = TRUE
  ),
  tol = 1e-6
)

expect_equal(
  brute_force_knn(
    ui10sp,
    k = 4,
    n_threads = 0,
    metric = "correlation"
  ),
  brute_force_knn(
    ui10z,
    k = 4,
    n_threads = 0,
    metric = "correlation"
  ),
  tol = 1e-5
)

# sparse queries
expect_equal(brute_force_knn_query(ui10sp, ui10sp, k = 4), brute_force_knn(ui10sp, k = 4))
expect_equal(brute_force_knn_query(ui10sp6, ui10sp4, k = 4), brute_force_knn_query(ui10z6, ui10z4, k = 4))
expect_equal(brute_force_knn_query(ui10sp4, ui10sp6, k = 4), brute_force_knn_query(ui10z4, ui10z6, k = 4))
