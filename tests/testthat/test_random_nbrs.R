library(rnndescent)
context("Random neighbors")

set.seed(1337)
rnbrs <- random_knn(ui10, k = 4, n_threads = 0)
check_nbrs(rnbrs, ui10_eucd, tolerance = 1e-6)

set.seed(1337)
rnbrs <- random_knn(ui10, k = 4, n_threads = 1)
check_nbrs(rnbrs, ui10_eucd, tolerance = 1e-6)

# turn off alt metric
set.seed(1337)
rnbrs <- random_knn(ui10, k = 4, n_threads = 1, use_alt_metric = FALSE)
check_nbrs(rnbrs, ui10_eucd, tolerance = 1e-6)

# turn off ordering
rnbrs <- random_knn(ui10, k = 4, order_by_distance = FALSE)
check_nbrs(rnbrs, ui10_eucd, tolerance = 1e-6, check_idx_order = FALSE)
rnbrs <- random_knn(ui10, k = 4, order_by_distance = FALSE, n_threads = 1)
check_nbrs(rnbrs, ui10_eucd, tolerance = 1e-6, check_idx_order = FALSE)

test_that("ordering random neighbors preserves sampled pairs", {
  coordinates <- matrix(c(0, 1, 4, 10, 20, 35, 56, 84), ncol = 1)
  logical_data <- outer(0:7, c(1L, 2L, 4L), function(x, y) {
    bitwAnd(x, y) > 0
  })
  euclidean_dist <- as.matrix(stats::dist(coordinates))
  hamming_dist <- as.matrix(stats::dist(logical_data, method = "manhattan")) /
    ncol(logical_data)
  cases <- list(
    dense_serial = list(data = coordinates, n_threads = 0),
    dense_parallel = list(data = coordinates, n_threads = 2),
    dense_columns = list(
      data = t(coordinates),
      n_threads = 0,
      obs = "C",
      use_alt_metric = FALSE
    ),
    sparse_columns = list(
      data = Matrix::Matrix(t(coordinates), sparse = TRUE),
      n_threads = 2,
      obs = "C"
    ),
    logical_parallel = list(
      data = logical_data,
      n_threads = 2,
      metric = "hamming"
    )
  )

  for (case_name in names(cases)) {
    args <- c(cases[[case_name]], list(k = 3))
    set.seed(1337)
    unordered <- do.call(random_knn, c(args, list(order_by_distance = FALSE)))
    set.seed(1337)
    ordered <- do.call(random_knn, c(args, list(order_by_distance = TRUE)))

    expected_dist <- if (case_name == "logical_parallel") {
      hamming_dist
    } else {
      euclidean_dist
    }
    for (neighbors in list(unordered, ordered)) {
      expect_equal(dim(neighbors$idx), c(8L, 3L), info = case_name)
      expect_equal(dim(neighbors$dist), c(8L, 3L), info = case_name)
      expect_true(all(neighbors$idx %in% 1:8), info = case_name)
      expect_equal(
        as.vector(neighbors$dist),
        expected_dist[cbind(
          as.vector(row(neighbors$idx)),
          as.vector(neighbors$idx)
        )],
        tolerance = 1e-6,
        info = case_name
      )
    }
    unordered_pairs <- ordered_pairs <- vector("list", 8)
    for (i in seq_len(8)) {
      unordered_order <- order(unordered$idx[i, ], unordered$dist[i, ])
      ordered_order <- order(ordered$idx[i, ], ordered$dist[i, ])
      unordered_pairs[[i]] <- cbind(
        unordered$idx[i, unordered_order],
        unordered$dist[i, unordered_order]
      )
      ordered_pairs[[i]] <- cbind(
        ordered$idx[i, ordered_order],
        ordered$dist[i, ordered_order]
      )
    }
    expect_equal(ordered_pairs, unordered_pairs, info = case_name)
    expect_false(any(apply(ordered$dist, 1, is.unsorted)), info = case_name)
  }
})

# large sample code path
res <- random_knn(matrix(rnorm(6000), nrow = 3000), k = 3)
check_nbrs_order(res)

# errors
expect_error(random_knn(ui10, k = 11), "k must be")
expect_error(random_knn(ui10, k = 0), "k must be")
expect_error(random_knn(ui10, k = 1.5), "k must be")
expect_error(random_knn(ui10, k = 4, metric = "not-a-real metric"), "metric")

# Queries -----------------------------------------------------------------

context("Random neighbor queries")

set.seed(1337)
qnbrs4 <- random_knn_query(reference = ui6, query = ui4, k = 4)
check_query_nbrs(
  nn = qnbrs4,
  query = ui4,
  ref_range = 1:6,
  query_range = 7:10,
  k = 4,
  expected_dist = ui10_eucd,
  tolerance = 1e-6
)

set.seed(1337)
qnbrs6 <- random_knn_query(reference = ui4, query = ui6, k = 4)
check_query_nbrs(
  nn = qnbrs6,
  query = ui6,
  ref_range = 7:10,
  query_range = 1:6,
  k = 4,
  expected_dist = ui10_eucd,
  tolerance = 1e-6
)

# turn off alt_metric
set.seed(1337)
qnbrs6 <- random_knn_query(
  reference = ui4,
  query = ui6,
  k = 4,
  use_alt_metric = FALSE
)
check_query_nbrs(
  nn = qnbrs6,
  query = ui6,
  ref_range = 7:10,
  query_range = 1:6,
  k = 4,
  expected_dist = ui10_eucd,
  tolerance = 1e-6
)

# no reordering
set.seed(1337)
qnbrs6 <- random_knn_query(
  reference = ui4,
  query = ui6,
  k = 4,
  order_by_distance = FALSE
)
check_query_nbrs(
  nn = qnbrs6,
  query = ui6,
  ref_range = 7:10,
  query_range = 1:6,
  k = 4,
  expected_dist = ui10_eucd,
  tolerance = 1e-6,
  check_order = FALSE
)

# threads
set.seed(1337)
qnbrs4 <- random_knn_query(reference = ui6, query = ui4, k = 4, n_threads = 1)
check_query_nbrs(
  nn = qnbrs4,
  query = ui4,
  ref_range = 1:6,
  query_range = 7:10,
  k = 4,
  expected_dist = ui10_eucd,
  tolerance = 1e-6
)

set.seed(1337)
qnbrs6 <- random_knn_query(reference = ui4, query = ui6, k = 4, n_threads = 1)
check_query_nbrs(
  nn = qnbrs6,
  query = ui6,
  ref_range = 7:10,
  query_range = 1:6,
  k = 4,
  expected_dist = ui10_eucd,
  tolerance = 1e-6
)

# no re-ordering
set.seed(1337)
qnbrs6 <- random_knn_query(
  reference = ui4,
  query = ui6,
  k = 4,
  n_threads = 1,
  order_by_distance = FALSE
)
check_query_nbrs(
  nn = qnbrs6,
  query = ui6,
  ref_range = 7:10,
  query_range = 1:6,
  k = 4,
  expected_dist = ui10_eucd,
  tolerance = 1e-6,
  check_order = FALSE
)

# Errors
expect_error(random_knn_query(reference = ui4, query = ui6, k = 7), "k must be")
expect_error(random_knn_query(reference = ui4, query = ui6, k = 0), "k must be")
expect_error(
  random_knn_query(
    reference = ui4,
    query = ui6,
    k = 4,
    metric = "not-a-real metric"
  ),
  "metric"
)
expect_error(
  random_knn_query(
    reference = matrix(FALSE, nrow = 2, ncol = 2),
    query = matrix(FALSE, nrow = 2, ncol = 3),
    k = 1,
    metric = "hamming"
  ),
  "same number of features"
)
expect_error(
  random_knn_query(
    reference = matrix(FALSE, nrow = 2, ncol = 2),
    query = matrix(FALSE, nrow = 3, ncol = 2),
    k = 1,
    metric = "hamming",
    obs = "C"
  ),
  "same number of features"
)

test_that("random neighbor APIs reject invalid n_threads values", {
  expect_error(random_knn(ui10, k = 4, n_threads = -1), "n_threads must be")
  expect_error(
    random_knn_query(reference = ui4, query = ui6, k = 4, n_threads = 1.5),
    "n_threads must be"
  )
})

# Other metrics

set.seed(1337)
qnbrs4 <- random_knn_query(
  reference = ui6,
  query = ui4,
  k = 4,
  metric = "cosine"
)
check_query_nbrs(
  nn = qnbrs4,
  query = ui4,
  ref_range = 1:6,
  query_range = 7:10,
  k = 4,
  expected_dist = ui10_cosd,
  tolerance = 1e-6
)

set.seed(1337)
qnbrs6 <- random_knn_query(
  reference = ui4,
  query = ui6,
  k = 4,
  metric = "cosine"
)
check_query_nbrs(
  nn = qnbrs6,
  query = ui6,
  ref_range = 7:10,
  query_range = 1:6,
  k = 4,
  expected_dist = ui10_cosd,
  tolerance = 1e-6
)

set.seed(1337)
qnbrs4 <- random_knn_query(
  reference = ui6,
  query = ui4,
  k = 4,
  metric = "manhattan"
)
check_query_nbrs(
  nn = qnbrs4,
  query = ui4,
  ref_range = 1:6,
  query_range = 7:10,
  k = 4,
  expected_dist = ui10_mand,
  tolerance = 1e-6
)

set.seed(1337)
qnbrs6 <- random_knn_query(
  reference = ui4,
  query = ui6,
  k = 4,
  metric = "manhattan"
)
check_query_nbrs(
  nn = qnbrs6,
  query = ui6,
  ref_range = 7:10,
  query_range = 1:6,
  k = 4,
  expected_dist = ui10_mand,
  tolerance = 1e-6
)

set.seed(1337)
qnbrs4 <- random_knn_query(
  reference = bit6,
  query = bit4,
  k = 4,
  metric = "hamming"
)
check_query_nbrs(
  nn = qnbrs4,
  query = bit4,
  ref_range = 1:6,
  query_range = 7:10,
  k = 4,
  expected_dist = bit10_hamd,
  tolerance = 1e-6
)

set.seed(1337)
qnbrs6 <- random_knn_query(
  reference = bit4,
  query = bit6,
  k = 4,
  metric = "hamming"
)
check_query_nbrs(
  nn = qnbrs6,
  query = bit6,
  ref_range = 7:10,
  query_range = 1:6,
  k = 4,
  expected_dist = bit10_hamd,
  tolerance = 1e-6
)


set.seed(1337)
qnbrs4 <- random_knn_query(
  reference = lbit6,
  query = lbit4,
  k = 4,
  metric = "hamming"
)
check_query_nbrs(
  nn = qnbrs4,
  query = bit4,
  ref_range = 1:6,
  query_range = 7:10,
  k = 4,
  expected_dist = bit10_hamd,
  tolerance = 1e-6
)

set.seed(1337)
qnbrs6 <- random_knn_query(
  reference = lbit4,
  query = lbit6,
  k = 4,
  metric = "hamming"
)
check_query_nbrs(
  nn = qnbrs6,
  query = bit6,
  ref_range = 7:10,
  query_range = 1:6,
  k = 4,
  expected_dist = bit10_hamd,
  tolerance = 1e-6
)

test_that("column orientation", {
  set.seed(1337)
  rnbrs <- random_knn(t(ui10), k = 4, n_threads = 0, obs = "C")
  check_nbrs(rnbrs, ui10_eucd, tolerance = 1e-6)

  set.seed(1337)
  qnbrs6 <- random_knn_query(
    reference = t(lbit4),
    query = t(lbit6),
    k = 4,
    metric = "hamming",
    obs = "C"
  )
  check_query_nbrs(
    nn = qnbrs6,
    query = lbit6,
    ref_range = 7:10,
    query_range = 1:6,
    k = 4,
    expected_dist = bit10_hamd,
    tolerance = 1e-6
  )
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
  dzrnbrs <- random_knn(
    ui10z,
    k = 4,
    n_threads = 0,
    use_alt_metric = FALSE,
    metric = "euclidean"
  )
  set.seed(1337)
  sprnbrs <- random_knn(
    ui10sp,
    k = 4,
    n_threads = 0,
    use_alt_metric = FALSE,
    metric = "euclidean"
  )
  expect_equal(dzrnbrs, sprnbrs)
  set.seed(1337)
  sprnbrs <- random_knn(
    ui10sp,
    k = 4,
    n_threads = 0,
    use_alt_metric = TRUE,
    metric = "euclidean"
  )
  expect_equal(dzrnbrs, sprnbrs, tolerance = 1e-7)
})

test_that("sparse cosine", {
  set.seed(1337)
  dzrnbrs <- random_knn(
    ui10z,
    k = 4,
    n_threads = 0,
    use_alt_metric = FALSE,
    metric = "cosine"
  )
  set.seed(1337)
  sprnbrs <- random_knn(
    ui10sp,
    k = 4,
    n_threads = 0,
    use_alt_metric = FALSE,
    metric = "cosine"
  )
  # Float cosine roundoff near zero needs an absolute distance bound.
  expect_identical(dzrnbrs$idx, sprnbrs$idx)
  expect_identical(dim(dzrnbrs$dist), dim(sprnbrs$dist))
  expect_true(all(is.finite(c(dzrnbrs$dist, sprnbrs$dist))))
  expect_true(all(abs(dzrnbrs$dist - sprnbrs$dist) <= 1e-6))
  set.seed(1337)
  sprnbrs <- random_knn(
    ui10sp,
    k = 4,
    n_threads = 0,
    use_alt_metric = TRUE,
    metric = "cosine"
  )
  expect_identical(dzrnbrs$idx, sprnbrs$idx)
  expect_identical(dim(dzrnbrs$dist), dim(sprnbrs$dist))
  expect_true(all(is.finite(c(dzrnbrs$dist, sprnbrs$dist))))
  expect_true(all(abs(dzrnbrs$dist - sprnbrs$dist) <= 1e-6))
})

test_that("sparse query", {
  set.seed(1337)
  sp6_4 <- random_knn_query(ui10sp6, ui10sp4, k = 4)
  set.seed(1337)
  dz6_4 <- random_knn_query(ui10z6, ui10z4, k = 4)
  expect_equal(sp6_4, dz6_4, tolerance = 1e-6)

  set.seed(1337)
  sp4_6 <- random_knn_query(ui10sp4, ui10sp6, k = 4)
  set.seed(1337)
  dz4_6 <- random_knn_query(ui10z4, ui10z6, k = 4)
  expect_equal(sp4_6, dz4_6, tolerance = 1e-6)
})

test_that("random neighbor APIs normalize non-dgC sparse inputs", {
  data_r <- methods::as(ui10sp, "RsparseMatrix")
  ref_r <- methods::as(ui10sp4, "RsparseMatrix")
  query_r <- methods::as(ui10sp6, "RsparseMatrix")

  set.seed(1337)
  expected_knn <- random_knn(ui10sp, k = 4, n_threads = 0)
  set.seed(1337)
  observed_knn <- random_knn(data_r, k = 4, n_threads = 0)
  expect_equal(observed_knn, expected_knn, tolerance = 1e-6)

  set.seed(1337)
  expected_query <- random_knn_query(
    reference = ui10sp4,
    query = ui10sp6,
    k = 4
  )
  set.seed(1337)
  observed_query <- random_knn_query(reference = ref_r, query = query_r, k = 4)
  expect_equal(observed_query, expected_query, tolerance = 1e-6)
})
