library(rnndescent)

set.seed(1337)
ui10rnn1 <- random_knn(ui10, k = 4, order_by_distance = FALSE)
ui10rnn2 <- random_knn(ui10, k = 4, order_by_distance = FALSE)
ui10rnn3 <- random_knn(ui10, k = 4, order_by_distance = FALSE)

# serial
output <- capture_everything({
  ui10mnn <- merge_knn(list(ui10rnn1, ui10rnn2), verbose = TRUE)
})
test_that("merging two graphs improves neighbors and reports progress", {
  expect_match(output, "Merging")
  expect_true(sum(ui10mnn$dist) < sum(ui10rnn1$dist))
  expect_true(sum(ui10mnn$dist) < sum(ui10rnn2$dist))
  check_nbrs(ui10mnn, ui10_eucd, tolerance = 1e-6)
})

test_that("merging graphs retains the first neighbor count", {
  # different k
  ui10rnnk5 <- random_knn(ui10, k = 5, order_by_distance = FALSE)
  ui10mnnk45 <- merge_knn(list(ui10rnn1, ui10rnnk5))
  expect_equal(ncol(ui10mnnk45$idx), 4)
  check_nbrs(ui10mnnk45, ui10_eucd, tolerance = 1e-6)
})


# query
set.seed(1337)
qnbrs1 <- random_knn_query(reference = ui6, query = ui4, k = 4)
qnbrs2 <- random_knn_query(reference = ui6, query = ui4, k = 4)
qnbrs3 <- random_knn_query(reference = ui6, query = ui4, k = 4)
qnbrsm <- merge_knn(list(qnbrs1, qnbrs2), is_query = TRUE)
test_that("merged query graphs contain valid neighbors and distances", {
  check_query_nbrs(
    nn = qnbrsm,
    query = ui4,
    ref_range = 1:6,
    query_range = 7:10,
    k = 4,
    expected_dist = ui10_eucd,
    tolerance = 1e-6
  )
})

# parallel
ui10mnnt <- merge_knn(list(ui10rnn1, ui10rnn2), n_threads = 1)
test_that("threaded merging improves neighbor quality", {
  expect_true(sum(ui10mnnt$dist) < sum(ui10rnn1$dist))
  expect_true(sum(ui10mnnt$dist) < sum(ui10rnn2$dist))
  check_nbrs(ui10mnnt, ui10_eucd, tolerance = 1e-6)
})

qnbrsmt <- merge_knn(list(qnbrs1, qnbrs2), is_query = TRUE, n_threads = 1)
test_that("threaded query merging preserves valid neighbors and distances", {
  check_query_nbrs(
    nn = qnbrsmt,
    query = ui4,
    ref_range = 1:6,
    query_range = 7:10,
    k = 4,
    expected_dist = ui10_eucd,
    tolerance = 1e-6
  )
})


test_that("merging an empty list returns an empty list", {
  # merge list
  # an empty list returns an empty list
  expect_equal(list(), merge_knn(list()))
})

test_that("merge_knn rejects invalid n_threads values", {
  expect_error(
    merge_knn(list(ui10rnn1, ui10rnn2), n_threads = -1),
    "n_threads must be"
  )
})

test_that("merge_knn rejects malformed graph indices", {
  bad_knn <- list(
    idx = matrix(c(11L, 2L, 3L, 4L), nrow = 4),
    dist = matrix(c(1, 2, 3, 4), nrow = 4)
  )
  expect_error(
    merge_knn(list(bad_knn)),
    "NN graph indices must be between 1 and 4 or 0 for missing entries"
  )

  bad_knn$idx[1, 1] <- NA_integer_
  expect_error(
    merge_knn(list(bad_knn)),
    "NN graph indices must not be NA; use 0 for missing entries"
  )

  bad_query <- list(
    idx = matrix(c(-1L, 2L, 3L, 4L), nrow = 2),
    dist = matrix(c(1, 2, 3, 4), nrow = 2)
  )
  expect_error(
    merge_knn(list(bad_query), is_query = TRUE),
    "Query NN graph indices must be >= 1 or 0 for missing entries"
  )
})

test_that("merging a single query graph preserves its neighbors", {
  # A single graph is unchanged when reverse candidates are disabled.
  ui10rnno <- random_knn(ui10, k = 4, order_by_distance = TRUE)
  ui10mnnl1 <- merge_knn(list(ui10rnno), is_query = TRUE)
  expect_equal(ui10mnnl1$idx, ui10rnno$idx)
  expect_equal(ui10mnnl1$dist, ui10rnno$dist, tolerance = 1e-7)
})

test_that("merging multiple graphs preserves serial and threaded quality", {
  # all 3 matrices are processed
  ui10mnnl3 <- merge_knn(list(ui10rnn1, ui10rnn2, ui10rnn3))
  expect_true(sum(ui10mnnl3$dist) <= sum(ui10mnn$dist))
  check_nbrs(ui10mnnl3, ui10_eucd, tolerance = 1e-6)

  # queries

  # all 3 matrices are processed
  qnbrsml3 <- merge_knn(list(qnbrs1, qnbrs2, qnbrs3), is_query = TRUE)
  expect_true(sum(qnbrsml3$dist) <= sum(qnbrsm$dist))
  check_query_nbrs(
    nn = qnbrsml3,
    query = ui4,
    ref_range = 1:6,
    query_range = 7:10,
    k = 4,
    expected_dist = ui10_eucd,
    tolerance = 1e-6
  )

  # parallel

  # all 3 matrices are processed
  ui10mnnl3t <- merge_knn(list(ui10rnn1, ui10rnn2, ui10rnn3), n_threads = 1)
  expect_true(sum(ui10mnnl3t$dist) <= sum(ui10mnnt$dist))
  check_nbrs(ui10mnnl3t, ui10_eucd, tolerance = 1e-6)

  # queries

  # all 3 matrices are processed
  qnbrsml3t <- merge_knn(
    list(qnbrs1, qnbrs2, qnbrs3),
    is_query = TRUE,
    n_threads = 1
  )
  expect_true(sum(qnbrsml3t$dist) <= sum(qnbrsmt$dist))
  check_query_nbrs(
    nn = qnbrsml3t,
    query = ui4,
    ref_range = 1:6,
    query_range = 7:10,
    k = 4,
    expected_dist = ui10_eucd,
    tolerance = 1e-6
  )
})

test_that("merging ignores missing neighbor entries", {
  # missing indices
  ui10rnn2$idx[1, 2] <- 0
  ui10rnn2$dist[1, 2] <- NA
  ui10mergemissing <- merge_knn(list(ui10rnn1, ui10rnn2))
  expect_equal(range(ui10mergemissing$idx), c(1, 10))

  ui10mergemissingl <- merge_knn(list(ui10rnn1, ui10rnn2, ui10rnn3))
  expect_equal(range(ui10mergemissingl$idx), c(1, 10))
})

test_that("repeated merging leaves both input graphs unchanged", {
  graphs <- list(ui10mnn, ui10rnn3)
  before <- serialize(graphs, NULL)
  merge_knn(graphs)
  expect_identical(serialize(graphs, NULL), before)
  merge_knn(rev(graphs))
  expect_identical(serialize(graphs, NULL), before)
})

test_that("merge rejects incompatible graph sizes and missing fields", {
  smaller <- brute_force_knn(ui6, k = 2)
  expect_error(merge_knn(list(ui10mnn, smaller)), "same number of rows")
  expect_error(merge_knn(list(ui10mnn, ui10mnn["dist"])), "must contain 'idx'")
  expect_error(merge_knn(list(ui10mnn, ui10mnn["idx"])), "must contain 'dist'")
})
