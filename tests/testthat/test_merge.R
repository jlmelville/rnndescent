library(rnndescent)
context("Merging")

set.seed(1337)
ui10rnn1 <- random_knn(ui10, k = 4, order_by_distance = FALSE)
ui10rnn2 <- random_knn(ui10, k = 4, order_by_distance = FALSE)
ui10rnn3 <- random_knn(ui10, k = 4, order_by_distance = FALSE)

# serial
ui10mnn <- merge_knn(ui10rnn1, ui10rnn2)
expect_true(sum(ui10mnn$dist) < sum(ui10rnn1$dist))
expect_true(sum(ui10mnn$dist) < sum(ui10rnn2$dist))
check_nbrs(ui10mnn, ui10_eucd, tol = 1e-6)

# different k
ui10rnnk5 <- random_knn(ui10, k = 5, order_by_distance = FALSE)
ui10mnnk45 <- merge_knn(ui10rnn1, ui10rnnk5)
expect_equal(ncol(ui10mnnk45$idx), 4)
check_nbrs(ui10mnnk45, ui10_eucd, tol = 1e-6)

ui10mnnk54 <- merge_knn(ui10rnnk5, ui10rnn1)
expect_equal(ncol(ui10mnnk54$idx), 5)
check_nbrs(ui10mnnk54, ui10_eucd, tol = 1e-6)

# query
set.seed(1337)
qnbrs1 <- random_knn_query(reference = ui6, query = ui4, k = 4)
qnbrs2 <- random_knn_query(reference = ui6, query = ui4, k = 4)
qnbrs3 <- random_knn_query(reference = ui6, query = ui4, k = 4)
qnbrsm <- merge_knn(qnbrs1, qnbrs2, is_query = TRUE)
check_query_nbrs(nn = qnbrsm, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

# parallel
ui10mnnt <- merge_knn(ui10rnn1, ui10rnn2, n_threads = 1)
expect_true(sum(ui10mnnt$dist) < sum(ui10rnn1$dist))
expect_true(sum(ui10mnnt$dist) < sum(ui10rnn2$dist))
check_nbrs(ui10mnnt, ui10_eucd, tol = 1e-6)

qnbrsmt <- merge_knn(qnbrs1, qnbrs2, is_query = TRUE, n_threads = 1)
check_query_nbrs(nn = qnbrsmt, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)


# merge list
# an empty list returns an empty list
expect_equal(list(), merge_knnl(list()))

# one list returns the original list (apart from some casting of distances)
ui10rnno <- random_knn(ui10, k = 4, order_by_distance = TRUE)
ui10mnnl1 <- merge_knnl(list(ui10rnno))
expect_equal(ui10mnnl1$idx, ui10rnno$idx)
expect_equal(ui10mnnl1$dist, ui10rnno$dist, tol = 1e-7)

# serial
# for two matrices merge_knnl and merge_knn give the same results
ui10mnnl <- merge_knnl(list(ui10rnn1, ui10rnn2))
expect_equal(ui10mnnl$idx, ui10mnn$idx)
expect_equal(ui10mnnl$dist, ui10mnn$dist)

# all 3 matrices are processed
ui10mnnl3 <- merge_knnl(list(ui10rnn1, ui10rnn2, ui10rnn3))
expect_true(sum(ui10mnnl3$dist) < sum(ui10mnn$dist))
check_nbrs(ui10mnnl3, ui10_eucd, tol = 1e-6)

# queries
# for two matrices merge_knnl and merge_knn give the same results
qnbrsml <- merge_knnl(list(qnbrs1, qnbrs2), is_query = TRUE)
expect_equal(qnbrsml$idx, qnbrsm$idx)
expect_equal(qnbrsml$dist, qnbrsm$dist)

# all 3 matrices are processed
qnbrsml3 <- merge_knnl(list(qnbrs1, qnbrs2, qnbrs3), is_query = TRUE)
expect_true(sum(qnbrsml3$dist) < sum(qnbrsm$dist))
check_query_nbrs(nn = qnbrsml3, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

# parallel
# for two matrices merge_knnl and merge_knn give the same results
ui10mnnlt <- merge_knnl(list(ui10rnn1, ui10rnn2), n_threads = 1)
expect_equal(ui10mnnlt$idx, ui10mnnt$idx)
expect_equal(ui10mnnlt$dist, ui10mnnt$dist)

# all 3 matrices are processed
ui10mnnl3t <- merge_knnl(list(ui10rnn1, ui10rnn2, ui10rnn3), n_threads = 1)
expect_true(sum(ui10mnnl3t$dist) < sum(ui10mnnt$dist))
check_nbrs(ui10mnnl3t, ui10_eucd, tol = 1e-6)

# queries
# for two matrices merge_knnl and merge_knn give the same results
qnbrsmlt <- merge_knnl(list(qnbrs1, qnbrs2), is_query = TRUE, n_threads = 1)
expect_equal(qnbrsmlt$idx, qnbrsmt$idx)
expect_equal(qnbrsmlt$dist, qnbrsmt$dist)

# all 3 matrices are processed
qnbrsml3t <- merge_knnl(list(qnbrs1, qnbrs2, qnbrs3), is_query = TRUE, n_threads = 1)
expect_true(sum(qnbrsml3t$dist) < sum(qnbrsmt$dist))
check_query_nbrs(nn = qnbrsml3, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

# Errors ------------------------------------------------------------------

expect_error(
  validate_nn_graph(list(
    idx = matrix(nrow = 10, ncol = 2),
    dist = matrix(nrow = 11, ncol = 2)
  )),
  "nn matrix has 11 rows"
)
expect_error(
  validate_nn_graph(list(
    idx = matrix(nrow = 10, ncol = 2),
    dist = matrix(nrow = 10, ncol = 3)
  )),
  "nn matrix has 3 cols"
)
expect_error(
  validate_are_mergeable(
    list(idx = matrix(nrow = 10, ncol = 2), dist = matrix(nrow = 10, ncol = 2)),
    list(idx = matrix(nrow = 11, ncol = 5), dist = matrix(nrow = 11, ncol = 5))
  ),
  "must have same number of rows"
)

expect_error(
  validate_are_mergeablel(list(list(
    idx = matrix(nrow = 10, ncol = 2),
    dist = matrix(nrow = 11, ncol = 2)
  ))),
  "nn matrix has 11 rows"
)
expect_error(
  validate_are_mergeablel(list(list(
    idx = matrix(nrow = 10, ncol = 2),
    dist = matrix(nrow = 10, ncol = 3)
  ))),
  "nn matrix has 3 cols"
)
expect_error(
  validate_are_mergeablel(list(
    list(idx = matrix(nrow = 10, ncol = 2), dist = matrix(nrow = 10, ncol = 2)),
    list(idx = matrix(nrow = 11, ncol = 5), dist = matrix(nrow = 11, ncol = 5))
  )),
  "must have same number of rows"
)
expect_error(
  validate_are_mergeablel(list(
    list(idx = matrix(nrow = 10, ncol = 2), dist = matrix(nrow = 10, ncol = 2)),
    list(badidx = matrix(nrow = 10, ncol = 5), dist = matrix(nrow = 10, ncol = 5))
  )),
  "must contain 'idx'"
)
expect_error(
  validate_are_mergeablel(list(
    list(idx = matrix(nrow = 10, ncol = 2), dist = matrix(nrow = 10, ncol = 2)),
    list(idx = matrix(nrow = 10, ncol = 5), baddist = matrix(nrow = 10, ncol = 5))
  )),
  "must contain 'dist'"
)
