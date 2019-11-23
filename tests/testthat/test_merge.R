library(rnndescent)
context("Merging")

set.seed(1337)
ui10rnn1 <- random_knn(ui10, k = 4, order_by_distance = FALSE)
ui10rnn2 <- random_knn(ui10, k = 4, order_by_distance = FALSE)

# serial
ui10mnn <- merge_knn(ui10rnn1, ui10rnn2)
expect_true(sum(ui10mnn$dist) < sum(ui10rnn1$dist))
expect_true(sum(ui10mnn$dist) < sum(ui10rnn2$dist))
check_nbrs(ui10mnn, ui10_eucd, tol = 1e-6)

# query
set.seed(1337)
qnbrs1 <- random_knn_query(reference = ui6, query = ui4, k = 4)
qnbrs2 <- random_knn_query(reference = ui6, query = ui4, k = 4)
qnbrsm <- merge_knn(qnbrs1, qnbrs2, is_query = TRUE)
check_query_nbrs(nn = qnbrsm, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

# parallel
ui10mnn <- merge_knn(ui10rnn1, ui10rnn2, n_threads = 1)
expect_true(sum(ui10mnn$dist) < sum(ui10rnn1$dist))
expect_true(sum(ui10mnn$dist) < sum(ui10rnn2$dist))
check_nbrs(ui10mnn, ui10_eucd, tol = 1e-6)

qnbrsm <- merge_knn(qnbrs1, qnbrs2, is_query = TRUE, n_threads = 1)
check_query_nbrs(nn = qnbrsm, query = ui4, ref_range = 1:6, query_range = 7:10, k = 4, expected_dist = ui10_eucd, tol = 1e-6)

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
