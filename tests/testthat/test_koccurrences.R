library(rnndescent)
context("Reverse neighbors")

rnbrs <- brute_force_knn(ui10, k = 4, n_threads = 0)
hub10 <- k_occur(rnbrs$idx)
expect_equal(hub10, c(3, 5, 7, 3, 5, 3, 5, 4, 2, 3))
expect_equal(
  k_occur(rnbrs$idx, include_self = FALSE),
  c(2, 4, 6, 2, 4, 2, 4, 3, 1, 2)
)
expect_equal(
  k_occur(rnbrs$idx, k = 2),
  c(2, 1, 2, 2, 2, 3, 3, 2, 2, 1)
)

test_that("ignore missing results", {
  rnbrs$idx[10, 4] <- -1
  expect_equal(
    k_occur(rnbrs),
    c(3, 5, 6, 3, 5, 3, 5, 4, 2, 3)
  )
})

test_that("ignore zero missing results", {
  expect_equal(k_occur(matrix(0L, nrow = 2, ncol = 1)), integer())
  expect_equal(
    k_occur(matrix(c(1L, 0L, 1L, 1L), nrow = 2, byrow = TRUE)),
    3L
  )
})

qnbrs4 <- brute_force_knn_query(reference = ui6, query = ui4, k = 4)
expect_equal(k_occur(qnbrs4$idx), c(1, 3, 4, 3, 4, 1))

qnbrs6 <- brute_force_knn_query(reference = ui4, query = ui6, k = 4)
expect_equal(k_occur(qnbrs6$idx, k = 2), c(5, 4, 1, 2))
expect_equal(k_occur(qnbrs6$idx), c(6, 6, 6, 6))

test_that("k-occurrence of sparse graph", {
  rnbrs_sparse <- graph_to_csparse(rnbrs)
  expect_equal(k_occur(rnbrs_sparse), hub10)
  expect_equal(k_occur(rnbrs_sparse, include_self = FALSE), hub10)
})

test_that("sparse k-occurrence preserves anti-hubs and self handling", {
  sparse_graph <- Matrix::sparseMatrix(
    i = c(1L, 2L),
    j = c(2L, 1L),
    x = c(1, 1),
    dims = c(4L, 4L)
  )
  self_graph <- Matrix::sparseMatrix(
    i = c(1L, 2L),
    j = c(1L, 1L),
    x = c(1, 1),
    dims = c(2L, 2L)
  )

  expect_equal(k_occur(sparse_graph), c(1L, 1L, 0L, 0L))
  expect_equal(k_occur(sparse_graph, include_self = FALSE), c(1L, 1L, 0L, 0L))
  expect_equal(k_occur(self_graph), c(2L, 0L))
  expect_equal(k_occur(self_graph, include_self = FALSE), c(1L, 0L))
})
