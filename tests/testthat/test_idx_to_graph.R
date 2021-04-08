library(rnndescent)
context("Index to graph")

testthat::test_that("convert reference graph", {
  set.seed(1337)
  ui4_nnd <- nnd_knn(ui4, k = 4)
  i2g <- idx_to_graph(ui4, ui4_nnd$idx)
  expect_equal(i2g$dist, ui4_nnd$dist, tol = 1e-7)
  expect_equal(i2g$idx, ui4_nnd$idx)

  # unpack idx from graph
  i2g <- idx_to_graph(ui4, ui4_nnd)
  expect_equal(i2g$dist, ui4_nnd$dist, tol = 1e-7)
  expect_equal(i2g$idx, ui4_nnd$idx)

  # non-default metric
  set.seed(1337)
  ui4_nndc <- nnd_knn(ui4, k = 4, metric = "cosine")
  i2g <- idx_to_graph(ui4, ui4_nndc$idx, metric = "cosine")
  expect_equal(i2g$dist, ui4_nndc$dist, tol = 1e-7)
  expect_equal(i2g$idx, ui4_nndc$idx)

  # multi-threading
  i2g <- idx_to_graph(ui4, ui4_nnd$idx, n_threads = 1)
  expect_equal(i2g$dist, ui4_nnd$dist, tol = 1e-7)
  expect_equal(i2g$idx, ui4_nnd$idx)
})

testthat::test_that("convert reference + query graph", {
  set.seed(1337)
  ui6_nnd <- nnd_knn(ui6, k = 4)
  qnbrs4 <- nnd_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, k = 4)

  i2g <- idx_to_graph_query(query = ui4, reference = ui6, idx = qnbrs4)
  expect_equal(i2g$dist, qnbrs4$dist, tol = 1e-7)
  expect_equal(i2g$idx, qnbrs4$idx)

  i2g <- idx_to_graph_query(query = ui4, reference = ui6, idx = qnbrs4$idx)
  expect_equal(i2g$dist, qnbrs4$dist, tol = 1e-7)
  expect_equal(i2g$idx, qnbrs4$idx)

  # non-default metric
  set.seed(1337)
  ui6_nnd <- nnd_knn(ui6, k = 4, metric = "cosine")
  qnbrs4 <- nnd_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, k = 4, metric = "cosine")
  i2g <- idx_to_graph_query(query = ui4, reference = ui6, idx = qnbrs4, metric = "cosine")
  expect_equal(i2g$dist, qnbrs4$dist, tol = 1e-7)
  expect_equal(i2g$idx, qnbrs4$idx)

  i2g <- idx_to_graph_query(query = ui4, reference = ui6, idx = qnbrs4$idx, metric = "cosine")
  expect_equal(i2g$dist, qnbrs4$dist, tol = 1e-7)
  expect_equal(i2g$idx, qnbrs4$idx)

  # multi-threading
  i2g <- idx_to_graph_query(query = ui4, reference = ui6, idx = qnbrs4$idx, metric = "cosine", n_threads = 1)
  expect_equal(i2g$dist, qnbrs4$dist, tol = 1e-7)
  expect_equal(i2g$idx, qnbrs4$idx)
})
