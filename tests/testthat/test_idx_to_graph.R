library(rnndescent)
context("Index to graph")

testthat::test_that("convert reference graph", {
  set.seed(1337)
  ui4_nnd <- nnd_knn(ui4, k = 4)
  i2g <- prepare_init_graph(data = t(ui4), nn = ui4_nnd$idx, k = 4)
  expect_equal(i2g$dist, ui4_nnd$dist, tol = 1e-7)
  expect_equal(i2g$idx, ui4_nnd$idx)

  # unpack idx from graph
  i2g <- prepare_init_graph(data = t(ui4), nn = ui4_nnd, k = 4, recalculate_distances = TRUE)
  expect_equal(i2g$dist, ui4_nnd$dist, tol = 1e-7)
  expect_equal(i2g$idx, ui4_nnd$idx)

  # non-default metric
  set.seed(1337)
  ui4_nndc <- nnd_knn(ui4, k = 4, metric = "cosine")
  i2g <- prepare_init_graph(data = t(ui4), nn = ui4_nndc$idx, k = 4, metric = "cosine")
  expect_equal(i2g$dist, ui4_nndc$dist, tol = 1e-7)
  expect_equal(i2g$idx, ui4_nndc$idx)

  # multi-threading
  i2g <- prepare_init_graph(data = t(ui4), nn = ui4_nnd$idx, k = 4, n_threads = 2)
  expect_equal(i2g$dist, ui4_nnd$dist, tol = 1e-7)
  expect_equal(i2g$idx, ui4_nnd$idx)
})

testthat::test_that("convert reference + query graph", {
  set.seed(1337)
  ui6_nnd <- nnd_knn(ui6, k = 4)
  qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, k = 4)

  i2g <- prepare_init_graph(query = t(ui4), data = t(ui6), nn = qnbrs4, k = 4, recalculate_distances = TRUE)
  expect_equal(i2g$dist, qnbrs4$dist, tol = 1e-7)
  expect_equal(i2g$idx, qnbrs4$idx)

  i2g <- prepare_init_graph(query = t(ui4), data = t(ui6), nn = qnbrs4$idx, k = 4)
  expect_equal(i2g$dist, qnbrs4$dist, tol = 1e-7)
  expect_equal(i2g$idx, qnbrs4$idx)

  # non-default metric
  set.seed(1337)
  ui6_nnd <- nnd_knn(ui6, k = 4, metric = "cosine")
  qnbrs4 <- graph_knn_query(reference = ui6, reference_graph = ui6_nnd, query = ui4, k = 4, metric = "cosine")
  i2g <- prepare_init_graph(query = t(ui4), data = t(ui6), nn = qnbrs4, k = 4, metric = "cosine")
  expect_equal(i2g$dist, qnbrs4$dist, tol = 1e-7)
  expect_equal(i2g$idx, qnbrs4$idx)

  i2g <- prepare_init_graph(query = t(ui4), data = t(ui6), nn = qnbrs4$idx, k = 4, metric = "cosine")
  expect_equal(i2g$dist, qnbrs4$dist, tol = 1e-7)
  expect_equal(i2g$idx, qnbrs4$idx)

  # multi-threading
  i2g <- prepare_init_graph(query = t(ui4), data = t(ui6), nn = qnbrs4$idx, k = 4, metric = "cosine", n_threads = 2)
  expect_equal(i2g$dist, qnbrs4$dist, tol = 1e-7)
  expect_equal(i2g$idx, qnbrs4$idx)
})

test_that("sparse", {
  set.seed(1337); dz6_4 <- random_knn_query(query = ui10z6, reference = ui10z4, k = 4, metric = "cosine")
  i2g <- prepare_init_graph(query = Matrix::t(ui10sp6), data = Matrix::t(ui10sp4),
                            nn = dz6_4$idx, k = 4, metric = "cosine")
  expect_equal(i2g$dist, dz6_4$dist, tol= 1e-6)
  expect_equal(i2g$idx, dz6_4$idx)
})
