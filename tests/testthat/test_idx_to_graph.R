library(rnndescent)
context("index to graph")

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
})
