test_that("hub splits shift an unbalanced bisector to the median", {
  data <- matrix(0:64, nrow = 1)
  degrees <- c(300L, 200L, 100L, rep(1L, 62))
  forest <- rnn_hub_forest_build(data, degrees, "sqeuclidean", 4L, 200L)
  tree <- forest$trees[[1L]]
  expect_equal(tree$hyperplanes[1L, ], -1)
  expect_equal(tree$offsets[[1L]], 32)
  expect_identical(sort(tree$indices), 0:64)

  size_at <- function(node) {
    children <- tree$children[node, ]
    if (is.nan(tree$offsets[[node]])) {
      return(diff(children))
    }
    sizes <- vapply(children + 1L, size_at, numeric(1L))
    expect_gte(min(sizes), sum(sizes) / 10)
    sum(sizes)
  }
  expect_equal(size_at(1L), 65)
  leaves <- tree$children[is.nan(tree$offsets), , drop = FALSE]
  expect_lte(max(leaves[, 2L] - leaves[, 1L]), 4)
})

test_that("hub trees honor the size and depth controls on tied points", {
  data <- matrix(0, nrow = 1, ncol = 11)
  degrees <- rep(0L, ncol(data))
  build <- function(depth) {
    rnn_hub_forest_build(data, degrees, "euclidean", 1L, depth)$trees[[1L]]
  }
  tree <- build(200L)
  expect_identical(sort(tree$indices), 0:10)
  expect_equal(sum(is.nan(tree$offsets)), 11)
  expect_equal(tree$leaf_size, 1)

  root <- build(0L)
  expect_equal(root$offsets, NaN)
  expect_identical(unname(root$children[1L, ]), c(0L, 11L))
})

test_that("dense Euclidean indexes use the hub tree for one explicit search tree", {
  data <- matrix(c(0:19, 40:59), ncol = 1)
  build <- function(...) {
    set.seed(739L)
    rnnd_build(data, k = 4L, n_trees = 2L, leaf_size = 8L, ...)
  }
  hub <- build()
  rp <- build(n_search_trees = 2L)
  expect_identical(hub$graph, rp$graph)
  expect_identical(hub$search_graph, rp$search_graph)
  expect_length(rp$search_forest$trees, 2)
  expect_length(hub$search_forest$trees, 1)
  degrees <- k_occur(hub$graph, include_self = FALSE)
  expected <- rnn_hub_forest_build(t(data), degrees, "sqeuclidean", 8L, 200L)
  expect_identical(hub$search_forest$trees, expected$trees)

  plain <- build(use_alt_metric = FALSE)
  expect_identical(plain$search_forest$actual_metric, "euclidean")
  expect_identical(build(margin = "implicit")$search_forest$margin, "implicit")
  for (index in list(hub, plain, rp)) {
    result <- rnnd_query(index, data[4:7, , drop = FALSE], k = 3L)
    expect_identical(dim(result$idx), c(4L, 3L))
    expect_true(all(result$idx > 0 & is.finite(result$dist)))
  }
})

test_that("parallel hub construction preserves serial splits and leaf order", {
  # Use a shallow, wide tree to exercise parallel construction cheaply.
  degree <- seq_len(1025L)
  data <- matrix(rep(degree, each = 8192L), nrow = 8192L)
  build <- function(threads) {
    rnn_hub_forest_build(data, degree, "sqeuclidean", 30L, 2L, threads)
  }

  serial <- build(0L)
  expect_gt(nrow(serial$trees[[1L]]$children), 3L)
  expect_true(identical(build(4L), serial, num.eq = FALSE))
})
