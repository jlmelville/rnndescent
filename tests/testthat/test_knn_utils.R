test_that("truncation and stable sorting keep the stored distances paired", {
  graph <- list(
    idx = rbind(c(4L, 0L, 2L, 1L), c(1L, 3L, 2L, 0L)),
    dist = rbind(c(9, NA, 1, 3), c(5, 5, 4, NA))
  )
  expect_identical(
    truncate_knn(graph, 1L),
    list(idx = matrix(c(4L, 1L), 2), dist = matrix(c(9, 5), 2))
  )
  expect_identical(dim(truncate_knn(graph, 0)$idx), c(2L, 0L))
  expect_error(truncate_knn(graph, 5), "k must be")
  expect_equal(
    sort_knn(graph),
    list(
      idx = rbind(c(2L, 1L, 4L, 0L), c(2L, 1L, 3L, 0L)),
      dist = rbind(c(1, 3, 9, NA), c(4, 5, 5, NA))
    )
  )

  by <- rbind(c(1, 0, NA, 1), c(3, 2, 1, 0))
  expect_equal(
    sort_knn(graph, by = by),
    list(
      idx = rbind(c(4L, 1L, 2L, 0L), c(2L, 3L, 1L, 0L)),
      dist = rbind(c(9, 3, 1, NA), c(4, 5, 5, NA))
    )
  )
  expect_error(sort_knn(graph, by = t(by)), "same dimensions")
})

test_that("self utilities select by identity and require enough candidates", {
  graph <- list(
    idx = rbind(
      c(2L, 1L, 1L, 3L, 0L),
      c(1L, 3L, 2L, 2L, 0L),
      c(2L, 1L, 0L, 0L, 0L)
    ),
    dist = rbind(
      c(0, 0.25, 0.8, 1, NA),
      c(2, 1, 0.5, 0.7, NA),
      c(0, 1, NA, NA, NA)
    )
  )
  nonself <- remove_self_knn(graph, k = 2)
  expect_equal(
    nonself,
    list(
      idx = rbind(c(2L, 3L), c(1L, 3L), c(2L, 1L)),
      dist = rbind(c(0, 1), c(2, 1), c(0, 1))
    )
  )
  expect_error(remove_self_knn(graph, 3), "row 1 has fewer than 3 nonself")
  expect_error(self_first_knn(graph, 3), "row 3 has no self neighbor")
  expect_equal(
    self_first_knn(graph, 3, self_dist = 0.75),
    list(
      idx = cbind(1:3, nonself$idx),
      dist = cbind(c(0.25, 0.5, 0.75), nonself$dist)
    )
  )
  expect_equal(
    self_first_knn(nonself, 3, self_dist = 0),
    list(idx = cbind(1:3, nonself$idx), dist = cbind(0, nonself$dist))
  )
  singleton <- list(idx = matrix(1L), dist = matrix(0))
  expect_identical(dim(remove_self_knn(singleton)$idx), c(1L, 0L))
})

test_that("sparse conversion preserves zero edges and reference dimensions", {
  graph <- list(
    idx = rbind(c(1L, 3L, 3L, 0L), c(1L, 0L, 0L, 0L)),
    dist = rbind(c(0, 2, 4, NA), c(0, NA, NA, NA))
  )
  sparse <- knn_to_sparse(graph, n_reference = 5L)
  expect_s4_class(sparse, "dgCMatrix")
  expect_identical(dim(sparse), c(2L, 5L))
  expect_equal(sparse[1, 3], 4)
  # Both the diagonal and the off-diagonal zero are stored edges.
  expect_equal(sparse@x, c(0, 0, 4))
  expect_identical(
    knn_to_sparse(graph, n_reference = 5L, transpose = TRUE),
    Matrix::t(sparse)
  )
  expect_error(knn_to_sparse(graph), "indices must be between 1 and 2")
  graph$dist[1, 4] <- 0
  expect_error(knn_to_sparse(graph, 5), "idx = 0 and dist = NA together")
})

test_that("fixed-index scores preserve candidate order, padding and duplicates", {
  data <- rbind(c(0, 0), c(3, 0), c(0, 4), c(8, 1))
  query <- rbind(c(0, 0), c(3, 4))
  idx <- rbind(c(4, 1, 0), c(2, 2, 3))
  result <- knn_distances(data, idx, query = query, n_threads = 2L)
  expect_equal(result$idx, idx)
  expect_equal(
    result$dist,
    rbind(c(sqrt(65), 0, NA), c(4, 4, 3)),
    tolerance = 1e-7
  )

  self_idx <- matrix(c(2L, 1L, 4L, 3L), ncol = 1)
  expect_equal(
    knn_distances(data, self_idx, metric = "sqeuclidean"),
    list(idx = self_idx, dist = matrix(c(9, 9, 73, 73), ncol = 1))
  )

  sparse_data <- Matrix::Matrix(t(data), sparse = TRUE)
  sparse_query <- methods::as(
    Matrix::Matrix(t(query), sparse = TRUE),
    "generalMatrix"
  )
  expect_equal(
    knn_distances(
      sparse_data,
      idx,
      metric = "manhattan",
      query = sparse_query,
      obs = "C"
    ),
    list(idx = idx, dist = rbind(c(9, 0, NA), c(4, 4, 3)))
  )
  binary <- rbind(c(TRUE, FALSE), c(TRUE, TRUE))
  binary_idx <- matrix(c(2L, 1L), ncol = 1)
  expect_equal(
    knn_distances(binary, binary_idx, metric = "jaccard"),
    list(idx = binary_idx, dist = matrix(0.5, nrow = 2))
  )
})

test_that("fixed-index scoring checks bounds and dimensions before native access", {
  data <- matrix(1:6, nrow = 3)
  idx <- matrix(c(2, 3, 1), ncol = 1)
  for (bad_idx in c(4, NA, 1.5)) {
    invalid <- idx
    invalid[1, 1] <- bad_idx
    expect_no_warning(expect_error(
      knn_distances(data, invalid),
      "'idx' indices"
    ))
  }
  expect_error(
    knn_distances(data, idx[-1, , drop = FALSE]),
    "'idx' must have 3 rows"
  )
  expect_error(knn_distances(data, idx, query = idx), "same number of features")
  expect_error(
    knn_distances(
      data > 0,
      matrix(1L),
      metric = "jaccard",
      query = matrix(c(TRUE, NA), nrow = 1)
    ),
    "Logical feature values must not be NA"
  )
  expect_no_warning(
    expect_error(
      knn_distances(data, idx, n_threads = 2^31),
      "n_threads must be"
    )
  )
})
