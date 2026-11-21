test_that("descent explains malformed initial graph components", {
  data <- matrix(c(0, 0, 1, 0, 0, 2, 3, 1), ncol = 2, byrow = TRUE)
  cases <- list(
    not_a_list = list(FALSE, "initial neighbor graph must be a list"),
    missing_idx = list(
      list(dist = 1:2),
      "initial neighbor graph must contain an 'idx' component"
    ),
    idx_vector = list(
      list(idx = 1:4),
      "initial neighbor graph 'idx' must be a matrix"
    ),
    dist_vector = list(
      list(idx = matrix(1L, 4, 2), dist = 1:6),
      "initial neighbor graph 'dist' must be a matrix"
    ),
    row_mismatch = list(
      list(idx = matrix(1L, 4, 2), dist = matrix(1, 3, 2)),
      "initial neighbor graph 'idx' and 'dist' must have the same dimensions"
    ),
    column_mismatch = list(
      list(idx = matrix(1L, 4, 2), dist = matrix(1, 4, 1)),
      "initial neighbor graph 'idx' and 'dist' must have the same dimensions"
    )
  )
  for (case in cases) {
    expect_error(
      nnd_knn(data, k = 2, init = case[[1]], n_iters = 0),
      case[[2]],
      fixed = TRUE
    )
  }
})

test_that("graph queries diagnose malformed initial distances", {
  data <- matrix(1:8, ncol = 2)
  graph <- brute_force_knn(data, k = 2)
  query <- data[1:2, , drop = FALSE]
  init <- list(idx = matrix(1L, 2, 2), dist = 1:4)
  expect_error(
    graph_knn_query(query, data, graph, k = 2, init = init),
    "initial neighbor graph 'dist' must be a matrix",
    fixed = TRUE
  )
})
