test_that("descent explains malformed initial graph components in check order", {
  data <- matrix(c(0, 0, 1, 0, 0, 2, 3, 1), ncol = 2, byrow = TRUE)
  cases <- list(
    list(FALSE, "initial neighbor graph must be a list"),
    list(
      list(dist = 1:2),
      "initial neighbor graph must contain an 'idx' component"
    ),
    list(
      list(idx = 1:4, dist = 1:8),
      "initial neighbor graph 'idx' must be a matrix"
    ),
    list(
      list(idx = matrix(NA_integer_, 3, 2), dist = 1:6),
      "initial neighbor graph 'dist' must be a matrix"
    ),
    list(
      list(idx = matrix(NA_integer_, 4, 2), dist = matrix(1, 3, 2)),
      "initial neighbor graph 'idx' and 'dist' must have the same dimensions"
    ),
    list(
      list(idx = matrix(1L, 4, 2), dist = matrix(1, 4, 1)),
      "initial neighbor graph 'idx' and 'dist' must have the same dimensions"
    )
  )
  for (case in cases) {
    set.seed(731)
    before <- .Random.seed
    error <- tryCatch(
      nnd_knn(data, k = 2, init = case[[1]], n_iters = 1),
      error = identity
    )
    expect_identical(conditionMessage(error), case[[2]])
    expect_identical(class(error), c("simpleError", "error", "condition"))
    expect_identical(.Random.seed, before)
  }
})

test_that("graph queries explain malformed initial graph matrices", {
  data <- matrix(c(0, 0, 1, 0, 0, 2, 3, 1), ncol = 2, byrow = TRUE)
  graph <- brute_force_knn(data, k = 2)
  query <- data[1:2, , drop = FALSE]
  cases <- list(
    list(
      list(idx = 1:2, dist = 1:2),
      "initial neighbor graph 'idx' must be a matrix"
    ),
    list(
      list(idx = matrix(NA_integer_, 3, 2), dist = 1:6),
      "initial neighbor graph 'dist' must be a matrix"
    ),
    list(
      list(idx = matrix(NA_integer_, 3, 2), dist = matrix(1, 2, 2)),
      "initial neighbor graph 'idx' and 'dist' must have the same dimensions"
    )
  )
  for (case in cases) {
    set.seed(731)
    before <- .Random.seed
    error <- tryCatch(
      graph_knn_query(query, data, graph, k = 2, init = case[[1]]),
      error = identity
    )
    expect_identical(conditionMessage(error), case[[2]])
    expect_identical(class(error), c("simpleError", "error", "condition"))
    expect_identical(.Random.seed, before)
  }
})
