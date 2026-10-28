test_that("supplied graph errors identify components and preserve check order", {
  data <- matrix(c(0, 0, 1, 0, 0, 2, 3, 1), ncol = 2, byrow = TRUE)
  query <- data[1:2, , drop = FALSE]
  idx_error <- "Expected a matrix of neighbor indices in 'idx'"
  dist_error <- "Expected a matrix of neighbor distances in 'dist'"
  dim_error <- "'idx' and 'dist' must have the same dimensions"
  cases <- list(
    list(list(dist = 1:4), idx_error),
    list(list(idx = 1:4, dist = 1:4), idx_error),
    list(list(idx = matrix(NA_integer_, 3, 2), dist = 1:6), dist_error),
    list(
      list(idx = matrix(NA_integer_, 4, 2), dist = matrix(0, 3, 2)),
      dim_error
    ),
    list(
      list(idx = matrix(0L, 4, 2), dist = matrix(0, 4, 1)),
      dim_error
    ),
    list(
      list(idx = matrix(integer(), 4, 0), dist = matrix(numeric(), 3, 0)),
      dim_error
    ),
    list(
      list(idx = matrix(integer(), 4, 0), dist = matrix(numeric(), 4, 0)),
      "'k' must be greater than 0"
    )
  )
  callers <- list(
    function(graph) prepare_search_graph(data, graph),
    function(graph) graph_knn_query(query, data, graph, k = 2)
  )
  for (call in callers) {
    for (case in cases) {
      error <- tryCatch(call(case[[1]]), error = identity)
      expect_identical(class(error), c("simpleError", "error", "condition"))
      expect_identical(conditionMessage(error), case[[2]])
    }
  }
})
