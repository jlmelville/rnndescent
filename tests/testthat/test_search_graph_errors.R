test_that("search graph errors identify the malformed graph component", {
  data <- matrix(1:8, nrow = 4)
  graph <- brute_force_knn(data, k = 2)
  expect_error(
    prepare_search_graph(data, graph["dist"]),
    "matrix of neighbor indices in 'idx'"
  )
  graph$dist <- 1:8
  expect_error(
    prepare_search_graph(data, graph),
    "matrix of neighbor distances in 'dist'"
  )
  graph$dist <- matrix(0, 4, 1)
  expect_error(
    graph_knn_query(data, data, graph, k = 2),
    "'idx' and 'dist' must have the same dimensions"
  )
  empty <- list(idx = matrix(integer(), 4, 0), dist = matrix(numeric(), 4, 0))
  expect_error(prepare_search_graph(data, empty), "'k' must be greater than 0")
})
