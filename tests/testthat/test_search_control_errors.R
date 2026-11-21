test_that("search pruning rejects invalid degree and probability values", {
  data <- matrix(1:6, nrow = 3)
  graph <- brute_force_knn(data, k = 2)
  for (value in list(0, NA_real_, c(1, 0))) {
    expect_error(
      prepare_search_graph(data, graph, pruning_degree_multiplier = value),
      "'pruning_degree_multiplier' must be greater than 0"
    )
  }
  for (value in list(-1, 2, NA_real_, c(0, 2))) {
    expect_error(
      prepare_search_graph(data, graph, diversify_prob = value),
      "'diversify_prob' must be between 0 and 1"
    )
  }
})
