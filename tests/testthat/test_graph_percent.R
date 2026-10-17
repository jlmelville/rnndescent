test_that("search graph sparsity messages report the zero fraction", {
  data <- matrix(c(0, 2, 5, 9, 14), ncol = 1)
  graph <- list(
    idx = matrix(c(2L, 3L, 4L, 5L, 1L), ncol = 1),
    dist = matrix(c(2, 3, 4, 5, 14), ncol = 1)
  )
  messages <- testthat::capture_messages(
    verbose_graph <- prepare_search_graph(
      data,
      graph,
      diversify_prob = NULL,
      pruning_degree_multiplier = NULL,
      use_alt_metric = FALSE,
      verbose = TRUE
    )
  )
  text <- paste(messages, collapse = "\n")
  expect_match(text, "Forward graph has # edges = 5 (80% sparse)", fixed = TRUE)
  expect_match(text, "Reverse graph has # edges = 5 (80% sparse)", fixed = TRUE)
  expect_match(text, "Merged graph has # edges = 10 (60% sparse)", fixed = TRUE)

  quiet_messages <- testthat::capture_messages(
    quiet_graph <- prepare_search_graph(
      data,
      graph,
      diversify_prob = NULL,
      pruning_degree_multiplier = NULL,
      use_alt_metric = FALSE
    )
  )
  expect_length(quiet_messages, 0)
  expect_identical(verbose_graph, quiet_graph)
})
