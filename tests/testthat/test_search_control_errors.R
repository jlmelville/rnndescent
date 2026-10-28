test_that("search pruning errors identify controls before graph validation", {
  degree_error <- "'pruning_degree_multiplier' must be greater than 0"
  probability_error <- "'diversify_prob' must be between 0 and 1"
  cases <- list(
    list(degree = 0, probability = 2, message = degree_error),
    list(degree = 1.5, probability = 2, message = probability_error),
    list(degree = 1.5, probability = -1, message = probability_error),
    list(degree = NA_real_, probability = 2, message = degree_error),
    list(degree = NULL, probability = NA_real_, message = probability_error),
    list(degree = c(1, 0), probability = NULL, message = degree_error),
    list(degree = NULL, probability = c(-1, 2), message = probability_error),
    list(degree = NULL, probability = c(0, -1), message = probability_error)
  )
  for (case in cases) {
    error <- tryCatch(
      prepare_search_graph(
        matrix(0, 3, 2),
        list(),
        pruning_degree_multiplier = case$degree,
        diversify_prob = case$probability
      ),
      error = identity
    )
    expect_identical(class(error), c("simpleError", "error", "condition"))
    expect_identical(conditionMessage(error), case$message)
  }
})

test_that("invalid pruning degree does not force later arguments", {
  error <- tryCatch(
    prepare_search_graph(
      data = stop("data forced"),
      graph = stop("graph forced"),
      pruning_degree_multiplier = 0,
      diversify_prob = stop("probability forced")
    ),
    error = identity
  )
  expect_identical(class(error), c("simpleError", "error", "condition"))
  expect_identical(
    conditionMessage(error),
    "'pruning_degree_multiplier' must be greater than 0"
  )
})
