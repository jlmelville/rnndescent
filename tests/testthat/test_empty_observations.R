test_that("neighbor builders identify empty data in either orientation", {
  empty <- matrix(numeric(), 0, 2)
  sparse <- Matrix::sparseMatrix(
    i = integer(),
    j = integer(),
    x = numeric(),
    dims = c(2, 0)
  )
  for (case in list(list(empty, "R"), list(sparse, "C"))) {
    for (name in c("brute_force_knn", "random_knn", "nnd_knn", "rpf_knn")) {
      expect_error(
        do.call(name, list(data = case[[1]], k = 1, obs = case[[2]])),
        "^data must contain at least one observation$",
        class = "simpleError"
      )
    }
  }
})

test_that("neighbor queries identify empty references in either orientation", {
  empty <- matrix(numeric(), 0, 2)
  query <- matrix(c(1, 2), 1, 2)
  graph <- list(idx = matrix(integer(), 0, 1), dist = matrix(numeric(), 0, 1))
  # The existing count check precedes forest validation.
  calls <- list(
    list("brute_force_knn_query", list()),
    list("random_knn_query", list()),
    list("rpf_knn_query", list(forest = list())),
    list("graph_knn_query", list(reference_graph = graph)),
    list(
      "graph_knn_query",
      list(reference_graph = graph, init = matrix(1L, 1, 1))
    )
  )
  sparse_reference <- Matrix::sparseMatrix(
    i = integer(),
    j = integer(),
    x = numeric(),
    dims = c(2, 0)
  )
  sparse_query <- methods::as(
    Matrix::Matrix(t(query), sparse = TRUE),
    "generalMatrix"
  )
  cases <- list(
    list(reference = empty, query = query, obs = "R", k = 1),
    list(reference = sparse_reference, query = sparse_query, obs = "C", k = 1)
  )
  for (case in cases) {
    for (call in calls) {
      expect_error(
        do.call(call[[1]], c(case, call[[2]])),
        "^reference must contain at least one observation$",
        class = "simpleError"
      )
    }
  }
})

test_that("a zero-by-zero input identifies the missing feature dimension", {
  expect_error(
    brute_force_knn(matrix(numeric(), 0, 0), k = 1),
    "data must contain at least one feature after conversion"
  )
})
