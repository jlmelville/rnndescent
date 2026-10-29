library(rnndescent)

test_that("graph-query reference validation precedes initialization errors", {
  reference <- matrix(1:6, nrow = 3)
  query <- matrix(1:2, nrow = 1)
  graph <- Matrix::sparseMatrix(i = 1L, j = 2L, x = 1, dims = c(2L, 2L))
  for (init in list(NULL, TRUE)) {
    error <- tryCatch(
      graph_knn_query(
        query,
        reference,
        graph,
        init = init,
        k = stop("k was forced")
      ),
      error = identity
    )
    expect_identical(
      conditionMessage(error),
      "reference_graph must have dimensions 3 x 3, but has 2 x 2"
    )
    expect_s3_class(error, "simpleError")
  }
})

test_that("unsupported graph-query initialization leaves k unforced", {
  reference <- matrix(1:6, nrow = 3)
  query <- matrix(1:2, nrow = 1)
  graph <- Matrix::sparseMatrix(
    i = c(1L, 2L, 3L),
    j = c(2L, 3L, 1L),
    x = 1,
    dims = c(3L, 3L)
  )
  error <- tryCatch(
    graph_knn_query(
      query,
      reference,
      graph,
      init = TRUE,
      k = stop("k was forced")
    ),
    error = identity
  )
  expect_identical(
    conditionMessage(error),
    "'init' must be NULL, a random projection forest, a neighbor-index matrix, or a graph list containing 'idx'"
  )
  expect_s3_class(error, "simpleError")
})
