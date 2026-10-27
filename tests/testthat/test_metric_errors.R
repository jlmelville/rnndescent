test_that("unsupported metrics identify the name and data context", {
  data <- matrix(c(0, 0, 1, 0, 0, 2, 3, 1), ncol = 2, byrow = TRUE)
  inputs <- list(
    list(data = data, context = "dense"),
    list(
      data = methods::as(Matrix::Matrix(data, sparse = TRUE), "dgCMatrix"),
      context = "sparse"
    ),
    list(data = data > 0, context = "dense")
  )
  calls <- list(
    function(x, metric) brute_force_knn(x, k = 2, metric = metric),
    function(x, metric) {
      brute_force_knn_query(x[1:2, , drop = FALSE], x, k = 2, metric = metric)
    },
    function(x, metric) {
      rnnd_build(
        x,
        k = 2,
        metric = metric,
        n_trees = 1,
        leaf_size = 4,
        n_iters = 1
      )
    }
  )
  for (input in inputs) {
    for (metric in c("euclidian", "bad%metric")) {
      for (call in calls) {
        error <- tryCatch(call(input$data, metric), error = identity)
        expect_identical(
          conditionMessage(error),
          paste0("Unknown metric '", metric, "' for ", input$context, " data")
        )
        expect_identical(
          class(error),
          c("Rcpp::exception", "C++Error", "error", "condition")
        )
      }
    }
  }
})

test_that("earlier input errors still precede unsupported metric errors", {
  data <- matrix(c(0, 0, 1, 0, 0, 2, 3, 1), ncol = 2, byrow = TRUE)
  for (x in list(
    data,
    methods::as(Matrix::Matrix(data, sparse = TRUE), "dgCMatrix")
  )) {
    expect_error(
      brute_force_knn(x, k = 0, metric = "euclidian"),
      "k must be a single integer between 1 and 4",
      fixed = TRUE
    )
    expect_error(
      brute_force_knn_query(
        x[, 1, drop = FALSE],
        x,
        k = 2,
        metric = "euclidian"
      ),
      "query and reference must have the same number of features",
      fixed = TRUE
    )
  }
})
