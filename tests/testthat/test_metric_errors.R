test_that("unsupported metrics identify the name and data context", {
  data <- matrix(c(0, 0, 1, 0, 0, 2, 3, 1), ncol = 2, byrow = TRUE)
  inputs <- list(
    dense_numeric = list(data = data, context = "dense"),
    sparse_numeric = list(
      data = methods::as(Matrix::Matrix(data, sparse = TRUE), "dgCMatrix"),
      context = "sparse"
    ),
    dense_logical = list(data = data > 0, context = "dense")
  )
  calls <- list(
    brute_force_knn = function(x, metric) {
      brute_force_knn(x, k = 2, metric = metric)
    },
    brute_force_knn_query = function(x, metric) {
      brute_force_knn_query(x[1:2, , drop = FALSE], x, k = 2, metric = metric)
    },
    rnnd_build = function(x, metric) {
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
  for (input_name in names(inputs)) {
    input <- inputs[[input_name]]
    for (metric in c("euclidian", "bad%metric")) {
      for (call_name in names(calls)) {
        call <- calls[[call_name]]
        test_that(paste(call_name, metric, input_name, sep = " / "), {
          error <- tryCatch(call(input$data, metric), error = identity)
          expect_identical(
            conditionMessage(error),
            paste0("Unknown metric '", metric, "' for ", input$context, " data")
          )
          expect_identical(
            class(error),
            c("Rcpp::exception", "C++Error", "error", "condition")
          )
        })
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
