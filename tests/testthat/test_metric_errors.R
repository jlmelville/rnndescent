test_that("unsupported metrics identify the name and data context", {
  data <- matrix(1:8, ncol = 2)
  expect_error(
    brute_force_knn(data, k = 2, metric = "euclidian"),
    "Unknown metric 'euclidian' for dense data",
    fixed = TRUE
  )
  sparse <- Matrix::Matrix(data, sparse = TRUE)
  expect_error(
    brute_force_knn_query(sparse, sparse, k = 2, metric = "bad%metric"),
    "Unknown metric 'bad%metric' for sparse data",
    fixed = TRUE
  )
  expect_error(
    rnnd_build(data > 3, k = 2, metric = "bad%metric"),
    "Unknown metric 'bad%metric' for dense data",
    fixed = TRUE
  )
})
