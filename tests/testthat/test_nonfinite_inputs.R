test_that("dense and sparse features reject every non-finite value", {
  data <- matrix(as.double(1:8), ncol = 2)
  message <- "data must contain only finite numeric feature values"
  for (value in c(NA_real_, NaN, Inf, -Inf)) {
    data[2, 1] <- value
    expect_error(brute_force_knn(data, k = 2), message)
    sparse <- Matrix::Matrix(t(data), sparse = TRUE)
    expect_error(brute_force_knn(sparse, k = 2, obs = "C"), message)
  }
  for (storage in c("RsparseMatrix", "TsparseMatrix")) {
    expect_error(
      brute_force_knn(methods::as(sparse, storage), k = 2, obs = "C"),
      message
    )
  }
})

test_that("integer and data-frame features reject missing numeric values", {
  integer_data <- matrix(c(1L, NA_integer_, 3L, 4L), ncol = 2)
  expect_error(
    brute_force_knn(integer_data, k = 1),
    "^data must contain only finite numeric feature values$"
  )
  expect_error(
    brute_force_knn(data.frame(x = c(1, NaN), y = c(2, 3)), k = 1),
    "^data must contain only finite numeric feature values$"
  )
})

test_that("public build and query routes identify the non-finite feature input", {
  set.seed(1934)
  reference <- matrix(c(0, 1, 2, 3, 1, 0, 3, 2), ncol = 2)
  invalid <- reference
  invalid[2, 1] <- NA_real_
  query <- reference[1, , drop = FALSE]
  bad_query <- query
  bad_query[1, 2] <- Inf
  graph <- brute_force_knn(reference, k = 2)
  forest <- rpf_build(reference, n_trees = 1, leaf_size = 2)
  builders <- list(
    random_knn = list(k = 2),
    nnd_knn = list(k = 2, n_iters = 0),
    rnnd_knn = list(k = 2, n_trees = 1, n_iters = 0),
    rnnd_build = list(k = 2, n_trees = 1, n_iters = 0),
    rpf_knn = list(k = 2, n_trees = 1),
    rpf_build = list(n_trees = 1),
    prepare_search_graph = list(graph = graph)
  )
  for (name in names(builders)) {
    expect_error(
      do.call(name, c(list(data = invalid), builders[[name]])),
      "^data must contain only finite numeric feature values$"
    )
  }
  queries <- list(
    brute_force_knn_query = list(),
    random_knn_query = list(),
    graph_knn_query = list(reference_graph = graph),
    rpf_knn_query = list(forest = forest)
  )
  for (name in names(queries)) {
    expect_error(
      do.call(
        name,
        c(list(query = query, reference = invalid, k = 2), queries[[name]])
      ),
      "^reference must contain only finite numeric feature values$"
    )
    expect_error(
      do.call(
        name,
        c(
          list(query = bad_query, reference = reference, k = 2),
          queries[[name]]
        )
      ),
      "^query must contain only finite numeric feature values$"
    )
  }
  index <- rnnd_build(reference, k = 2, n_trees = 1, n_iters = 0)
  expect_error(
    rnnd_query(index, bad_query, k = 2),
    "^query must contain only finite numeric feature values$"
  )
})

test_that("cosine queries reject an infinite reference before neighbor selection", {
  # This sparse reference previously ranked first at distance 1 for a finite query.
  # fmt: skip
  reference <- rbind(
    c(0, 0, 1), c(1, 0, 2), c(0, 3, 1), c(4, Inf, 0),
    c(2, 5, 1), c(7, 0, 3), c(3, 2, 6), c(6, 4, 2)
  )
  query <- matrix(c(-1, 0, -1), nrow = 1)
  for (sparse in c(FALSE, TRUE)) {
    r <- if (sparse) Matrix::Matrix(reference, sparse = TRUE) else reference
    q <- if (sparse) Matrix::Matrix(query, sparse = TRUE) else query
    expect_error(
      brute_force_knn_query(q, r, k = 3, metric = "cosine"),
      "^reference must contain only finite numeric feature values$"
    )
  }
})

test_that("finite complex features retain the existing coercion warning and result", {
  data <- matrix(c(1 + 1i, 2 + 2i, 3 + 1i, 4 + 2i), nrow = 2)
  expect_warning(
    actual <- brute_force_knn(data, k = 1),
    "imaginary parts discarded in coercion",
    fixed = TRUE
  )
  expect_identical(actual, brute_force_knn(Re(data), k = 1))
})
