library(rnndescent)

test_that("builders reject zero features after conversion", {
  empty <- matrix(numeric(0), nrow = 5, ncol = 0)
  sparse <- Matrix::sparseMatrix(
    i = integer(0),
    j = integer(0),
    x = numeric(0),
    dims = c(5, 0)
  )
  graph <- brute_force_knn(matrix(1:10, nrow = 5), k = 2)
  cases <- list(
    rnnd_knn = list(data = empty, k = 2),
    rnnd_build = list(data = data.frame(label = letters[1:5]), k = 2),
    nnd_knn = list(
      data = matrix(logical(0), nrow = 0, ncol = 5),
      obs = "C",
      k = 2,
      metric = "hamming"
    ),
    brute_force_knn = list(data = sparse, k = 2),
    random_knn = list(
      data = methods::as(Matrix::t(sparse), "RsparseMatrix"),
      obs = "C",
      k = 2
    ),
    rpf_knn = list(data = methods::as(sparse, "TsparseMatrix"), k = 2),
    rpf_build = list(data = data.frame(label = factor(letters[1:5]))),
    prepare_search_graph = list(data = t(empty), graph = graph, obs = "C")
  )
  for (name in names(cases)) {
    expect_error(
      do.call(name, cases[[name]]),
      "^data must contain at least one feature after conversion$"
    )
  }
})

test_that("queries identify which converted input has no features", {
  data <- matrix(1:10, nrow = 5)
  empty <- matrix(numeric(0), nrow = 5, ncol = 0)
  graph <- brute_force_knn(data, k = 2)
  forest <- rpf_build(data, n_trees = 1)
  cases <- list(
    brute_force_knn_query = list(k = 2),
    random_knn_query = list(k = 2, obs = "C"),
    graph_knn_query = list(reference_graph = graph, k = 2),
    rpf_knn_query = list(forest = forest, k = 2, obs = "C")
  )
  for (name in names(cases)) {
    controls <- cases[[name]]
    column_obs <- identical(controls$obs, "C")
    valid <- if (column_obs) t(data) else data
    invalid <- if (column_obs) t(empty) else empty
    for (input in c("query", "reference")) {
      args <- c(list(query = valid, reference = valid), controls)
      args[[input]] <- invalid
      expect_error(
        do.call(name, args),
        paste0(
          "^",
          input,
          " must contain at least one feature after conversion$"
        )
      )
    }
  }

  index <- rnnd_build(data, k = 2, n_trees = 1, n_iters = 0)
  expect_error(
    rnnd_query(index, t(empty), k = 2, obs = "C"),
    "^query must contain at least one feature after conversion$"
  )
  expect_equal(rnnd_query(index, data[1:2, ], k = 1)$dist, matrix(0, 2, 1))
})

test_that("feature validation preserves values and positive-feature shapes", {
  data <- matrix(1:10, nrow = 5)
  mixed <- data.frame(data, label = letters[1:5])
  expect_equal(brute_force_knn(mixed, k = 2), brute_force_knn(data, k = 2))
  expect_equal(brute_force_knn(matrix(0, 5, 2), k = 2)$dist, matrix(0, 5, 2))
  expect_equal(
    brute_force_knn(matrix(FALSE, 5, 2), k = 2, metric = "hamming")$dist,
    matrix(0, 5, 2)
  )

  # Observe zero observations at conversion, without entering native work.
  empty_rows <- matrix(numeric(0), nrow = 0, ncol = 2)
  expect_identical(x2m(empty_rows, obs = "R"), empty_rows)
  expect_identical(x2m(t(empty_rows), obs = "C"), t(empty_rows))
})

test_that("zero-feature coercion is catchable in a fresh R process", {
  package_path <- system.file(package = "rnndescent")
  skip_if_not(
    file.exists(file.path(package_path, "Meta", "package.rds")),
    "fresh-process regression uses the installed candidate"
  )
  script <- tempfile(fileext = ".R")
  on.exit(unlink(script))
  writeLines(
    c(
      "library(rnndescent, lib.loc = commandArgs(TRUE)[[1]])",
      "error <- tryCatch(",
      "  brute_force_knn(data.frame(label = letters[1:5]), k = 2),",
      "  error = identity",
      ")",
      "stopifnot(inherits(error, 'error'))",
      "stopifnot(grepl('data must contain at least one feature', conditionMessage(error)))",
      "valid <- brute_force_knn(matrix(1:10, nrow = 5), k = 2)",
      "stopifnot(identical(dim(valid$idx), c(5L, 2L)))"
    ),
    script
  )
  output <- suppressWarnings(system2(
    file.path(R.home("bin"), "Rscript"),
    c("--vanilla", shQuote(script), shQuote(dirname(package_path))),
    stdout = TRUE,
    stderr = TRUE,
    timeout = 30
  ))
  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }
  expect_equal(status, 0L, info = paste(output, collapse = "\n"))
})
