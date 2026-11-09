test_that("prepared dense Euclidean queries preserve results and RNG state", {
  data <- matrix(c(0:63, (0:63)^2 %% 17), ncol = 2)
  query <- data[seq(1, 57, by = 4), , drop = FALSE]
  run <- function(index, query, threads, epsilon, fraction) {
    set.seed(291L)
    dqrng::dqset.seed(291L)
    result <- rnnd_query(
      index,
      query,
      k = 8L,
      epsilon = epsilon,
      max_search_fraction = fraction,
      n_threads = threads
    )
    list(result = result, rng = .Random.seed, dq = dqrng::dqrng_get_state())
  }
  for (use_alt_metric in c(TRUE, FALSE)) {
    set.seed(739L)
    index <- rnnd_build(
      data,
      k = 8L,
      n_trees = 2L,
      leaf_size = 4L,
      n_iters = 1L,
      use_alt_metric = use_alt_metric,
      n_search_trees = if (use_alt_metric) 1L else 2L
    )
    before <- .Random.seed
    prepared <- rnnd_prepare(index)
    expect_identical(.Random.seed, before)
    expect_s3_class(prepared, "rnndescent_prepared_index")
    for (threads in c(0L, 2L)) {
      for (fraction in c(0, 5 / nrow(data), 1)) {
        expect_identical(
          run(prepared, query, threads, 0.2, fraction),
          run(index, query, threads, 0.2, fraction)
        )
      }
      empty <- query[integer(), , drop = FALSE]
      expect_identical(
        run(prepared, empty, threads, 0.1, 1),
        run(index, empty, threads, 0.1, 1)
      )
    }
  }
})

test_that("prepared indexes own a snapshot and reject invalid inputs", {
  data <- matrix(0:63, ncol = 2)
  index <- rnnd_build(data, k = 4L, n_trees = 2L, n_iters = 1L)
  prepared <- rnnd_prepare(index)
  query <- data[1:3, , drop = FALSE]
  set.seed(401L)
  expected <- rnnd_query(prepared, query, k = 4L)

  bad_tree <- index
  bad_tree$search_forest$trees[[1L]]$children[1L, 1L] <- -1L
  expect_error(rnnd_prepare(bad_tree), "Invalid child")
  bad_reference <- index
  bad_reference$data[1L] <- Inf
  expect_error(rnnd_prepare(bad_reference), "finite numeric feature values")
  implicit <- index
  implicit$search_forest$margin <- "implicit"
  expect_error(rnnd_prepare(implicit), "explicit Euclidean search forest")

  index$data[] <- -1L
  index$search_forest <- NULL
  rm(index)
  set.seed(401L)
  expect_identical(rnnd_query(prepared, query, k = 4L), expected)
  expect_error(
    rnnd_query(prepared, query, k = 4L, init = expected),
    "init must be NULL"
  )
  expect_error(
    rnnd_query(prepared, query[, 1L, drop = FALSE], k = 4L),
    "same number of features"
  )
  expect_error(rnnd_query(prepared, query, k = 33L), "between 1 and 32")
  expect_error(
    rnnd_query(prepared, query * Inf, k = 4L),
    "finite numeric feature values"
  )
  restored <- unserialize(serialize(prepared, NULL))
  expect_error(rnnd_query(restored, query, k = 4L), "Call rnnd_prepare\\(\\)")
})

test_that("unloading releases live prepared handles before the DLL", {
  package <- find.package("rnndescent")
  skip_if_not(file.exists(file.path(package, "Meta", "package.rds")))
  script <- tempfile(fileext = ".R")
  output <- tempfile()
  on.exit(unlink(c(script, output)))
  subprocess <- quote({
    .libPaths(c(commandArgs(TRUE)[[1L]], .libPaths()))
    data <- matrix(as.double(1:40), ncol = 2)
    index <- rnndescent::rnnd_build(data, k = 4L, n_iters = 0L)
    transient <- rnndescent::rnnd_prepare(index)
    prepared <- rnndescent::rnnd_prepare(index)
    rm(transient)
    invisible(gc())
    unloadNamespace("rnndescent")
    stopifnot(!"rnndescent" %in% names(getLoadedDLLs()))
    invisible(gc())
    error <- tryCatch(
      rnndescent::rnnd_query(prepared, data, k = 4L),
      error = identity
    )
    stopifnot(
      inherits(error, "error"),
      grepl("Prepared index is invalid", conditionMessage(error))
    )
    rebuilt <- rnndescent::rnnd_prepare(index)
    stopifnot(nrow(rnndescent::rnnd_query(rebuilt, data, k = 4L)$idx) == 20L)
  })
  writeLines(deparse(subprocess), script)
  status <- system2(
    file.path(R.home("bin"), "Rscript"),
    c("--vanilla", shQuote(script), shQuote(dirname(package))),
    stdout = output,
    stderr = output
  )
  expect_equal(
    status,
    0L,
    info = paste(readLines(output, warn = FALSE), collapse = "\n")
  )
})
