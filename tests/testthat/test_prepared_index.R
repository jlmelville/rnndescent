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

test_that("prepared dense cosine queries preserve signed distances and search budgets", {
  set.seed(120L)
  reference <- matrix(rnorm(321L * 37L), nrow = 321L)
  reference[1L, ] <- 0
  reference[2L, ] <- -reference[3L, ]
  query <- rbind(reference[1:4, ], matrix(rnorm(12L * 37L), ncol = 37L))
  index <- rnnd_build(
    reference,
    k = 12L,
    metric = "cosine-preprocess",
    n_trees = 3L,
    leaf_size = 12L
  )
  prepared <- rnnd_prepare(index)
  run <- function(index, threads, fraction, k = 8L) {
    set.seed(291L)
    dqrng::dqset.seed(291L)
    result <- rnnd_query(
      index,
      query,
      k = k,
      max_search_fraction = fraction,
      n_threads = threads
    )
    list(result = result, rng = .Random.seed, dq = dqrng::dqrng_get_state())
  }
  for (threads in c(0L, 2L)) {
    for (fraction in c(0, 7.25 / nrow(reference), 1)) {
      actual <- run(prepared, threads, fraction)
      expected <- run(index, threads, fraction)
      expect_identical(actual$result$idx, expected$result$idx)
      expect_lt(max(abs(actual$result$dist - expected$result$dist)), 1e-6)
      expect_identical(actual$result$dist[1L, ], rep(1, 8L))
      expect_identical(actual[c("rng", "dq")], expected[c("rng", "dq")])
    }
  }
  wide <- run(prepared, 0L, 0, k = 200L)$result
  expect_true(any(wide$dist[-1L, ] > 1))
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
  expect_error(rnnd_prepare(implicit), "explicit dense search forest")

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

sparse_prepared_metrics <- c(
  "alternative-cosine",
  "alternative-dot",
  "alternative-hellinger",
  "alternative-jaccard",
  "braycurtis",
  "canberra",
  "chebyshev",
  "correlation",
  "cosine",
  "dice",
  "dot",
  "euclidean",
  "hamming",
  "hellinger",
  "jaccard",
  "jensenshannon",
  "kulsinski",
  "manhattan",
  "matching",
  "rogerstanimoto",
  "russellrao",
  "sokalmichener",
  "sokalsneath",
  "spearmanr",
  "sqeuclidean",
  "symmetrickl",
  "trueangular",
  "tsss",
  "yule"
)

for (metric in sparse_prepared_metrics) {
  test_that(
    paste("prepared sparse queries match ordinary queries for", metric),
    {
      set.seed(739L)
      data <- matrix(sample(0:5, 32L * 8L, replace = TRUE), ncol = 8L)
      if (
        metric %in%
          c(
            "dice",
            "hamming",
            "jaccard",
            "alternative-jaccard",
            "kulsinski",
            "matching",
            "rogerstanimoto",
            "russellrao",
            "sokalmichener",
            "sokalsneath",
            "yule"
          )
      ) {
        data <- (data >= 3) * 1
      } else if (
        metric %in% c("cosine", "correlation", "dot", "alternative-dot")
      ) {
        negative_rows <- seq.int(1L, nrow(data), by = 2L)
        data[negative_rows, ] <- -data[negative_rows, ]
      }
      data <- Matrix::Matrix(data, sparse = TRUE)
      query <- data[c(2L, 4L, 6L, 8L, 11L), , drop = FALSE]
      run <- function(index, query, threads, fraction) {
        set.seed(291L)
        dqrng::dqset.seed(291L)
        result <- rnnd_query(
          index,
          query,
          k = 6L,
          epsilon = 0.2,
          n_threads = threads,
          max_search_fraction = fraction
        )
        list(result = result, rng = .Random.seed, dq = dqrng::dqrng_get_state())
      }
      alt_modes <- if (
        metric %in% c("euclidean", "dot", "hellinger", "jaccard")
      ) {
        c(TRUE, FALSE)
      } else {
        TRUE
      }
      for (margin in c("auto", "implicit")) {
        for (use_alt_metric in alt_modes) {
          index <- rnnd_build(
            data,
            k = 6L,
            metric = metric,
            use_alt_metric = use_alt_metric,
            margin = margin,
            n_trees = 2L,
            n_search_trees = 2L,
            leaf_size = 4L,
            n_iters = 1L
          )
          before <- list(.Random.seed, dqrng::dqrng_get_state())
          prepared <- rnnd_prepare(index)
          expect_identical(list(.Random.seed, dqrng::dqrng_get_state()), before)
          expect_s3_class(prepared, "rnndescent_prepared_index")
          for (threads in c(0L, 2L)) {
            for (fraction in c(0, 5 / nrow(data), 1)) {
              expect_identical(
                run(prepared, query, threads, fraction),
                run(index, query, threads, fraction)
              )
            }
          }
          empty <- query[integer(), , drop = FALSE]
          expect_identical(
            run(prepared, empty, 0L, 1),
            run(index, empty, 0L, 1)
          )
        }
      }
    }
  )
}

test_that("prepared sparse indexes validate inputs and own their snapshot", {
  data <- Matrix::Matrix(matrix(0:39, ncol = 2L), sparse = TRUE)
  query <- data[1:3, , drop = FALSE]
  index <- rnnd_build(
    data,
    k = 4L,
    metric = "cosine",
    n_trees = 2L,
    leaf_size = 3L
  )
  index$search_forest$actual_metric <- "alternative-cosine"
  prepared <- rnnd_prepare(index)

  set.seed(291L)
  expected <- rnnd_query(index, query, k = 4L)
  set.seed(291L)
  expect_identical(
    rnnd_query(prepared, Matrix::t(query), k = 4L, obs = "C"),
    expected
  )

  bad_tree <- index
  bad_tree$search_forest$trees[[1L]]$hyperplanes_ptr[1L] <- -1L
  expect_error(rnnd_prepare(bad_tree), "Invalid sparse storage")
  bad_tree <- rnnd_build(data, k = 4L, margin = "implicit", leaf_size = 3L)
  bad_tree$search_forest$trees[[1L]]$normal_indices[1L, 1L] <- nrow(data)
  expect_error(rnnd_prepare(bad_tree), "Invalid reference index")
  bad_data <- index
  bad_data$data@x[1L] <- Inf
  expect_error(rnnd_prepare(bad_data), "finite numeric feature values")
  expect_error(
    rnnd_query(prepared, as.matrix(query), k = 4L),
    "require a sparse query"
  )
  expect_error(
    rnnd_query(prepared, query[, 1L, drop = FALSE], k = 4L),
    "same number of features"
  )
  probability <- rnnd_prepare(rnnd_build(data, k = 4L, metric = "hellinger"))
  expect_error(rnnd_query(probability, -query, k = 4L), "non-negative")

  index$data@x[] <- 0
  index$search_forest <- NULL
  set.seed(291L)
  expect_identical(rnnd_query(prepared, query, k = 4L), expected)
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
    sparse <- Matrix::Matrix(data, sparse = TRUE)
    sparse_index <- rnndescent::rnnd_build(sparse, k = 4L, n_iters = 0L)
    transient <- rnndescent::rnnd_prepare(index)
    prepared <- rnndescent::rnnd_prepare(index)
    prepared_sparse <- rnndescent::rnnd_prepare(sparse_index)
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
    error <- tryCatch(
      rnndescent::rnnd_query(prepared_sparse, sparse, k = 4L),
      error = identity
    )
    stopifnot(
      inherits(error, "error"),
      grepl("Prepared index is invalid", conditionMessage(error))
    )
    rebuilt <- rnndescent::rnnd_prepare(index)
    stopifnot(nrow(rnndescent::rnnd_query(rebuilt, data, k = 4L)$idx) == 20L)
    rebuilt_sparse <- rnndescent::rnnd_prepare(sparse_index)
    stopifnot(
      nrow(rnndescent::rnnd_query(rebuilt_sparse, sparse, k = 4L)$idx) == 20L
    )
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
