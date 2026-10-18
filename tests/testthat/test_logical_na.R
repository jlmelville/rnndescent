for (metric in c("euclidean", "hamming")) {
  test_that(paste("missing logical features are rejected for", metric), {
    reference <- rbind(c(TRUE, FALSE), c(TRUE, TRUE))
    missing_reference <- reference
    missing_reference[2, 2] <- NA
    query <- matrix(c(TRUE, NA), nrow = 1)
    error <- "Logical feature values must not be NA"

    expect_error(
      brute_force_knn(missing_reference, k = 1, metric = metric),
      error,
      fixed = TRUE
    )
    expect_error(
      brute_force_knn_query(
        t(query),
        t(reference),
        k = 1,
        metric = metric,
        obs = "C"
      ),
      error,
      fixed = TRUE
    )
    expect_error(
      brute_force_knn_query(
        reference[1, , drop = FALSE],
        missing_reference,
        k = 1,
        metric = metric
      ),
      error,
      fixed = TRUE
    )
  })
}

test_that("explicit-margin builds reject missing logical features before coercion", {
  data <- rbind(c(TRUE, FALSE), c(TRUE, NA))
  error <- "Logical feature values must not be NA"
  expect_error(
    rpf_build(data, n_trees = 1, margin = "explicit"),
    error,
    fixed = TRUE
  )
  expect_error(
    rpf_knn(data, k = 1, n_trees = 1, margin = "explicit"),
    error,
    fixed = TRUE
  )
})

test_that("valid logical features retain zero rows and numeric results", {
  reference <- rbind(c(FALSE, FALSE), c(TRUE, FALSE), c(TRUE, TRUE))
  for (metric in c("euclidean", "hamming")) {
    expect_equal(
      brute_force_knn_query(reference, reference, k = 3, metric = metric),
      brute_force_knn_query(
        reference * 1,
        reference * 1,
        k = 3,
        metric = metric
      )
    )
  }
})
