for (metric in c("jensenshannon", "symmetrickl")) {
  for (obs in c("R", "C")) {
    test_that(
      paste(metric, "keeps sparse neighbor ranks with shared zeros", obs),
      {
        query <- matrix(c(0.7, 0.3, rep(0, 131070)), nrow = 1)
        reference <- rbind(
          c(0.65, 0.35, rep(0, 131070)),
          c(6.503, 3.497, rep(0, 131070))
        )
        if (obs == "C") {
          query <- t(query)
          reference <- t(reference)
        }
        dense <- brute_force_knn_query(
          query,
          reference,
          k = 2,
          metric = metric,
          obs = obs
        )
        sparse <- brute_force_knn_query(
          methods::as(query, "dgCMatrix"),
          methods::as(reference, "dgCMatrix"),
          k = 2,
          metric = metric,
          obs = obs
        )
        expect_identical(sparse$idx, dense$idx)
        expect_equal_abs(sparse$dist, dense$dist, tolerance = 1e-6)
      }
    )
  }

  test_that(paste(metric, "retains distances without shared zeros"), {
    query <- matrix(c(0.7, 0.2, 0.1), nrow = 1)
    reference <- rbind(c(0.1, 0.3, 0.6), c(0.5, 0.25, 0.25))
    dense <- brute_force_knn_query(query, reference, k = 2, metric = metric)
    sparse <- brute_force_knn_query(
      methods::as(query, "dgCMatrix"),
      methods::as(reference, "dgCMatrix"),
      k = 2,
      metric = metric
    )
    expect_identical(sparse$idx, dense$idx)
    expect_equal_abs(sparse$dist, dense$dist, tolerance = 1e-6)
  })
}

for (obs in c("R", "C")) {
  test_that(paste("sparse symmetric KL includes shared-zero terms", obs), {
    # These positive totals make normalized zero components exceed epsilon.
    query <- matrix(1e-5 * c(0.7, 0.3, 0, 0), nrow = 1)
    reference <- matrix(2e-5 * c(0.6, 0.4, 0, 0), nrow = 1)
    if (obs == "C") {
      query <- t(query)
      reference <- t(reference)
    }
    dense <- brute_force_knn_query(
      query,
      reference,
      k = 1,
      metric = "symmetrickl",
      obs = obs
    )
    sparse <- brute_force_knn_query(
      methods::as(query, "dgCMatrix"),
      methods::as(reference, "dgCMatrix"),
      k = 1,
      metric = "symmetrickl",
      obs = obs
    )
    expect_equal_abs(sparse$dist, dense$dist, tolerance = 1e-6)
  })
}
