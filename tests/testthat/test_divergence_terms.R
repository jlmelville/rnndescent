divergence_terms_reference <- function(x, y, metric) {
  # Match the documented smoothing, then sum all mathematical terms in double.
  epsilon <- 2^-23
  p <- (x + epsilon) / sum(x + epsilon)
  q <- (y + epsilon) / sum(y + epsilon)
  if (metric == "jensenshannon") {
    m <- (p + q) / 2
    return(sum(p * log(p / m) + q * log(q / m)) / 2)
  }
  sum((p - q) * log(p / q))
}

for (metric in c("jensenshannon", "symmetrickl")) {
  defect <- if (metric == "jensenshannon") {
    list(x = c(1e-5, 0), y = c(.99, .01))
  } else {
    list(
      x = c(1e7, rep(1, 512), rep(0, 512)),
      y = c(1e7, rep(0, 512), rep(1, 512))
    )
  }
  cases <- list(
    omitted_terms = defect,
    common_zeros = list(
      x = c(1, rep(0, 1024)),
      y = c(10, rep(0, 1024))
    ),
    full_support = list(x = c(.7, .3), y = c(.6, .4))
  )
  for (case_name in names(cases)) {
    pair <- cases[[case_name]]
    expected <- c(0, divergence_terms_reference(pair$x, pair$y, metric))
    for (sparse in c(FALSE, TRUE)) {
      for (obs in c("R", "C")) {
        test_that(paste(metric, case_name, "full terms", sparse, obs), {
          query <- matrix(pair$x, nrow = 1)
          reference <- rbind(pair$x, pair$y)
          if (obs == "C") {
            query <- t(query)
            reference <- t(reference)
          }
          if (sparse) {
            query <- methods::as(query, "dgCMatrix")
            reference <- methods::as(reference, "dgCMatrix")
          }
          result <- brute_force_knn_query(
            query,
            reference,
            k = 2,
            metric = metric,
            obs = obs,
            n_threads = 0
          )
          # The omitted JS term previously let an unlike point beat exact self;
          # KL dropped the rare counts entirely. Both must retain their distance.
          expect_identical(result$idx, matrix(c(1, 2), nrow = 1))
          expect_true(all(
            is.finite(result$dist) & abs(result$dist - expected) < 2e-7
          ))
        })
      }
    }
  }
}
