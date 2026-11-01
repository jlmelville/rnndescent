for (sparse in c(FALSE, TRUE)) {
  for (obs in c("R", "C")) {
    test_that(paste("plain Hellinger keeps near counterparts", sparse, obs), {
      reference <- matrix((seq_len(200) * 0.013) %% 1, 20, 10)
      query <- reference + 1e-6
      if (obs == "C") {
        reference <- t(reference)
        query <- t(query)
      }
      if (sparse) {
        reference <- Matrix::Matrix(reference, sparse = TRUE)
        query <- Matrix::Matrix(query, sparse = TRUE)
      }
      plain <- brute_force_knn_query(
        query,
        reference,
        k = 1,
        metric = "hellinger",
        use_alt_metric = FALSE,
        obs = obs
      )
      expect_identical(as.integer(plain$idx), seq_len(20))
      expect_true(all(is.finite(plain$dist)))
      # Float arithmetic still limits tiny-distance accuracy; keep the neighbors.
      expect_true(all(plain$dist < 1e-3))

      default <- brute_force_knn_query(
        query,
        reference,
        k = 1,
        metric = "hellinger",
        obs = obs
      )
      expect_identical(as.integer(default$idx), seq_len(20))
      for (use_alt_metric in c(FALSE, TRUE)) {
        self <- brute_force_knn_query(
          reference,
          reference,
          k = 1,
          metric = "hellinger",
          use_alt_metric = use_alt_metric,
          obs = obs
        )
        expect_identical(as.integer(self$idx), seq_len(20))
      }
      built <- brute_force_knn(
        reference,
        k = 1,
        metric = "hellinger",
        use_alt_metric = FALSE,
        obs = obs
      )
      expect_identical(as.integer(built$idx), seq_len(20))
    })
  }

  test_that(
    paste("plain Hellinger retains finite and zero distances", sparse),
    {
      data <- rbind(c(1, 0), c(0, 1), c(1, 1), c(0, 0))
      # Independent square-root definition, with package zero conventions.
      oracle <- outer(
        seq_len(4),
        seq_len(4),
        Vectorize(function(i, j) {
          x <- data[i, ]
          y <- data[j, ]
          sx <- sum(x)
          sy <- sum(y)
          if (sx == 0 || sy == 0) {
            return(if (sx == sy) 0 else 1)
          }
          sqrt(sum((sqrt(x / sx) - sqrt(y / sy))^2) / 2)
        })
      )
      input <- if (sparse) Matrix::Matrix(data, sparse = TRUE) else data
      nn <- brute_force_knn_query(
        input,
        input,
        k = 4,
        metric = "hellinger",
        use_alt_metric = FALSE
      )
      selected <- matrix(
        oracle[cbind(rep(seq_len(4), 4), as.vector(nn$idx))],
        4
      )
      expect_equal_abs(nn$dist, selected, tolerance = 1e-6)
    }
  )
}
