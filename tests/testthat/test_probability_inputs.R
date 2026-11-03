probability_input_forms <- function(x) {
  list(x, methods::as(Matrix::Matrix(x, sparse = TRUE), "dgCMatrix"))
}

test_that("protected native metrics reject negative self and query features", {
  good <- rbind(c(0, 0), c(1, 0), c(0, 1), c(1, 1))
  bad <- good
  bad[2, 1] <- -0.1
  for (sparse in seq_len(2)) {
    reference <- probability_input_forms(good)[[sparse]]
    negative <- probability_input_forms(bad)[[sparse]]
    for (metric in c(
      "hellinger",
      "alternative-hellinger",
      "jensenshannon",
      "symmetrickl"
    )) {
      expect_error(
        brute_force_knn(
          negative,
          k = 2,
          metric = metric,
          use_alt_metric = FALSE
        ),
        "non-negative feature values in 'data'"
      )
      expect_error(
        brute_force_knn_query(
          reference,
          negative,
          k = 2,
          metric = metric,
          use_alt_metric = FALSE
        ),
        "non-negative feature values in 'reference'"
      )
      expect_error(
        brute_force_knn_query(
          negative,
          reference,
          k = 2,
          metric = metric,
          use_alt_metric = FALSE
        ),
        "non-negative feature values in 'query'"
      )
      zeros <- brute_force_knn_query(
        reference,
        reference,
        k = 2,
        metric = metric,
        use_alt_metric = FALSE
      )
      expect_true(all(is.finite(zeros$dist)))
      expect_identical(zeros$dist[1, 1], 0)
    }
    expect_error(
      brute_force_knn(negative, k = 2, metric = "hellinger"),
      "metric 'alternative-hellinger'.*non-negative"
    )
  }
})

test_that("original negatives are checked before float conversion", {
  data <- rbind(c(0, 0), c(-1e-300, 1), c(1, 0), c(1, 1))
  for (x in probability_input_forms(data)) {
    expect_error(
      random_knn(x, k = 2, metric = "hellinger"),
      "non-negative feature values"
    )
    expect_error(
      nnd_knn(x, k = 2, metric = "hellinger", n_iters = 0),
      "non-negative feature values"
    )
  }
})

test_that("explicit RP admission rejects original negatives before RNG work", {
  data <- rbind(c(0, 0), c(-1e-300, 1), c(1, 0), c(1, 1))
  calls <- list(
    function(x) {
      rpf_build(
        x,
        metric = "hellinger",
        margin = "explicit",
        n_trees = 1,
        leaf_size = 2
      )
    },
    function(x) {
      rpf_knn(
        x,
        k = 2,
        metric = "hellinger",
        margin = "explicit",
        n_trees = 1,
        leaf_size = 2
      )
    }
  )
  for (x in probability_input_forms(data)) {
    for (call in calls) {
      set.seed(42)
      dqrng::dqset.seed(42)
      before <- list(R = .Random.seed, dq = dqrng::dqrng_get_state())
      expect_error(call(x), "non-negative feature values in 'data'")
      expect_identical(
        list(R = .Random.seed, dq = dqrng::dqrng_get_state()),
        before
      )
    }
  }
})

test_that("forest and mutable index context guard both query roles", {
  good <- rbind(c(0, 0), c(1, 0), c(0, 1), c(1, 1))
  bad <- good
  bad[2, 1] <- -0.1
  for (sparse in seq_len(2)) {
    reference <- probability_input_forms(good)[[sparse]]
    negative <- probability_input_forms(bad)[[sparse]]
    forest <- rpf_build(
      reference,
      metric = "hellinger",
      n_trees = 1,
      leaf_size = 2
    )
    graph <- brute_force_knn(reference, k = 2, metric = "hellinger")
    expect_error(
      rpf_knn_query(reference, negative, forest, k = 2),
      "non-negative feature values in 'reference'"
    )
    expect_error(
      rpf_knn_query(negative, reference, forest, k = 2),
      "non-negative feature values in 'query'"
    )
    expect_error(
      graph_knn_query(
        negative,
        reference,
        graph,
        k = 2,
        metric = "euclidean",
        init = forest
      ),
      "non-negative feature values in 'query'"
    )
    index <- rnnd_build(
      reference,
      k = 2,
      metric = "hellinger",
      n_trees = 1,
      leaf_size = 2,
      n_iters = 0,
      diversify_prob = NULL,
      pruning_degree_multiplier = NULL
    )
    index$data[1, 2] <- -1e-300
    expect_error(
      rnnd_query(index, reference, k = 2),
      "non-negative feature values in 'reference'"
    )
  }
})

test_that("signed ordinary and unused feature inputs retain their behavior", {
  good <- rbind(c(0, 0), c(1, 0), c(0, 1), c(1, 1))
  signed <- good - 0.5
  for (x in probability_input_forms(signed)) {
    for (metric in c("euclidean", "cosine", "manhattan")) {
      nn <- brute_force_knn_query(x, x, k = 2, metric = metric)
      expect_true(all(is.finite(nn$dist)))
    }
  }
  forest <- rpf_build(signed, metric = "cosine", n_trees = 1, leaf_size = 2)
  graph <- brute_force_knn(signed, k = 2, metric = "cosine")
  expect_no_error(graph_knn_query(
    signed,
    signed,
    graph,
    k = 2,
    metric = stop("unused caller metric must remain unforced"),
    init = forest
  ))
  graph <- brute_force_knn(good, k = 2, metric = "hellinger")
  expect_no_error(prepare_search_graph(
    signed,
    graph,
    metric = "hellinger",
    diversify_prob = 0,
    pruning_degree_multiplier = NULL
  ))
  expect_error(
    prepare_search_graph(
      signed,
      graph,
      metric = "hellinger",
      diversify_prob = 1,
      pruning_degree_multiplier = NULL
    ),
    "non-negative feature values"
  )
  expect_no_error(brute_force_knn(good > 0, k = 2, metric = "hellinger"))
})

test_that("partially copied probability inputs reject and recover without mutation", {
  good <- matrix(seq_len(24) / 24, nrow = 6)
  bad <- good
  bad[6, 4] <- -1e-300
  for (sparse in seq_len(2)) {
    positive <- probability_input_forms(good)[[sparse]]
    negative <- probability_input_forms(bad)[[sparse]]
    originals <- list(positive = positive, negative = negative)
    expected <- brute_force_knn_query(
      positive,
      positive,
      k = 2,
      metric = "hellinger"
    )
    set.seed(43)
    dqrng::dqset.seed(43)
    before <- list(R = .Random.seed, dq = dqrng::dqrng_get_state())
    for (iteration in seq_len(2)) {
      expect_error(
        brute_force_knn_query(positive, negative, k = 2, metric = "hellinger"),
        "non-negative feature values in 'reference'"
      )
      expect_error(
        brute_force_knn_query(negative, positive, k = 2, metric = "hellinger"),
        "non-negative feature values in 'query'"
      )
    }
    expect_identical(
      brute_force_knn_query(positive, positive, k = 2, metric = "hellinger"),
      expected
    )
    expect_identical(list(positive = positive, negative = negative), originals)
    expect_identical(
      list(R = .Random.seed, dq = dqrng::dqrng_get_state()),
      before
    )
  }
})
