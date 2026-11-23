library(rnndescent)

test_that("numeric and packed logical Hamming descent return the expected neighbors", {
  for (input in list(bitdata, lbitdata)) {
    set.seed(1337)
    result <- nnd_knn(
      input,
      4,
      metric = "hamming",
      max_candidates = 10,
      low_memory = is.logical(input),
      n_threads = 1L
    )
    expect_equal(result$idx, expected_hamm_idx, ignore_attr = "dimnames")
    expect_equal(
      result$dist,
      expected_hamm_dist,
      ignore_attr = "dimnames",
      tolerance = 1e-7
    )
  }
})

test_that("logical Hamming descent recalculates and resizes initialization", {
  # logical data and distance recalculation
  lbit_rnn <- nnd_knn(
    lbitdata,
    k = 4,
    metric = "hamming",
    init = random_knn(bitdata, k = 1, metric = "hamming")$idx,
    max_candidates = 10
  )
  expect_equal(lbit_rnn$idx, expected_hamm_idx, ignore_attr = "dimnames")
  expect_equal(
    lbit_rnn$dist,
    expected_hamm_dist,
    ignore_attr = "dimnames",
    tolerance = 1e-7
  )

  # test using an excessive number of initial indices
  lbit_rnn <- nnd_knn(
    lbitdata,
    4,
    metric = "hamming",
    n_threads = 1,
    init = random_knn(lbitdata, k = 5, metric = "hamming")$idx,
    max_candidates = 10,
    low_memory = FALSE
  )
  expect_equal(lbit_rnn$idx, expected_hamm_idx, ignore_attr = "dimnames")
  expect_equal(
    lbit_rnn$dist,
    expected_hamm_dist,
    ignore_attr = "dimnames",
    tolerance = 1e-7
  )
})

test_that("Hamming queries score numeric and packed logical neighbors", {
  for (input in list(bitdata, lbitdata)) {
    reference <- input[1:6, ]
    query <- input[7:10, ]
    set.seed(1337)
    graph <- nnd_knn(reference, k = 4, metric = "hamming")
    result <- graph_knn_query(
      query,
      reference,
      graph,
      k = 4,
      metric = "hamming"
    )
    check_query_nbrs(result, query, 1:6, 7:10, 4, bit10_hamd, tolerance = 1e-6)
    expect_equal(sum(result$dist), bit4q_hdsum, tolerance = 1e-6)

    graph$dist[6, 4] <- 0
    prepared <- prepare_search_graph(
      reference,
      graph,
      metric = "hamming",
      n_threads = 1
    )
    result <- graph_knn_query(
      query,
      reference,
      prepared,
      k = 4,
      metric = "hamming",
      n_threads = 1
    )
    check_query_nbrs(result, query, 1:6, 7:10, 4, bit10_hamd, tolerance = 1e-6)
  }
})

test_that("binary tree descent preserves the serial symmetric import and forest", {
  set.seed(818)
  seed <- rpf_knn(
    lbitdata,
    4,
    metric = "hamming",
    n_trees = 2,
    leaf_size = 4,
    include_self = FALSE,
    ret_forest = TRUE
  )
  expected <- nnd_knn(
    lbitdata,
    k = 4,
    metric = "hamming",
    init = seed,
    ret_forest = TRUE
  )
  set.seed(818)
  actual <- nnd_knn(
    lbitdata,
    k = 4,
    metric = "hamming",
    init = "tree",
    init_args = list(n_trees = 2, leaf_size = 4),
    ret_forest = TRUE
  )
  expect_identical(actual, expected)

  # Tree initialization must use the descent metric, including in a saved forest.
  set.seed(818)
  conflicting <- nnd_knn(
    lbitdata,
    k = 4,
    metric = "hamming",
    init = "tree",
    init_args = list(
      n_trees = 2,
      leaf_size = 4,
      metric = "euclidean",
      use_alt_metric = FALSE
    ),
    ret_forest = TRUE
  )
  expect_identical(conflicting, expected)
})

test_that("default logical Jaccard builds hand the leaf heap to descent", {
  native <- rnn_binary_tree_descent
  scores <- character()
  local_mocked_bindings(
    rnn_binary_tree_descent = function(..., metric) {
      scores <<- c(scores, metric)
      native(..., metric = metric)
    },
    .package = "rnndescent"
  )
  rnnd_knn(lbitdata, k = 4, metric = "jaccard")
  index <- rnnd_build(lbitdata, k = 4, metric = "jaccard")
  plain <- rnnd_build(
    lbitdata,
    k = 4,
    metric = "jaccard",
    use_alt_metric = FALSE
  )
  expect_identical(scores, rep("jaccard", 3L))
  expect_identical(index$search_forest$actual_metric, "alternative-jaccard")
  expect_identical(plain$search_forest$actual_metric, "jaccard")
})
