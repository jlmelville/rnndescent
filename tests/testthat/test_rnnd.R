library(rnndescent)
context("rnnd")

set.seed(1337)
iris_index <- rnnd_build(
  data = ui10,
  k = 4,
  diversify_prob = 1.0
)
test_that("Euclidean indexes and queries agree across pruning options", {
  expect_equal(iris_index$graph, brute_force_knn(ui10, k = 4))

  iris_bf <- brute_force_knn_query(ui10, ui10, k = 4)
  msg <- capture_everything(
    iris_query <- rnnd_query(
      index = iris_index,
      query = ui10,
      k = 4,
      verbose = TRUE
    )
  )
  expect_equal(iris_query, iris_bf)
  expect_match(msg, "max distance")

  iris_queryp <-
    rnnd_query(index = iris_index, query = ui10, k = 4)
  expect_equal(iris_queryp, iris_bf)

  set.seed(1337)
  iris_knn <- rnnd_knn(
    data = ui10,
    k = 4,
  )
  expect_equal(iris_knn, iris_index$graph)

  set.seed(1337)
  iris_index_pr <- rnnd_build(
    data = ui10,
    k = 4,
    diversify_prob = 1.0,
    prune_reverse = TRUE
  )
  iris_query_pr <- rnnd_query(index = iris_index_pr, query = ui10, k = 4)
  expect_equal(iris_query_pr, iris_bf)

  set.seed(1337)
  iris_index_w <- rnnd_build(
    data = ui10,
    k = 4,
    diversify_prob = 1.0,
    weight_by_degree = TRUE
  )
  iris_query_w <- rnnd_query(index = iris_index_w, query = ui10, k = 4)
  expect_equal(iris_query_w, iris_bf)
})

test_that("cosine rnnd build/query", {
  metric <- "cosine"
  iris_bf <- brute_force_knn_query(ui10, ui10, k = 4, metric = metric)

  set.seed(1337)
  iris_index <- rnnd_build(
    data = ui10,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  iris_query <- rnnd_query(index = iris_index, query = ui10, k = 4)
  expect_equal(iris_query, iris_bf)

  bit_bf <- brute_force_knn_query(bitdata, bitdata, k = 4, metric = metric)
  set.seed(1337)
  bit_index <- rnnd_build(
    data = bitdata,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  bit_query <- rnnd_query(index = bit_index, query = bitdata, k = 4)
  expect_equal(bit_query, bit_bf)

  bitsp_bf <- brute_force_knn_query(
    bitdatasp,
    bitdatasp,
    k = 4,
    metric = metric
  )
  set.seed(1337)
  bitsp_index <- rnnd_build(
    data = bitdatasp,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  bitsp_query <- rnnd_query(index = bitsp_index, query = bitdatasp, k = 4)
  expect_equal(bitsp_query, bitsp_bf)
})

test_that("dot rnnd build/query", {
  metric <- "dot"
  iris_bf <- brute_force_knn_query(ui10, ui10, k = 4, metric = metric)

  set.seed(1337)
  iris_index <- rnnd_build(
    data = ui10,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  iris_query <- rnnd_query(index = iris_index, query = ui10, k = 4)
  expect_equal(iris_query, iris_bf)

  bit_bf <- brute_force_knn_query(bitdata, bitdata, k = 4, metric = metric)
  set.seed(1337)
  bit_index <- rnnd_build(
    data = bitdata,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  bit_query <- rnnd_query(index = bit_index, query = bitdata, k = 4)
  expect_equal(bit_query, bit_bf)

  bitsp_bf <- brute_force_knn_query(
    bitdatasp,
    bitdatasp,
    k = 4,
    metric = metric
  )
  set.seed(1337)
  bitsp_index <- rnnd_build(
    data = bitdatasp,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  bitsp_query <- rnnd_query(index = bitsp_index, query = bitdatasp, k = 4)
  expect_equal(bitsp_query, bitsp_bf)
})

test_that("trueangular index and forest queries retain angular distances", {
  reference <- rbind(
    c(1, 0),
    c(1, sqrt(3)),
    c(0, 1),
    c(-1, sqrt(3)),
    c(-1, 0),
    c(0, 0)
  )
  query <- rbind(c(1, 0), c(0, 0))
  expected <- rbind(
    c(0, 1 / 3, 1 / 2, 2 / 3, 1, 1 / 2),
    c(1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 0)
  )
  as_sparse <- function(x) methods::as(Matrix::drop0(x), "generalMatrix")
  for (convert in list(identity, as_sparse)) {
    data <- convert(reference)
    set.seed(1337)
    index <- rnnd_build(
      data,
      k = 6,
      metric = "trueangular",
      progress = "dist",
      n_trees = 1,
      leaf_size = 6,
      diversify_prob = 0,
      prune_reverse = FALSE
    )
    index <- unserialize(serialize(index, NULL))
    expect_equal(index$original_metric, "trueangular")
    expect_equal(index$search_forest$actual_metric, "trueangular")
    expect_equal(index$search_forest$original_metric, "trueangular")
    # A single leaf contains every reference, making this tiny search exact.
    results <- list(
      rnnd_query(index, convert(query), k = 6),
      rpf_knn_query(convert(query), data, index$search_forest, k = 6)
    )
    for (result in results) {
      for (i in seq_len(nrow(query))) {
        expect_equal(sort(result$idx[i, ]), 1:6)
        expect_equal(
          result$dist[i, order(result$idx[i, ])],
          expected[i, ],
          tolerance = 1e-6
        )
      }
    }
  }
})

test_that("jaccard rnnd build/query", {
  metric <- "jaccard"

  bit_bf <- brute_force_knn_query(bitdata, bitdata, k = 4, metric = metric)
  set.seed(1337)
  bit_index <- rnnd_build(
    data = bitdata,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  bit_query <- rnnd_query(index = bit_index, query = bitdata, k = 4)
  expect_equal(bit_query, bit_bf)

  set.seed(1337)
  lbit_index <- rnnd_build(
    data = lbitdata,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  lbit_query <- rnnd_query(index = lbit_index, query = lbitdata, k = 4)
  expect_equal(lbit_query, bit_bf)

  bitsp_bf <- brute_force_knn_query(
    bitdatasp,
    bitdatasp,
    k = 4,
    metric = metric
  )
  set.seed(1337)
  bitsp_index <- rnnd_build(
    data = bitdatasp,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  bitsp_query <- rnnd_query(index = bitsp_index, query = bitdatasp, k = 4)
  expect_equal(bitsp_query, bitsp_bf)
})

test_that("hellinger rnnd build/query", {
  metric <- "hellinger"
  iris_bf <- brute_force_knn_query(ui10, ui10, k = 4, metric = metric)

  set.seed(1337)
  iris_index <- rnnd_build(
    data = ui10,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  iris_query <- rnnd_query(index = iris_index, query = ui10, k = 4)
  expect_equal(iris_query, iris_bf)

  bit_bf <- brute_force_knn_query(bitdata, bitdata, k = 4, metric = metric)
  set.seed(1337)
  bit_index <- rnnd_build(
    data = bitdata,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  bit_query <- rnnd_query(index = bit_index, query = bitdata, k = 4)
  expect_equal(bit_query, bit_bf)

  bitsp_bf <- brute_force_knn_query(
    bitdatasp,
    bitdatasp,
    k = 4,
    metric = metric
  )
  set.seed(1337)
  bitsp_index <- rnnd_build(
    data = bitdatasp,
    k = 4,
    diversify_prob = 1.0,
    metric = metric
  )
  bitsp_query <- rnnd_query(index = bitsp_index, query = bitdatasp, k = 4)
  expect_equal(bitsp_query, bitsp_bf)
})

test_that("rnnd obs is normalized and validated", {
  set.seed(1337)
  expected_knn <- rnnd_knn(ui10, k = 4, obs = "R")
  set.seed(1337)
  expect_equal(rnnd_knn(ui10, k = 4, obs = "r"), expected_knn)

  set.seed(1337)
  expected_index <- rnnd_build(ui10, k = 4, obs = "R")
  set.seed(1337)
  lower_index <- rnnd_build(ui10, k = 4, obs = "r")
  expect_equal(lower_index$graph, expected_index$graph)
  expect_equal(lower_index$data, expected_index$data)
  expect_equal(lower_index$search_graph, expected_index$search_graph)

  expected_query <- rnnd_query(
    index = expected_index,
    query = ui10,
    k = 4,
    obs = "R"
  )
  expect_equal(
    rnnd_query(index = expected_index, query = ui10, k = 4, obs = "r"),
    expected_query
  )

  expect_error(rnnd_knn(ui10, k = 4, obs = "rows"), "should be one of")
  expect_error(rnnd_build(ui10, k = 4, obs = "rows"), "should be one of")
  expect_error(
    rnnd_query(index = expected_index, query = ui10, k = 4, obs = "rows"),
    "should be one of"
  )
})

test_that("rnnd search and descent controls are validated", {
  expect_error(
    rnnd_query(index = iris_index, query = ui10, k = 4, epsilon = -0.1),
    "epsilon must be"
  )
  expect_error(
    rnnd_query(
      index = iris_index,
      query = ui10,
      k = 4,
      max_search_fraction = 1.1
    ),
    "max_search_fraction must be"
  )
  expect_error(rnnd_build(ui10, k = 4, delta = 1.1), "delta must be")
  expect_error(rnnd_knn(ui10, k = 4, delta = -0.1), "delta must be")
  expect_error(rnnd_knn(ui10, k = 4, n_iters = 1.5), "n_iters must be")
  expect_error(
    rnnd_build(ui10, k = 4, n_search_trees = 1.5),
    "n_search_trees must be"
  )
})

test_that("rnnd APIs reject invalid n_threads values", {
  expect_error(rnnd_build(ui10, k = 4, n_threads = -1), "n_threads must be")
  expect_error(rnnd_knn(ui10, k = 4, n_threads = 1.5), "n_threads must be")
  expect_error(
    rnnd_query(index = iris_index, query = ui10, k = 4, n_threads = 1.5),
    "n_threads must be"
  )
})

test_that("rnnd exported APIs reject invalid k values", {
  expect_error(rnnd_build(ui10, k = 0), "k must be")
  expect_error(rnnd_knn(ui10, k = 1.5), "k must be")
  expect_error(rnnd_query(index = iris_index, query = ui10, k = 0), "k must be")
})

test_that("graph query verbose search stats report true min and average counts", {
  ref <- matrix(c(0, 0, 1, 1, 2, 2), ncol = 2, byrow = TRUE)
  search_graph <- prepare_search_graph(ref, brute_force_knn(ref, k = 2))
  query <- ref[c(2, 3), , drop = FALSE]
  init <- list(
    idx = matrix(c(1L, 1L), nrow = 2),
    dist = matrix(c(2, 8), nrow = 2)
  )

  msg <- capture_everything(graph_knn_query(
    query = query,
    reference = ref,
    reference_graph = search_graph,
    init = init,
    k = 1,
    epsilon = 0.1,
    max_search_fraction = 1,
    verbose = TRUE
  ))

  expect_match(
    msg,
    "min distance calculation = 1 \\(33\\.33%\\) of reference data"
  )
  expect_match(
    msg,
    "max distance calculation = 2 \\(66\\.67%\\) of reference data"
  )
  expect_match(
    msg,
    "avg distance calculation = 2 \\(50\\.00%\\) of reference data"
  )
})


test_that("integer queries preserve numeric results, RNG and caller data", {
  reference <- matrix(
    c(
      0L,
      1L,
      2L,
      2L,
      0L,
      -1L,
      3L,
      2L,
      1L,
      1L,
      -2L,
      0L,
      -1L,
      0L,
      3L,
      4L,
      1L,
      -2L,
      0L,
      0L,
      0L,
      2L,
      2L,
      2L
    ),
    ncol = 3,
    byrow = TRUE
  )
  dimnames(reference) <- list(letters[1:8], LETTERS[1:3])
  query <- reference[c(2, 5, 7), , drop = FALSE]
  set.seed(707)
  index <- rnnd_build(reference, k = 4, n_trees = 1, leaf_size = 8)
  expect_type(index$data, "integer")

  numeric_index <- index
  storage.mode(numeric_index$data) <- "double"
  numeric_query <- query
  storage.mode(numeric_query) <- "double"
  before <- serialize(list(index = index, query = query), NULL)
  run_query <- function(index, query) {
    set.seed(8107)
    value <- rnnd_query(index, query, k = 4, n_threads = 0)
    list(value = value, rng = .Random.seed)
  }
  expected <- run_query(numeric_index, numeric_query)
  for (input_index in list(index, numeric_index)) {
    for (input_query in list(query, numeric_query)) {
      expect_identical(run_query(input_index, input_query), expected)
    }
  }
  expect_identical(serialize(list(index = index, query = query), NULL), before)
})
