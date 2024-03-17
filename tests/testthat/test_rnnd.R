library(rnndescent)
context("rnnd")

set.seed(1337)
iris_index <- rnnd_build(
  data = ui10,
  k = 4,
  diversify_prob = 1.0
)
expect_equal(iris_index$graph, brute_force_knn(ui10, k = 4))

iris_bf <- brute_force_knn_query(ui10, ui10, k = 4)
msg <- capture_everything(iris_query <- rnnd_query(index = iris_index, query = ui10, k = 4, verbose = TRUE))
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
expect_equal(iris_query_pr, iris_bf)

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

  bitsp_bf <- brute_force_knn_query(bitdatasp, bitdatasp, k = 4, metric = metric)
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

  bitsp_bf <- brute_force_knn_query(bitdatasp, bitdatasp, k = 4, metric = metric)
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

test_that("trueangular rnnd build/query", {
  metric <- "trueangular"
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

  bitsp_bf <- brute_force_knn_query(bitdatasp, bitdatasp, k = 4, metric = metric)
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

  bitsp_bf <- brute_force_knn_query(bitdatasp, bitdatasp, k = 4, metric = metric)
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

  bitsp_bf <- brute_force_knn_query(bitdatasp, bitdatasp, k = 4, metric = metric)
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
