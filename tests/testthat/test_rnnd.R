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
