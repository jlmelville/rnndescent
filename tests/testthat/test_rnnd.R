library(rnndescent)
context("rnnd")

set.seed(1337)
iris_index <- rnnd_build(
  data = ui10,
  k = 4,
  diversify_prob = 1.0
)
expect_equal(iris_index$graph, brute_force_knn(ui10, k = 4))

set.seed(1337)
iris_prep <- rnnd_prepare(index = iris_index)
iris_index_prep <-
  rnnd_build(
    data = ui10,
    k = 4,
    diversify_prob = 1.0,
    prepare = TRUE
  )

expect_equal(iris_prep, iris_index_prep)

iris_bf <- brute_force_knn_query(ui10, ui10, k = 4)
iris_query <- rnnd_query(index = iris_index, query = ui10, k = 4)
expect_equal(iris_query, iris_bf)

iris_queryp <-
  rnnd_query(index = iris_index_prep, query = ui10, k = 4)
expect_equal(iris_queryp, iris_bf)

iris_queryp2 <- rnnd_query(index = iris_prep, query = ui10, k = 4)
expect_equal(iris_queryp2, iris_bf)
