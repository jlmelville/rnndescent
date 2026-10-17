empty_query_fixture <- function(
  data,
  obs = "R",
  n_threads = 0,
  metric = "euclidean"
) {
  reference <- if (obs == "C") Matrix::t(data) else data
  empty <- data[FALSE, , drop = FALSE]
  one <- data[1, , drop = FALSE]
  if (obs == "C") {
    empty <- Matrix::t(empty)
    one <- Matrix::t(one)
  }
  forest <- rpf_build(
    reference,
    n_trees = 1,
    leaf_size = nrow(data),
    metric = metric,
    obs = obs
  )
  index <- rnnd_build(
    reference,
    k = 2,
    n_trees = 1,
    leaf_size = nrow(data),
    n_iters = 1,
    metric = metric,
    obs = obs
  )
  graph <- brute_force_knn(reference, k = 2, metric = metric, obs = obs)
  queries <- list(
    brute = function(q) {
      brute_force_knn_query(
        q,
        reference,
        k = 2,
        metric = metric,
        obs = obs,
        n_threads = n_threads
      )
    },
    random = function(q) {
      random_knn_query(
        q,
        reference,
        k = 2,
        metric = metric,
        obs = obs,
        n_threads = n_threads
      )
    },
    forest = function(q) {
      rpf_knn_query(
        q,
        reference,
        forest,
        k = 2,
        obs = obs,
        n_threads = n_threads
      )
    },
    graph = function(q) {
      graph_knn_query(
        q,
        reference,
        graph,
        k = 2,
        metric = metric,
        obs = obs,
        n_threads = n_threads
      )
    },
    index = function(q) {
      rnnd_query(index, q, k = 2, obs = obs, n_threads = n_threads)
    }
  )
  list(
    queries = queries,
    empty = empty,
    one = one,
    reference = reference,
    forest = forest,
    index = index,
    graph = graph
  )
}

empty_neighbors <- function(k) {
  list(idx = matrix(integer(), 0, k), dist = matrix(numeric(), 0, k))
}

test_that("empty queries return typed matrices and allow subsequent ordinary queries", {
  numeric_data <- matrix(c(0, 1, 3, 4, 2, 1, 5, 8), ncol = 2)
  cases <- list(
    list(numeric_data, "R", 0, "euclidean"),
    list(
      methods::as(Matrix::Matrix(numeric_data, sparse = TRUE), "generalMatrix"),
      "C",
      2,
      "euclidean"
    ),
    list(
      matrix(c(FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE), ncol = 2),
      "R",
      0,
      "hamming"
    )
  )
  for (case in cases) {
    set.seed(273)
    fixture <- empty_query_fixture(case[[1]], case[[2]], case[[3]], case[[4]])
    saved <- serialize(
      fixture[c("reference", "forest", "index", "graph")],
      NULL
    )
    for (name in names(fixture$queries)) {
      query <- fixture$queries[[name]]
      expect_identical(query(fixture$empty), empty_neighbors(2), info = name)
      control <- query(fixture$one)
      expect_identical(
        typeof(control$idx),
        if (name %in% c("brute", "random")) "double" else "integer"
      )
      pairs <- as.matrix(case[[1]])[control$idx[1, ], , drop = FALSE]
      first <- as.matrix(case[[1]])[1, ]
      distances <- if (case[[4]] == "hamming") {
        rowMeans(sweep(pairs, 2, first, `!=`))
      } else {
        sqrt(rowSums(sweep(pairs, 2, first)^2))
      }
      expect_equal(control$dist, matrix(distances, nrow = 1), tolerance = 1e-6)
    }
    expect_identical(
      serialize(fixture[c("reference", "forest", "index", "graph")], NULL),
      saved
    )
  }
})

test_that("empty queries retain dataframe and other sparse input support", {
  data <- matrix(c(0, 1, 3, 4, 2, 1, 5, 8), ncol = 2)
  frame <- as.data.frame(data)
  expect_identical(
    brute_force_knn_query(frame[FALSE, ], frame, k = 3),
    empty_neighbors(3)
  )
  sparse <- methods::as(Matrix::Matrix(data, sparse = TRUE), "generalMatrix")
  for (type in c("RsparseMatrix", "TsparseMatrix")) {
    reference <- methods::as(sparse, type)
    query <- methods::as(sparse[FALSE, , drop = FALSE], type)
    expect_identical(
      random_knn_query(query, reference, k = 3),
      empty_neighbors(3)
    )
  }
  expect_identical(
    brute_force_knn_query(t(data[FALSE, ]), t(data), k = 1, obs = "C"),
    empty_neighbors(1)
  )
})

test_that("empty graph and index queries retain existing k resolution", {
  data <- matrix(c(0, 1, 3, 4, 2, 1, 5, 8), ncol = 2)
  fixture <- empty_query_fixture(data)
  expect_identical(
    graph_knn_query(fixture$empty, data, fixture$graph),
    empty_neighbors(2)
  )
  initial <- empty_neighbors(3)
  expect_identical(
    graph_knn_query(fixture$empty, data, fixture$graph, init = initial),
    empty_neighbors(3)
  )
  expect_identical(
    graph_knn_query(fixture$empty, data, fixture$graph, init = initial$idx),
    empty_neighbors(3)
  )
  expect_identical(
    rnnd_query(fixture$index, fixture$empty, k = NULL, init = initial$idx),
    empty_neighbors(3)
  )
})

test_that("empty queries preserve applicable validation", {
  data <- matrix(c(0, 1, 3, 4, 2, 1, 5, 8), ncol = 2)
  fixture <- empty_query_fixture(data)
  for (query in fixture$queries) {
    expect_error(query(matrix(numeric(), 0, 3)), "same number of features")
    expect_error(query(matrix(numeric(), 0, 0)), "at least one feature")
  }
  empty <- fixture$empty
  expect_error(brute_force_knn_query(empty, data, k = 0))
  expect_error(random_knn_query(empty, data, k = 0))
  expect_error(rpf_knn_query(empty, data, fixture$forest, k = 0))
  expect_error(graph_knn_query(empty, data, fixture$graph, k = 0))
  expect_error(rnnd_query(fixture$index, empty, k = 0))
  expect_error(brute_force_knn_query(empty, data[FALSE, ], k = 1))
  expect_error(brute_force_knn_query(empty, data, k = 2, metric = "unknown"))
  expect_error(random_knn_query(empty, data, k = 2, metric = "unknown"))
  expect_error(graph_knn_query(
    empty,
    data,
    fixture$graph,
    k = 2,
    metric = "unknown"
  ))
  expect_error(graph_knn_query(
    empty,
    data,
    fixture$graph,
    init = matrix(integer(), 0, 0)
  ))
  expect_error(graph_knn_query(
    empty,
    data,
    fixture$graph,
    k = 5,
    init = empty_neighbors(2)
  ))
  wrong_forest <- rpf_build(data[1:2, ], n_trees = 1, leaf_size = 2)
  expect_error(rpf_knn_query(empty, data, wrong_forest, k = 2))
  expect_error(graph_knn_query(
    empty,
    data,
    brute_force_knn(data[1:2, ], k = 2),
    k = 2
  ))
  wrong_margin <- fixture$forest
  wrong_margin$margin <- "unknown"
  expect_error(rpf_knn_query(empty, data, wrong_margin, k = 2))
})
