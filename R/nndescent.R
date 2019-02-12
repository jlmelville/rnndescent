#' @useDynLib rnndescent, .registration = TRUE
#' @importFrom Rcpp sourceCpp
.onUnload <- function(libpath) {
  library.dynam.unload("rnndescent", libpath)
}

# 3 matrices
# 0: candidate indices
# 1: distances
# 2: flag: new element?
make_heap <- function(N, k) {
  heap <- array(dim = c(3, N, k))
  heap[1, , ] <- -1
  heap[2, , ] <- Inf
  heap[3, , ] <- 0

  heap
}

heap_push <- function(heap, row, weight, index, flag) {
  indices <- heap[1, row + 1, ]
  weights <- heap[2, row + 1, ]
  is_new <- heap[3, row + 1, ]

  heap_shape <- dim(heap)

  if (weight >= weights[1]) {
    return(list(success = 0, heap = heap))
  }

  # break if we already have this element.
  for (i in 0:(length(indices) - 1)) {
    if (index == indices[i + 1]) {
      return(list(success = 0, heap = heap))
    }
  }

  # insert val at position one
  weights[1] <- weight
  indices[1] <- index
  is_new[1] <- flag

  # descend the heap, swapping values until the max heap criterion is met
  i <- 0
  while (TRUE) {
    ic1 <- 2 * i + 1
    ic2 <- ic1 + 1

    if (ic1 >= heap_shape[3]) {
      break
    }
    else if (ic2 >= heap_shape[3]) {
      if (weights[ic1 + 1] >= weight) {
        i_swap <- ic1
      }
      else {
        break
      }
    }
    else if (weights[ic1 + 1] >= weights[ic2 + 1]) {
      if (weight < weights[ic1 + 1]) {
        i_swap <- ic1
      }
      else {
        break
      }
    }
    else {
      if (weight < weights[ic2 + 1]) {
        i_swap <- ic2
      }
      else {
        break
      }
    }

    weights[i + 1] = weights[i_swap + 1]
    indices[i + 1] = indices[i_swap + 1]
    is_new[i + 1] = is_new[i_swap + 1]

    i <- i_swap
  }

  weights[i + 1] <- weight
  indices[i + 1] <- index
  is_new[i + 1] <- flag

  heap[1, row + 1, ] <- indices
  heap[2, row + 1, ] <- weights
  heap[3, row + 1, ] <- is_new

  list(
    success = 1,
    heap = heap
  )
}

siftdown <- function(heap1, heap2, elt) {
  N <- length(heap1)
  while (elt * 2 + 1 < N) {
    left_child <- elt * 2 + 1
    right_child <- left_child + 1
    swap <- elt

    if (heap1[swap + 1] < heap1[left_child + 1]) {
      swap <- left_child
    }

    if (right_child < N && heap1[swap + 1] < heap1[right_child + 1]) {
      swap <- right_child
    }

    if (swap == elt) {
      break
    }
    else {
      tmp <- heap1[swap + 1]
      heap1[swap + 1] <- heap1[elt + 1]
      heap1[elt + 1] <- tmp

      tmp <- heap2[swap + 1]
      heap2[swap + 1] <- heap2[elt + 1]
      heap2[elt + 1] <- tmp

      elt <- swap
    }
  }

  list(heap1 = heap1, heap2 = heap2)
}

deheap_sort <- function(heap) {
  indices <- heap[1, , ]
  weights <- heap[2, , ]

  N <- nrow(indices)
  k <- ncol(indices)

  for (i in 0:(N - 1)) {
    ind_heap <- indices[i + 1, ]
    dist_heap <- weights[i + 1, ]
    for (j in 0:(k - 2)) {
      tmp <- ind_heap[1]
      ind_heap[1] <- ind_heap[k - j]
      ind_heap[k - j] <- tmp

      tmp <- dist_heap[1]
      dist_heap[1] <- dist_heap[k - j]
      dist_heap[k - j] <- tmp

      res <- siftdown(
        dist_heap[1:(k - j - 1)],
        ind_heap[1:(k - j - 1)],
        0
      )
      dist_heap[1:(k - j - 1)] <- res$heap1
      ind_heap[1:(k - j - 1)] <- res$heap2
    }
    indices[i + 1, ] <- ind_heap
    weights[i + 1, ] <- dist_heap
  }

  list(
    idx = indices,
    dist = weights
  )
}

l2d <- function(a, b) {
  diff <- a - b
  sum(diff * diff)
}

l2_dist <- function(X, i, j) {
  l2d(X[i, ], X[j, ])
}

eucd <- function(a, b) {
  sqrt(l2d(a, b))
}

euc_dist <- function(X, i, j) {
  eucd(X[i, ], X[j, ])
}

dot <- function(a, b = a) {
  sum(a * b)
}

normv <- function(a) {
  a / sqrt(dot(a))
}

norm2 <- function(a) {
  sqrt(dot(a))
}

coss <- function(a, b) {
  dot(a, b) / sqrt(dot(a) * dot(b))
}

cosd <- function(a, b) {
  1.0 - coss(a, b)
}

cos_dist <- function(X, i, j) {
  cosd(X[i, ], X[j, ])
}

manhattand <- function(a, b) {
  sum(abs(a - b))
}

manhattan_dist <- function(X, i, j) {
  manhattand(X[i, ], X[j, ])
}

hammingd <- function(a, b) {
  sum(bitwXor(a, b))
}

hamming_dist <- function(X, i, j) {
  hammingd(X[i, ], X[j, ])
}

nn_to_heap <- function(indices, dist) {
  N <- nrow(indices)
  k <- ncol(indices)
  current_graph <- make_heap(N, k)
  for (i in 1:N) {
    for (j in 1:k) {
      res <- heap_push(current_graph, i - 1, dist[i, j], indices[i, j], 1)
      current_graph <- res$heap
      res <- heap_push(current_graph, indices[i, j], dist[i, j], i - 1, 1)
      current_graph <- res$heap
    }
  }
  current_graph
}

nn_to_heapl <- function(l) {
  nn_to_heap(l$indices, l$dist)
}

build_candidates <- function(current_graph, n_vertices, n_neighbors,
                             max_candidates = 50) {

  candidate_neighbors <- make_heap(n_vertices, max_candidates)
  for (i in 1:n_vertices) {
    for (j in 1:n_neighbors) {
      if (current_graph[1, i, j] < 0) {
        next
      }
      idx <- current_graph[1, i, j]
      isn <- current_graph[3, i, j]
      d <- stats::runif(1)

      candidate_neighbors <- heap_push(candidate_neighbors, i - 1, d, idx, isn)$heap
      candidate_neighbors <- heap_push(candidate_neighbors, idx, d, i - 1, isn)$heap

      current_graph[3, i, j] <- 0
    }
  }

  list(candidate_neighbors = candidate_neighbors,
       current_graph = current_graph)
}


nn_descent_opt <- function(data, metric, indices, dist, n_iters = 10, max_candidates = 50,
                           delta = 0.001, rho = 0.5,
                           verbose = FALSE) {
  dist_fn <- create_dist_fn(metric)
  n_vertices <- nrow(indices)
  n_neighbors <- ncol(indices)
  current_graph <- nn_to_heap(indices, dist)
  for (iter in 1:n_iters) {
    tsmessage(iter, " / ", n_iters, " ", formatC(sum(current_graph[2, , ])))

    res <- build_candidates(current_graph, n_vertices, n_neighbors,
                            max_candidates)
    candidate_neighbors <- res$candidate_neighbors
    current_graph <- res$current_graph

    c <- 0
    for (i in 1:n_vertices) {
      for (j in 1:max_candidates) {

        p <- candidate_neighbors[1, i, j]

        if (p < 0 || stats::runif(1) < rho) {
          next
        }

        for (k in 1:max_candidates) {
          q <- candidate_neighbors[1, i, k]
          if (q < 0 || candidate_neighbors[3, i, j] == 0 &&
              candidate_neighbors[3, i, k] == 0)
          {
            next
          }
          d <- dist_fn(data, p + 1, q + 1)
          res <- heap_push(current_graph, p, d, q, 1)
          current_graph <- res$heap
          c <- c + res$success

          res <- heap_push(current_graph, q, d, p, 1)
          current_graph <- res$heap
          c <- c + res$success
        }
      }
    }

    if (c <= delta * n_neighbors * n_vertices) {
      tsmessage("c = ", c, " crit = ", delta * n_neighbors * n_vertices)
      break
    }
  }

  deheap_sort(current_graph)
}

nn_descent_optl <- function(data, l, metric = "euclidean", n_iters = 10,
                            max_candidates = 50,
                            delta = 0.001, rho = 0.5,
                            verbose = FALSE) {
  nn_descent_opt(data, metric, l$indices, l$dist,
                 n_iters = n_iters, max_candidates = max_candidates,
                 delta = delta, rho = rho,
                 verbose = verbose)
}

create_dist_fn <- function(metric) {
  switch(metric,
         l2 = l2_dist,
         euclidean = euc_dist,
         cosine = cos_dist,
         manhattan = manhattan_dist,
         hamming = hamming_dist,
         stop("Unknown metric '", metric, "'")
  )
}

det_nbrs <- function(X, k, metric = "euclidean") {
  dist_fn <- create_dist_fn(metric)
  nr = nrow(X)
  indices = matrix(0, nrow = nr, ncol = k)
  dist = matrix(Inf, nrow = nr, ncol = k)

  for (i in 1:nr) {
    for (j in 1:k) {
      idx <- i + j
      if (idx > nr) {
        idx <- idx %% nr
      }
      indices[i, j] <- idx
      dist[i, j] <- dist_fn(X, i, idx)
    }
  }

  list(indices = indices, dist = dist)
}

random_nbrs <- function(X, k, metric = "euclidean") {
  nr = nrow(X)
  indices = matrix(0, nrow = nr, ncol = k)
  dist = matrix(Inf, nrow = nr, ncol = k)
  dist_fn <- create_dist_fn(metric)

  for (i in 1:nr) {
    # we include i as its own neighbor
    # now sample k - 1  from 1:nr, excluding i
    # same as sampling from 1:(nr - 1) and adding one if its >= i
    idxi <- sample.int(nr - 1, k - 1)
    idxi[idxi >= i] <- idxi[idxi >= i] + 1
    indices[i, ] <- c(i, idxi)
    for (j in 2:k) {
      dist[i, j] <- dist_fn(X, i, indices[i, j])
    }
  }
  dist[, 1] <- 0.0
  list(indices = indices, dist = dist)
}


#' Find Nearest Neighbors and Distances
#'
#' @param data Matrix of \code{n} items to search.
#' @param k Number of nearest neighbors to return.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"l2"} (squared Euclidean), \code{"cosine"}, \code{"manhattan"}
#'   or \code{"hamming"}.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item.
#' @param delta precision parameter. Routine will terminate early if
#'   fewer than \eqn{\delta k N}{delta x k x n} updates are made to the nearest
#'   neighbor list in a given iteration.
#' @param rho Sample rate. This fraction of possible items will be used in the
#'   local join stage.
#' @param use_cpp If \code{TRUE}, use the faster C++ code path.
#' @param verbose If \code{TRUE}, log information to the console.
#' @return a list containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the nearest neighbor indices.
#'   \item \code{dist} an n by k matrix containing the nearest neighbor
#'    distances.
#' }
#' @export
nnd_knn <- function(data, k,
                    metric = "euclidean",
                    n_iters = 10,
                    max_candidates = 50,
                    delta = 0.001, rho = 0.5,
                    use_cpp = TRUE,
                    verbose = FALSE) {
  # As a minor optimization, we will use L2 internally if the user asks for
  # Euclidean and only take the square root of the final distances.
  actual_metric <- metric
  if (metric == "euclidean") {
    actual_metric <- "l2"
  }

  tsmessage("Initializing from random neighbors")
  init <- random_nbrs(data, k, metric = actual_metric)
  tsmessage("Init dsum = ", formatC(sum(init$dist)))
  init$indices <- init$indices - 1

  if (use_cpp) {
    res <- nn_descent(data, init$indices, init$dist,
                      metric = actual_metric,
                      n_iters = n_iters, max_candidates = max_candidates,
                      delta = delta, rho = rho, verbose = verbose)
  }
  else {
    res <- nn_descent_optl(data, init, metric = actual_metric, n_iters = n_iters,
                           max_candidates = max_candidates,
                           delta = delta, rho = rho, verbose = verbose)

  }
  if (metric == "euclidean") {
    res$dist <- sqrt(res$dist)
  }
  tsmessage("Final dsum = ", formatC(sum(res$dist)))
  res$idx <- res$idx + 1
  res
}

#' Nearest Neighbor Descent
#'
#' This function uses the nearest neighbor descent method to improve
#' approximate nearest neighbor data.
#'
#' @param data Matrix of \code{n} items to search.
#' @param idx an n by k matrix containing the initial nearest neighbor indices,
#'   where n is the number of items in \code{data} and k is the number of
#'   neighbors.
#' @param dist an n by k matrix containing the initial nearest neighbor
#'   distances, where n is the number of items in \code{data} and k is the
#'   number of neighbors.
#' @param metric Type of distance calculation to use. One of \code{"euclidean"},
#'   \code{"cosine"}, \code{"manhattan"} or \code{"hamming"}.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item.
#' @param delta precision parameter. Routine will terminate early if
#'   fewer than \eqn{\delta k N}{delta x k x n} updates are made to the nearest
#'   neighbor list in a given iteration.
#' @param rho Sample rate. This fraction of possible items will be used in the
#'   local join stage
#' @param verbose If \code{TRUE}, log information to the console.
#' @name nn_descent
#' @return a list containing:
#' \itemize{
#'   \item \code{idx} an n by k matrix containing the improved nearest neighbor
#'    indices.
#'   \item \code{dist} an n by k matrix containing the improved nearest neighbor
#'    distances.
#' }
#' @export
NULL
