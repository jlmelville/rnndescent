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

    weights[i + 1] <- weights[i_swap + 1]
    indices[i + 1] <- indices[i_swap + 1]
    is_new[i + 1] <- is_new[i_swap + 1]

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
