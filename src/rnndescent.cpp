#include <Rcpp.h>

struct Heap
{
  std::vector<std::vector<std::size_t>> idx;
  std::vector<std::vector<double>> dist;
  std::vector<std::vector<bool>> flags; // vector of bool, yes ugh
  Heap(const std::size_t n_points, const std::size_t size) {
    for (std::size_t i = 0; i < n_points; i++) {
      for (std::size_t j = 0; j < size; j++) {
        idx.push_back(std::vector<std::size_t>(n_points, -1));
        dist.push_back(std::vector<double>(n_points, std::numeric_limits<double>::max()));
        flags.push_back(std::vector<bool>(n_points, false));
      }
    }
  }

  unsigned int push(std::size_t row, double weight, std::size_t index, bool flag) {
    std::vector<std::size_t>& indices = idx[row];
    std::vector<double>& weights = dist[row];
    std::vector<bool>& is_new = flags[row];

    if (weight >= weights[0]) {
      return 0;
    }

    // break if we already have this element
    const std::size_t n_nbrs = indices.size();
    for (std::size_t i = 0; i < n_nbrs; i++) {
      if (index == indices[i]) {
        return 0;
      }
    }

    // insert val at position zero
    weights[0] = weight;
    indices[0] = index;
    is_new[0] = flag;

    // descend the heap, swapping values until the max heap criterion is met
    std::size_t i = 0;
    std::size_t i_swap = 0;
    while (true) {
        std::size_t ic1 = 2 * i + 1;
        std::size_t ic2 = ic1 + 1;

        if (ic1 >= n_nbrs) {
          break;
        }
        else if (ic2 >= n_nbrs) {
          if (weights[ic1] >= weight) {
            i_swap = ic1;
          }
          else {
            break;
          }
        }
        else if (weights[ic1] >= weights[ic2]) {
          if (weight < weights[ic1]) {
            i_swap = ic1;
          }
          else {
            break;
          }
        }
        else {
          if (weight < weights[ic2]) {
            i_swap = ic2;
          }
          else {
            break;
          }
        }

        weights[i] = weights[i_swap];
        indices[i] = indices[i_swap];
        is_new[i] = is_new[i_swap];

        i = i_swap;
      }

      weights[i] = weight;
      indices[i] = index;
      is_new[i] = flag;

      return 1;
  }

  void deheap_sort() {
    const std::size_t npoints = idx.size();

    for (std::size_t i = 0; i < npoints; i++) {
      std::vector<std::size_t>& ind_heap = idx[i];
      std::vector<double>& dist_heap = dist[i];

      const std::size_t nnbrs = ind_heap.size();
      for (std::size_t j = 0; j < nnbrs - 1; j++) {
        std::swap(ind_heap[0], ind_heap[nnbrs - j - 1]);
        std::swap(dist_heap[0], dist_heap[nnbrs - j - 1]);
        siftdown(dist_heap, ind_heap, nnbrs - j - 1, 0);
      }
    }
  }

  void siftdown(std::vector<double>& dist_heap,
                std::vector<std::size_t>& ind_heap,
                const std::size_t len,
                std::size_t elt) {

    while (elt * 2 + 1 < len) {
        std::size_t left_child = elt * 2 + 1;
        std::size_t right_child = left_child + 1;
        std::size_t swap = elt;

        if (dist_heap[swap] < dist_heap[left_child]) {
          swap = left_child;
        }

        if (right_child < len && dist_heap[swap] < dist_heap[right_child]) {
          swap = right_child;
        }

        if (swap == elt) {
          break;
        }
        else {
          std::swap(dist_heap[elt], dist_heap[swap]);
          std::swap(ind_heap[elt], ind_heap[swap]);
          elt = swap;
        }
    }
  }
};

// [[Rcpp::export]]
Rcpp::List nn_descent_cpp(Rcpp::IntegerMatrix idx, Rcpp::NumericMatrix dist) {
  std::size_t npoints = idx.nrow();
  std::size_t nnbrs = idx.ncol();

  Heap heap(npoints, nnbrs);
  for (std::size_t i = 0; i < npoints; i++) {
    for (std::size_t j = 0; j < nnbrs; j++) {
      heap.push(i, dist(i, j), idx(i, j), true);
      heap.push(idx(i, j), dist(i, j), i, true);
    }
  }
  heap.deheap_sort();

  Rcpp::IntegerMatrix idxres(npoints, nnbrs);
  Rcpp::NumericMatrix distres(npoints, nnbrs);
  for (std::size_t i = 0; i < npoints; i++) {
    for (std::size_t j = 0; j < nnbrs; j++) {
      idxres(i, j) = heap.idx[i][j];
      distres(i, j) = heap.dist[i][j];
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("idx") = idxres,
    Rcpp::Named("dist") = distres
  );
}

