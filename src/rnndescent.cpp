#include <Rcpp.h>

struct Heap
{
  std::vector<std::vector<int>> idx;
  std::vector<std::vector<double>> dist;
  std::vector<std::vector<bool>> flags; // vector of bool, yes ugh
  Heap(const std::size_t n_points, const std::size_t size) {
    for (std::size_t i = 0; i < n_points; i++) {
      idx.push_back(std::vector<int>(size, -1));
      dist.push_back(std::vector<double>(size, std::numeric_limits<double>::max()));
      flags.push_back(std::vector<bool>(size, false));
    }
  }

  unsigned int push(std::size_t row, double weight, std::size_t index, bool flag) {
    std::vector<int>& indices = idx[row];
    std::vector<double>& weights = dist[row];
    std::vector<bool>& is_new = flags[row];

    if (weight >= weights[0]) {
      return 0;
    }

    // break if we already have this element
    int iindex = static_cast<int>(index);
    const std::size_t n_nbrs = indices.size();
    for (std::size_t i = 0; i < n_nbrs; i++) {
      if (iindex == indices[i]) {
        return 0;
      }
    }

    // insert val at position zero
    weights[0] = weight;
    indices[0] = iindex;
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
      indices[i] = iindex;
      is_new[i] = flag;

      return 1;
  }

  void deheap_sort() {
    const std::size_t npoints = idx.size();

    for (std::size_t i = 0; i < npoints; i++) {
      std::vector<int>& ind_heap = idx[i];
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
                std::vector<int>& ind_heap,
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

struct EuclideanDistance
{
  EuclideanDistance(Rcpp::NumericMatrix data) : data(data), ndim(data.ncol()) {}

  double operator()(std::size_t i, std::size_t j) {
    double sum = 0.0;
    double diff = 0.0;
    for (std::size_t d = 0; d < ndim; d++) {
      diff = data(i, d) - data(j, d);
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }

  Rcpp::NumericMatrix data;
  std::size_t ndim;
};


double runif() {
  return Rcpp::runif(1, 0.0, 1.0)[0];
}

void build_candidates(Heap& current_graph, Heap& candidate_neighbors,
                      const std::size_t npoints, const std::size_t nnbrs) {
  for (std::size_t i = 0; i < npoints; i++) {
    for (std::size_t j = 0; j < nnbrs; j++) {
      if (current_graph.idx[i][j] < 0) {
        continue;
      }
      int idx = current_graph.idx[i][j];
      bool isn = current_graph.flags[i][j];
      double d = runif();

      candidate_neighbors.push(i, d, idx, isn);
      candidate_neighbors.push(static_cast<std::size_t>(idx), d, i, isn);

      current_graph.flags[i][j] = false;
    }
  }
}

// [[Rcpp::export]]
Rcpp::List nn_descent(
    Rcpp::NumericMatrix data,
    Rcpp::IntegerMatrix idx,
    Rcpp::NumericMatrix dist,
    const std::string metric = "euclidean",
    const std::size_t max_candidates = 50,
    const std::size_t n_iters = 10,
    const double delta = 0.001,
    const double rho = 0.5,
    bool verbose = false) {
  std::size_t npoints = idx.nrow();
  std::size_t nnbrs = idx.ncol();

  // initialize heap structures
  Heap heap(npoints, nnbrs);
  for (std::size_t i = 0; i < npoints; i++) {
    for (std::size_t j = 0; j < nnbrs; j++) {
      heap.push(i, dist(i, j), idx(i, j), true);
      heap.push(idx(i, j), dist(i, j), i, true);
    }
  }

  EuclideanDistance distance(data);

  for (std::size_t n = 0; n < n_iters; n++) {
    if (verbose) {
      double sum = 0.0;
      for (std::size_t i = 0; i < npoints; i++) {
        for (std::size_t j = 0; j < nnbrs; j++) {
          sum += heap.dist[i][j];
        }
      }
      Rcpp::Rcout << (n + 1) << " / " << n_iters << " " << sum << std::endl;
    }

    Heap candidate_neighbors(npoints, max_candidates);

    build_candidates(heap, candidate_neighbors, npoints, nnbrs);

    double c = 0.0;
    for (std::size_t i = 0; i < npoints; i++) {
      for (std::size_t j = 0; j < max_candidates; j++) {
        int p = candidate_neighbors.idx[i][j];
        if (p < 0 || runif() < rho) {
          continue;
        }

        for (std::size_t k = 0; k < max_candidates; k++) {
          int q = candidate_neighbors.idx[i][k];
          if (q < 0 || (!candidate_neighbors.flags[i][j] &&
              !candidate_neighbors.flags[i][k])) {
            continue;
          }
          double d = distance(p, q);
          c += heap.push(p, d, q, true);
          c += heap.push(q, d, p, true);
        }
      }
      Rcpp::checkUserInterrupt();
    }
    if (c <= delta * nnbrs * npoints) {
      if (verbose) {
        Rcpp::Rcout << "c = " << c << " crit = " << delta * nnbrs * npoints <<
          std::endl;
      }
      break;
    }
  }

  // sort data
  heap.deheap_sort();

  // transfer data into R Matrices
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

