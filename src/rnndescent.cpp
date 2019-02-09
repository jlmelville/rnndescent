#include <bitset>
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

struct Euclidean
{

  Euclidean(const std::vector<double>& data, std::size_t ndim)
    : data(data), ndim(ndim) { }

  double operator()(std::size_t i, std::size_t j) {
    double sum = 0.0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      const double diff = data[di + d] - data[dj + d];
      sum += diff * diff;
    }

    return std::sqrt(sum);
  }

  std::vector<double> data;
  std::size_t ndim;
};

struct Cosine
{

  Cosine(const std::vector<double>& data, std::size_t ndim)
    : data(data), ndim(ndim) {}

  double operator()(std::size_t i, std::size_t j) {
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    normalize(i);
    normalize(j);

    double sum = 0.0;
    for (std::size_t d = 0; d < ndim; d++) {
      sum += data[di + d] * data[dj + d];
    }

    return 1.0 - sum;
  }

  void normalize(std::size_t i) {
    const std::size_t di = ndim * i;
    double norm = 0.0;

    for (std::size_t d = 0; d < ndim; d++) {
      norm += data[di + d] * data[di + d];
    }
    norm = 1.0 / (std::sqrt(norm) + 1e-30);

    for (std::size_t d = 0; d < ndim; d++) {
      data[di + d] *= norm;
    }
  }

  std::vector<double> data;
  std::size_t ndim;
};


struct Manhattan
{

  Manhattan(const std::vector<double>& data, std::size_t ndim)
    : data(data), ndim(ndim) { }

  double operator()(std::size_t i, std::size_t j) {
    double sum = 0.0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      sum += std::abs(data[di + d] - data[dj + d]);
    }

    return sum;
  }

  std::vector<double> data;
  std::size_t ndim;
};

struct Hamming
{
  Hamming(const std::vector<uint8_t>& vdata, std::size_t vndim) {
    // Instead of storing each bit as an element, we will pack them
    // into a series of 64-bit bitsets. Possibly compilers are smart enough
    // to use built in integer popcount routines for the bitset count()
    // method.
    std::bitset<64> bits;
    std::size_t bit_count = 0;
    std::size_t vd_count = 0;

    for (std::size_t i = 0; i < vdata.size(); i++) {
      if (bit_count == 64 || vd_count == vndim) {
        // filled up current bitset
        data.push_back(bits);
        bit_count = 0;
        bits.reset();

        if (vd_count == vndim) {
          // end of item
          vd_count = 0;
        }
      }
      bits[bit_count] = vdata[i];

      ++vd_count;
      ++bit_count;
    }
    if (bit_count > 0) {
      data.push_back(bits);
    }

    ndim = std::ceil(vndim / 64.0);
  }

  std::size_t operator()(std::size_t i, std::size_t j) {
    std::size_t sum = 0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      sum += (data[di + d] ^ data[dj + d]).count();
    }

    return sum;
  }

  std::vector<std::bitset<64>> data;
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



template <typename Distance, typename DistanceType>
Rcpp::List nn_descent_impl(
    Rcpp::NumericMatrix data,
    Rcpp::IntegerMatrix idx,
    Rcpp::NumericMatrix dist,
    const std::size_t max_candidates = 50,
    const std::size_t n_iters = 10,
    const double delta = 0.001,
    const double rho = 0.5,
    bool verbose = false) {
  const std::size_t npoints = idx.nrow();
  const std::size_t nnbrs = idx.ncol();

  const std::size_t ndim = data.ncol();
  data = Rcpp::transpose(data);
  auto data_vec = Rcpp::as<std::vector<DistanceType>>(data);

  // initialize heap structures
  Heap heap(npoints, nnbrs);
  for (std::size_t i = 0; i < npoints; i++) {
    for (std::size_t j = 0; j < nnbrs; j++) {
      heap.push(i, dist(i, j), idx(i, j), true);
      heap.push(idx(i, j), dist(i, j), i, true);
    }
  }

  Distance distance(data_vec, ndim);

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
  if (metric == "euclidean") {
    return nn_descent_impl<Euclidean, double>(data, idx, dist,
                                              max_candidates, n_iters, delta, rho,
                                              verbose);
  }
  else if (metric == "cosine") {
    return nn_descent_impl<Cosine, double>(data, idx, dist,
                                           max_candidates, n_iters, delta, rho,
                                           verbose);
  }
  else if (metric == "manhattan") {
    return nn_descent_impl<Manhattan, double>(data, idx, dist,
                                              max_candidates, n_iters, delta, rho,
                                              verbose);
  }
  else if (metric == "hamming") {
    return nn_descent_impl<Hamming, uint8_t>(data, idx, dist,
                                             max_candidates, n_iters, delta, rho,
                                             verbose);
  }
  else {
    Rcpp::stop("Bad metric");
  }
}
