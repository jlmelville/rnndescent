// Raw-vector empty construction is latent behind guarded public R paths.
#include "tdoann/nngraph.h"

#include <cassert>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

void empty_copy() {
  const std::vector<uint32_t> idx;
  const std::vector<float> dist;
  const tdoann::NNGraph<float> graph(idx, dist, 0);
  assert(graph.n_points == 0 && graph.n_nbrs == 0);
  assert(graph.idx.empty() && graph.dist.empty());
}

void empty_move() {
  std::vector<uint32_t> idx;
  std::vector<float> dist;
  const tdoann::NNGraph<float> graph(std::move(idx), std::move(dist), 0);
  assert(graph.n_points == 0 && graph.n_nbrs == 0);
  assert(graph.idx.empty() && graph.dist.empty());
}

void controls() {
  const std::vector<uint32_t> idx{1, 0, 0, 1};
  const std::vector<float> dist{0.5f, 0, 0.5f, 0};
  const tdoann::NNGraph<float> copied(idx, dist, 2);
  auto move_idx = idx;
  auto move_dist = dist;
  const tdoann::NNGraph<float> moved(std::move(move_idx), std::move(move_dist),
                                     2);
  for (const auto *graph : {&copied, &moved}) {
    assert(graph->n_points == 2 && graph->n_nbrs == 2);
    assert(graph->idx == idx && graph->dist == dist);
  }
  const tdoann::NNGraph<float> shaped(0, 7);
  assert(shaped.n_points == 0 && shaped.n_nbrs == 7);
  assert(shaped.idx.empty() && shaped.dist.empty());
  tdoann::NNHeap<float> empty_heap(0, 7);
  const auto converted = tdoann::heap_to_graph(std::move(empty_heap));
  assert(converted.n_points == 0 && converted.n_nbrs == 7);
  assert(converted.idx.empty() && converted.dist.empty());
}

int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "all";
  if (mode == "all" || mode == "copy") {
    empty_copy();
  }
  if (mode == "all" || mode == "move") {
    empty_move();
  }
  if (mode == "all" || mode == "controls") {
    controls();
  }
  std::cout << "nngraph construction: PASS (" << mode << ")\n";
}
