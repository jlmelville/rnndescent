# Package index

## Main interface

Find approximate neighbors or build and query an index.

- [`rnnd_knn()`](https://jlmelville.github.io/rnndescent/reference/rnnd_knn.md)
  : Find approximate nearest neighbors
- [`rnnd_build()`](https://jlmelville.github.io/rnndescent/reference/rnnd_build.md)
  : Build approximate nearest neighbors index and neighbor graph
- [`rnnd_query()`](https://jlmelville.github.io/rnndescent/reference/rnnd_query.md)
  : Query an index for approximate nearest neighbors

## Individual search methods

Find or query neighbors using brute force, random selection, nearest
neighbor descent, or random projection forests.

- [`brute_force_knn()`](https://jlmelville.github.io/rnndescent/reference/brute_force_knn.md)
  : Find exact nearest neighbors by brute force
- [`brute_force_knn_query()`](https://jlmelville.github.io/rnndescent/reference/brute_force_knn_query.md)
  : Query exact nearest neighbors by brute force
- [`random_knn()`](https://jlmelville.github.io/rnndescent/reference/random_knn.md)
  : Find nearest neighbors by random selection
- [`random_knn_query()`](https://jlmelville.github.io/rnndescent/reference/random_knn_query.md)
  : Query nearest neighbors by random selection
- [`nnd_knn()`](https://jlmelville.github.io/rnndescent/reference/nnd_knn.md)
  : Find nearest neighbors using nearest neighbor descent
- [`rpf_knn()`](https://jlmelville.github.io/rnndescent/reference/rpf_knn.md)
  : Find nearest neighbors using a random projection forest
- [`rpf_build()`](https://jlmelville.github.io/rnndescent/reference/rpf_build.md)
  : Create a random projection forest nearest neighbor index
- [`rpf_knn_query()`](https://jlmelville.github.io/rnndescent/reference/rpf_knn_query.md)
  : Query a random projection forest index for nearest neighbors
- [`rpf_filter()`](https://jlmelville.github.io/rnndescent/reference/rpf_filter.md)
  : Keep the best trees in a random projection forest

## Graph search and utilities

Prepare and query search graphs, merge neighbor graphs, and assess their
properties.

- [`prepare_search_graph()`](https://jlmelville.github.io/rnndescent/reference/prepare_search_graph.md)
  : Convert a nearest neighbor graph into a search graph
- [`graph_knn_query()`](https://jlmelville.github.io/rnndescent/reference/graph_knn_query.md)
  : Query a search graph for nearest neighbors
- [`merge_knn()`](https://jlmelville.github.io/rnndescent/reference/merge_knn.md)
  : Merge multiple approximate nearest neighbors graphs
- [`k_occur()`](https://jlmelville.github.io/rnndescent/reference/k_occur.md)
  : Quantify hubness of a nearest neighbor graph
- [`neighbor_overlap()`](https://jlmelville.github.io/rnndescent/reference/neighbor_overlap.md)
  : Overlap between the indices of two nearest neighbor graphs
