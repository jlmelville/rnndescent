//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2019 James Melville
//
//  This file is part of rnndescent
//
//  rnndescent is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  rnndescent is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with rnndescent.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RNN_NNGRAPH_H
#define RNN_NNGRAPH_H

#include <vector>

struct NNGraph {
  std::vector<int> idx;
  std::vector<double> dist;

  std::size_t n_points;
  std::size_t n_nbrs;

  NNGraph(const std::vector<int> &idx, const std::vector<double> &dist,
          std::size_t n_points)
      : idx(idx), dist(dist), n_points(n_points),
        n_nbrs(idx.size() / n_points) {}
};

#endif // RNN_NNGRAPH_H
