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

#ifndef RNN_PARALLEL_H
#define RNN_PARALLEL_H

#include "RcppPerpendicular.h"
#include "tdoann/parallel.h"

class RParallelExecutor : public tdoann::Executor {
public:
  void parallel_for(std::size_t begin, std::size_t end,
                    std::function<void(std::size_t, std::size_t)> worker,
                    std::size_t n_threads,
                    std::size_t grain_size = 1) override {
    RcppPerpendicular::parallel_for(begin, end, worker, n_threads, grain_size);
  }
};

#endif // RNN_PARALLEL_H
