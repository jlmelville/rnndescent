#ifndef NND_PROGRESS_H
#define NND_PROGRESS_H

// Defines the methods required, but does nothing. Safe to use from
// multi-threaded code if a dummy no-op version is needed.
struct NullProgress {
  NullProgress(){};
  NullProgress(std::size_t n_iters, bool verbose){};
  void increment(std::size_t amount = 1) {}
  void update(std::size_t current){};
  void stopping_early(){};
  bool check_interrupt() { return false; };
};

#endif // NND_PROGRESS_H
