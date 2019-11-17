#ifndef TDOANN_PROGRESS_H
#define TDOANN_PROGRESS_H

#define TDOANN_BREAKIFINTERRUPTED()                                            \
  if (progress.check_interrupt()) {                                            \
    break;                                                                     \
  }

namespace tdoann {
// Defines the methods required, but does nothing. Safe to use from
// multi-threaded code if a dummy no-op version is needed.
struct NullProgress {
  NullProgress() {}
  NullProgress(std::size_t n_iters, bool verbose) {}
  void block_finished() {}
  void iter_finished() {}
  void stopping_early() {}
  bool check_interrupt() { return false; }
};
} // namespace tdoann

#endif // TDOANN_PROGRESS_H
