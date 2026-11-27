# Run from the repository root via tools/native-quality.sh.
Sys.setenv(
  PKG_CPPFLAGS = paste(
    Sys.getenv("PKG_CPPFLAGS"),
    paste0("-I", shQuote(normalizePath("inst/include")))
  )
)
probe <- new.env(parent = baseenv())
Rcpp::cppFunction(
  code = r"(
void check_progress_increments() {
  std::string failures;
  RPProgress zero(0, false);
  if (!std::isfinite(zero.iter_increment) || zero.iter_increment != 0.0) {
    failures += " zero constructor";
  }
  RPProgress changed(3, false);
  changed.set_n_iters(0);
  if (!std::isfinite(changed.iter_increment) || changed.iter_increment != 0.0) {
    failures += " zero setter";
  }
  zero.set_n_batches(2);
  changed.set_n_batches(2);
  if (zero.batch_increment != 0.0 || changed.batch_increment != 0.0) {
    failures += " zero iteration batches";
  }
  for (uint32_t count : {1U, 3U, 7U, std::numeric_limits<uint32_t>::max()}) {
    const double expected = static_cast<double>(RPProgress::scale) / count;
    RPProgress positive(count, false);
    changed.set_n_iters(count);
    if (positive.iter_increment != expected || changed.iter_increment != expected) {
      failures += " positive count";
    }
  }
  auto stars_at = [](unsigned maximum, unsigned value) {
    std::ostringstream output;
    tdoann::ProgressBar bar(maximum, true, output);
    bar.update(value);
    const auto rendered = output.str();
    return std::count(rendered.begin(), rendered.end(), '*');
  };
  if (stars_at(100, 50) != 26) {
    failures += " halfway star count";
  }
  const auto largest = std::numeric_limits<unsigned>::max();
  if (stars_at(largest, largest) != 51) {
    failures += " full star count at maximum unsigned value";
  }
  if (!failures.empty()) {
    Rcpp::stop("progress increment regression:" + failures);
  }
})",
  # The probe directs the progress bar to a string stream.
  includes = paste(
    "#include <sstream>",
    paste0('#include "', normalizePath("src/rnn_progress.h"), '"'),
    "void ts(const std::string &) {}",
    sep = "\n"
  ),
  plugins = "cpp17",
  env = probe
)
probe$check_progress_increments()
cat("progress increments and rendered star counts: PASS\n")
