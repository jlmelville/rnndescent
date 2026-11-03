// Fixed valid states reach the endpoints without probabilistic sampling.
#include "tdoann/tauprng.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

struct Fixture {
  std::array<uint64_t, 3> seeds;
  int32_t raw;
  double converted;
};

int main() {
  const auto max_raw = (std::numeric_limits<int32_t>::max)();
  const double upper = std::nextafter(1.0, 0.0);
  const std::array<Fixture, 5> fixtures{{
      {{2848216586ULL, 591734665ULL, 878082202ULL},
       (std::numeric_limits<int32_t>::min)(),
       upper},
      {{2848696756ULL, 591751049ULL, 878082202ULL}, max_raw, upper},
      {{2848216650ULL, 591751049ULL, 878082202ULL}, -max_raw, upper},
      {{2848740938ULL, 591751049ULL, 878082202ULL}, 0, 0.0},
      {{3487310774ULL, 591734665ULL, 878082202ULL},
       123456789,
       0.05748904731939037},
  }};
  int failures = 0;
  for (const auto &fixture : fixtures) {
    const auto &seeds = fixture.seeds;
    tdoann::tau_prng integers(seeds[0], seeds[1], seeds[2]);
    tdoann::tau_prng converted(seeds[0], seeds[1], seeds[2]);
    const auto raw = integers();
    const double value = converted.rand();
    if (raw != fixture.raw || value != fixture.converted || value < 0.0 ||
        value >= 1.0) {
      std::cerr << std::setprecision(17)
                << "tau_prng conversion mismatch: raw=" << raw
                << " value=" << value << " expected=" << fixture.converted
                << '\n';
      ++failures;
    }
    for (int draw = 0; draw < 8; ++draw) {
      if (integers() != converted()) {
        std::cerr << "tau_prng changed subsequent state/draw count\n";
        ++failures;
        break;
      }
    }
  }
  if (failures != 0) {
    return 1;
  }
  std::cout
      << "tau_prng endpoints: PASS (five states and subsequent draw parity)\n";
}
