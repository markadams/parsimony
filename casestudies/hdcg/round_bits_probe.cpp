/* Test harness for round_bits.h: rounds stdin against a format named on the
 * command line and writes the results to stdout.
 *
 * Values cross both ways in C99 hex-float notation ("%a"), which is exact, so
 * a disagreement with parsimony's round_to() is a real disagreement about
 * rounding and never about decimal formatting.  Driven by
 * tests/test_hdcg_round.py; it has no other purpose.
 *
 *   c++ -O2 -o round_bits_probe round_bits_probe.cpp
 *   echo 0x1.5p+3 | ./round_bits_probe v8
 */

#include <cstdio>
#include <cstdlib>

#include "round_bits.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <format>   (v40, fp32, bf16, ...)\n", argv[0]);
    return 2;
  }

  Parsimony::RoundSpec spec;
  if (!Parsimony::parse_round_spec(argv[1], spec)) {
    std::fprintf(stderr, "unrecognized format '%s'\n", argv[1]);
    return 2;
  }

  char line[256];
  while (std::fgets(line, sizeof(line), stdin) != 0) {
    char *end = 0;
    const double x = std::strtod(line, &end);
    if (end == line) continue;  /* blank line */
    std::printf("%a\n", Parsimony::round_bits(x, spec));
  }
  return 0;
}
