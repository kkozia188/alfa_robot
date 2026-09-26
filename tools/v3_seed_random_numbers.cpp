#include <random_numbers/random_numbers.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace {
boost::uint32_t nextSeed()
{
  const char* text = std::getenv("V3_OMPL_SEED");
  const std::uint64_t base = text && *text ? std::strtoull(text, nullptr, 10) : 1U;
  static std::atomic<std::uint32_t> serial{0U};
  return static_cast<boost::uint32_t>(base + serial.fetch_add(1U) * 0x9E3779B9U);
}
}
namespace random_numbers {
RandomNumberGenerator::RandomNumberGenerator() : RandomNumberGenerator(nextSeed()) {}
}
