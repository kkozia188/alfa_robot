#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace alfa_robot::motion {
inline uint32_t deterministicOmplSeed(uint32_t base_seed, const std::string& identity,
                                     const std::vector<std::vector<double>>& states) {
  uint32_t hash = 2166136261U;
  const auto mix = [&hash](const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t index = 0; index < size; ++index) { hash ^= bytes[index]; hash *= 16777619U; }
  };
  mix(&base_seed, sizeof(base_seed));
  mix(identity.data(), identity.size());
  for (const auto& state : states) {
    const uint64_t separator = 0x9e3779b97f4a7c15ULL ^ state.size();
    mix(&separator, sizeof(separator));
    for (const double value : state) {
      if (!std::isfinite(value)) throw std::invalid_argument("OMPL seed state must be finite");
      uint64_t bits = 0; std::memcpy(&bits, &value, sizeof(bits)); mix(&bits, sizeof(bits));
    }
  }
  return hash == 0 ? 1U : hash;
}
}  // namespace alfa_robot::motion
