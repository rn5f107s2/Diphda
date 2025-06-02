#pragma once

#include <cstdint>
#include <array>

namespace Chess {
    
namespace Zobrist {

constexpr inline uint64_t xorShift(uint64_t seed) {
    seed ^= (seed << 21);
    seed ^= (seed >> 35);
    seed ^= (seed << 4);
    return seed;
}

template<int N> constexpr inline
std::array<uint64_t, N> generateKeys(uint64_t seed) {
    std::array<uint64_t, N> ret = {};

    for (int i = 0; i < N; i++) {
        seed = xorShift(seed);
        ret[i] = seed;
    }

    return ret;
}

static constexpr std::array<uint64_t, 768> PSQT     = generateKeys<768>(0x5f10752);
static constexpr std::array<uint64_t,   8> EP       = generateKeys<  8>(0xdc7895cf388d110f);
static constexpr std::array<uint64_t,   4> CASTLING = generateKeys<  4>(0xecce799cd3fadf03);
static constexpr uint64_t STM                       = 0xb762913f0b3fad5f;

}

} // namespace Chess 