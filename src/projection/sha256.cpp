#include "projection/sha256.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::projection_internal {
namespace {

std::uint32_t load_be32(const std::uint8_t* bytes) {
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) |
           static_cast<std::uint32_t>(bytes[3]);
}

void store_be32(std::uint32_t value, std::uint8_t* bytes) {
    bytes[0] = static_cast<std::uint8_t>(value >> 24U);
    bytes[1] = static_cast<std::uint8_t>(value >> 16U);
    bytes[2] = static_cast<std::uint8_t>(value >> 8U);
    bytes[3] = static_cast<std::uint8_t>(value);
}

void sha256_block(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) {
    constexpr std::array<std::uint32_t, 64> round{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
        0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
        0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
        0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
        words[index] = load_be32(block + 4 * index);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
        const std::uint32_t s0 = std::rotr(words[index - 15], 7) ^
                                 std::rotr(words[index - 15], 18) ^
                                 (words[index - 15] >> 3U);
        const std::uint32_t s1 = std::rotr(words[index - 2], 17) ^
                                 std::rotr(words[index - 2], 19) ^
                                 (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
        const std::uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const std::uint32_t choose = (e & f) ^ (~e & g);
        const std::uint32_t t1 = h + sum1 + choose + round[index] + words[index];
        const std::uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

} // namespace

void Sha256Stream::update(std::span<const std::uint8_t> input) {
    if (input.size() > std::numeric_limits<std::uint64_t>::max() - total_bytes_) {
        throw std::invalid_argument("payload is too large to hash");
    }
    total_bytes_ += static_cast<std::uint64_t>(input.size());
    std::size_t offset = 0;
    if (pending_bytes_ != 0) {
        const std::size_t take = std::min(input.size(), pending_.size() - pending_bytes_);
        std::copy_n(input.data(), take, pending_.data() + pending_bytes_);
        pending_bytes_ += take;
        offset += take;
        if (pending_bytes_ == pending_.size()) {
            sha256_block(state_, pending_.data());
            pending_bytes_ = 0;
        }
    }
    while (input.size() - offset >= 64) {
        sha256_block(state_, input.data() + offset);
        offset += 64;
    }
    const std::size_t remaining = input.size() - offset;
    if (remaining != 0) {
        std::copy_n(input.data() + offset, remaining, pending_.data());
        pending_bytes_ = remaining;
    }
}

Sha256Digest Sha256Stream::finish() const {
    std::array<std::uint32_t, 8> state = state_;
    std::array<std::uint8_t, 128> tail{};
    std::copy_n(pending_.data(), pending_bytes_, tail.data());
    tail.at(pending_bytes_) = 0x80U;
    const std::size_t tail_bytes = pending_bytes_ < 56 ? 64 : 128;
    const std::uint64_t bits = total_bytes_ * 8ULL;
    for (std::size_t index = 0; index < 8; ++index) {
        tail[tail_bytes - 1 - index] = static_cast<std::uint8_t>(bits >> (8U * index));
    }
    sha256_block(state, tail.data());
    if (tail_bytes == 128) { sha256_block(state, tail.data() + 64); }
    Sha256Digest result{};
    for (std::size_t index = 0; index < state.size(); ++index) {
        store_be32(state[index], result.data() + 4 * index);
    }
    return result;
}

Sha256Digest sha256(std::span<const std::uint8_t> input) {
    Sha256Stream accumulator;
    accumulator.update(input);
    return accumulator.finish();
}

Sha256Digest sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("could not open file for SHA-256"); }
    std::vector<std::uint8_t> buffer(8ULL << 20);
    Sha256Stream accumulator;
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            accumulator.update(std::span<const std::uint8_t>(
                buffer.data(), static_cast<std::size_t>(count)));
        }
    }
    if (!input.eof()) { throw std::runtime_error("could not read complete file for SHA-256"); }
    return accumulator.finish();
}

std::string sha256_hex(const Sha256Digest& digest) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '\0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
        result[2 * index] = hex[digest[index] >> 4U];
        result[2 * index + 1] = hex[digest[index] & 0x0fU];
    }
    return result;
}

std::string sha256_hex(std::span<const std::uint8_t> input) {
    return sha256_hex(sha256(input));
}

} // namespace ninfer::projection_internal
