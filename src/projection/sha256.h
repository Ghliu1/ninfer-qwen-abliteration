#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

namespace ninfer::projection_internal {

using Sha256Digest = std::array<std::uint8_t, 32>;

class Sha256Stream {
public:
    void update(std::span<const std::uint8_t> input);
    [[nodiscard]] Sha256Digest finish() const;

private:
    std::array<std::uint32_t, 8> state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                        0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                        0x1f83d9abU, 0x5be0cd19U};
    std::array<std::uint8_t, 64> pending_{};
    std::size_t pending_bytes_ = 0;
    std::uint64_t total_bytes_ = 0;
};

[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input);
[[nodiscard]] Sha256Digest sha256_file(const std::filesystem::path& path);
[[nodiscard]] std::string sha256_hex(const Sha256Digest& digest);
[[nodiscard]] std::string sha256_hex(std::span<const std::uint8_t> input);

} // namespace ninfer::projection_internal
