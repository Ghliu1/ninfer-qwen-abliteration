#include "projection/sha256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>

int main() {
    struct KnownAnswer {
        std::size_t bytes;
        const char* digest;
    };
    constexpr std::array<KnownAnswer, 4> known_answers{{
        {55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
        {56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
        {63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"},
        {64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
    }};

    for (const KnownAnswer& answer : known_answers) {
        const std::string input(answer.bytes, 'a');
        const auto bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(input.data()), input.size());
        const std::string actual = ninfer::projection_internal::sha256_hex(bytes);
        if (actual != answer.digest) {
            std::cerr << "SHA-256 known-answer mismatch at " << answer.bytes << " bytes: "
                      << actual << '\n';
            return 1;
        }
    }
    return 0;
}
