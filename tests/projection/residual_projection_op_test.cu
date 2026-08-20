#include "ninfer/ops/linear_add.h"
#include "ninfer/projection/residual_projection.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr std::int32_t kOutputRows = 5120;
constexpr std::int32_t kSparseWeightColumn = 0;
constexpr std::array<std::int32_t, 6> kTokenCases{1, 16, 22, 25, 256, 1024};
constexpr std::int32_t kNvfp4SmallTToken = 2;
constexpr std::size_t kMinimumOrderingSentinelsPerToken = 16;
constexpr float kOrderingSignatureValue = 32.0F;
constexpr float kMaxAbsoluteError = 0.03125F;
constexpr double kMinimumCosine = 0.9999;

struct RouteFamily {
    const char* label;
    QType qtype;
    std::int32_t input_rows;
    ops::LinearPolicy policy;
    std::int32_t first_quantized_token;
    std::uint32_t seed;
};

constexpr std::array<RouteFamily, 3> kRouteFamilies{
    RouteFamily{"FP8 attention", QType::FP8_E4M3FN_ROW_BF16S, 6144,
                ops::LinearPolicy::AllowA8, 22, 1201U},
    RouteFamily{"FP8 MLP", QType::FP8_E4M3FN_ROW_BF16S, 17408,
                ops::LinearPolicy::AllowA8, 25, 1207U},
    RouteFamily{"NVFP4 MLP", QType::NVFP4, 17408, ops::LinearPolicy::AllowA4, 8, 1213U},
};

double decode_e4m3fn(std::uint8_t code) {
    const bool negative = (code & 0x80U) != 0;
    const std::uint32_t exponent = (code >> 3) & 0x0fU;
    const std::uint32_t mantissa = code & 0x07U;
    double magnitude = 0.0;
    if (exponent == 0) {
        magnitude = static_cast<double>(mantissa) * std::ldexp(1.0, -9);
    } else {
        magnitude = (1.0 + static_cast<double>(mantissa) / 8.0) *
                    std::ldexp(1.0, static_cast<int>(exponent) - 7);
    }
    return negative ? -magnitude : magnitude;
}

std::uint8_t encode_e4m3fn(float value) {
    std::uint8_t best = 0;
    double best_error = std::numeric_limits<double>::infinity();
    for (std::uint32_t candidate = 0; candidate < 256; ++candidate) {
        if (candidate == 0x7fU || candidate == 0xffU) { continue; }
        const double error = std::abs(static_cast<double>(value) -
                                      decode_e4m3fn(static_cast<std::uint8_t>(candidate)));
        if (error < best_error ||
            (error == best_error && (candidate & 1U) == 0U && (best & 1U) != 0U)) {
            best = static_cast<std::uint8_t>(candidate);
            best_error = error;
        }
    }
    return best;
}

double decode_e2m1(std::uint8_t code) {
    constexpr std::array<double, 8> magnitudes{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double magnitude = magnitudes[code & 0x07U];
    return (code & 0x08U) == 0 ? magnitude : -magnitude;
}

std::uint8_t encode_e2m1(float value) {
    std::uint8_t best = 0;
    double best_error = std::numeric_limits<double>::infinity();
    for (std::uint32_t candidate = 0; candidate < 16; ++candidate) {
        const double error =
            std::abs(static_cast<double>(value) - decode_e2m1(static_cast<std::uint8_t>(candidate)));
        if (error < best_error ||
            (error == best_error && (candidate & 1U) == 0U && (best & 1U) != 0U)) {
            best = static_cast<std::uint8_t>(candidate);
            best_error = error;
        }
    }
    return best;
}

std::vector<std::uint16_t> make_activation(std::int32_t rows, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::uint32_t coordinate = static_cast<std::uint32_t>(row) * 17U +
                                             static_cast<std::uint32_t>(token) * 29U + seed;
            float value = static_cast<float>(static_cast<int>(coordinate % 97U) - 48) / 320.0F;
            if (row == rows - 1) { value = 1.0F + static_cast<float>(token % 5) / 8.0F; }
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(value);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_residual(std::int32_t tokens, std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kOutputRows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < kOutputRows; ++row) {
            const std::uint32_t coordinate = static_cast<std::uint32_t>(row) * 23U +
                                             static_cast<std::uint32_t>(token) * 41U + seed;
            const float value = static_cast<float>(static_cast<int>(coordinate % 257U) - 128) /
                                64.0F;
            result[static_cast<std::size_t>(token) * kOutputRows + row] = f32_to_bf16(value);
        }
    }
    return result;
}

std::vector<float> make_signature(std::int32_t input_rows) {
    std::vector<float> signature(input_rows, 0.0F);
    for (std::int32_t column = 0; column < 16; ++column) {
        signature[column] = (column & 1) == 0 ? 32.0F : -16.0F;
    }
    return signature;
}

std::vector<float> make_ordering_signature(std::int32_t input_rows) {
    std::vector<float> signature(input_rows, 0.0F);
    signature.back() = kOrderingSignatureValue;
    return signature;
}

std::vector<float> make_direction() {
    const float value = 1.0F / std::sqrt(static_cast<float>(kOutputRows));
    return std::vector<float>(kOutputRows, value);
}

float sparse_weight_value(std::int32_t row) {
    constexpr std::array<float, 6> kValues{0.5F, 1.0F, 1.5F, -0.5F, -1.0F, -1.5F};
    return kValues[static_cast<std::size_t>(row) % kValues.size()];
}

float route_score(const RouteFamily& family, std::span<const std::uint16_t> activation,
                  std::span<const float> signature, std::int32_t token, std::int32_t tokens) {
    const auto* x = activation.data() + static_cast<std::size_t>(token) * family.input_rows;
    const bool quantized = tokens >= family.first_quantized_token;
    float score = 0.0F;
    if (!quantized) {
        for (std::int32_t column = 0; column < family.input_rows; ++column) {
            score += signature[column] * bf16_to_f32(x[column]);
        }
        return score;
    }

    if (family.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        float maximum = 0.0F;
        for (std::int32_t column = 0; column < family.input_rows; ++column) {
            maximum = std::max(maximum, std::abs(bf16_to_f32(x[column])));
        }
        const float scale = maximum > 0.0F ? maximum / 448.0F : 0.0F;
        for (std::int32_t column = 0; column < family.input_rows; ++column) {
            const float effective = scale == 0.0F
                                        ? 0.0F
                                        : static_cast<float>(decode_e4m3fn(
                                              encode_e4m3fn(bf16_to_f32(x[column]) / scale))) *
                                              scale;
            score += signature[column] * effective;
        }
        return score;
    }

    constexpr std::int32_t kGroup = 16;
    constexpr float kInputScaleDivisor = 3.5F;
    for (std::int32_t group = 0; group < family.input_rows; group += kGroup) {
        float maximum = 0.0F;
        for (std::int32_t lane = 0; lane < kGroup; ++lane) {
            maximum = std::max(maximum, std::abs(bf16_to_f32(x[group + lane])));
        }
        const std::uint8_t scale_code = encode_e4m3fn(kInputScaleDivisor * maximum / 6.0F);
        const float scale = static_cast<float>(decode_e4m3fn(scale_code));
        if (scale == 0.0F) { continue; }
        for (std::int32_t lane = 0; lane < kGroup; ++lane) {
            const std::int32_t column = group + lane;
            const std::uint8_t code =
                encode_e2m1(bf16_to_f32(x[column]) * kInputScaleDivisor / scale);
            const float effective =
                static_cast<float>(decode_e2m1(code)) * scale / kInputScaleDivisor;
            score += signature[column] * effective;
        }
    }
    return score;
}

float nvfp4_ordering_score(const RouteFamily& family,
                           std::span<const std::uint16_t> activation, std::int32_t token) {
    constexpr std::int32_t kGroup = 16;
    constexpr float kInputScaleDivisor = 3.5F;
    const auto* x = activation.data() + static_cast<std::size_t>(token) * family.input_rows;
    const std::int32_t group = family.input_rows - kGroup;
    float maximum = 0.0F;
    for (std::int32_t lane = 0; lane < kGroup; ++lane) {
        maximum = std::max(maximum, std::abs(bf16_to_f32(x[group + lane])));
    }
    const std::uint8_t scale_code = encode_e4m3fn(kInputScaleDivisor * maximum / 6.0F);
    const float scale = static_cast<float>(decode_e4m3fn(scale_code));
    if (scale == 0.0F) { return 0.0F; }
    const float output_scale = scale / kInputScaleDivisor;
    const std::uint8_t code = encode_e2m1(
        bf16_to_f32(x[family.input_rows - 1]) * kInputScaleDivisor / scale);
    const float decoded = static_cast<float>(decode_e2m1(code));
    return (kOrderingSignatureValue * decoded) * output_scale;
}

struct SparseActivationProfile {
    float logical_value;
    float emitted_code;
    float emitted_scale;
    float output_scale;
};

SparseActivationProfile route_activation_at_sparse_column(
    const RouteFamily& family, std::span<const std::uint16_t> activation, std::int32_t token,
    std::int32_t tokens) {
    const auto* x = activation.data() + static_cast<std::size_t>(token) * family.input_rows;
    if (tokens < family.first_quantized_token) {
        const float value = bf16_to_f32(x[kSparseWeightColumn]);
        return SparseActivationProfile{value, value, 1.0F, 1.0F};
    }

    if (family.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        float maximum = 0.0F;
        for (std::int32_t column = 0; column < family.input_rows; ++column) {
            maximum = std::max(maximum, std::abs(bf16_to_f32(x[column])));
        }
        if (maximum == 0.0F) { return SparseActivationProfile{}; }
        const float scale = maximum / 448.0F;
        const float code = static_cast<float>(decode_e4m3fn(
            encode_e4m3fn(bf16_to_f32(x[kSparseWeightColumn]) / scale)));
        return SparseActivationProfile{code * scale, code, scale, 1.0F};
    }

    constexpr float kInputScaleDivisor = 3.5F;
    float maximum = 0.0F;
    for (std::int32_t lane = 0; lane < 16; ++lane) {
        maximum = std::max(maximum, std::abs(bf16_to_f32(x[lane])));
    }
    const std::uint8_t scale_code = encode_e4m3fn(kInputScaleDivisor * maximum / 6.0F);
    const float scale = static_cast<float>(decode_e4m3fn(scale_code));
    if (scale == 0.0F) { return SparseActivationProfile{}; }
    const std::uint8_t code =
        encode_e2m1(bf16_to_f32(x[kSparseWeightColumn]) * kInputScaleDivisor / scale);
    const float decoded = static_cast<float>(decode_e2m1(code));
    return SparseActivationProfile{decoded * scale / kInputScaleDivisor, decoded, scale,
                                   1.0F / kInputScaleDivisor};
}

quantized_weight::PackedWeight make_sparse_projection_weight(const RouteFamily& family) {
    quantized_weight::PatternedWeightOptions options;
    if (family.qtype == QType::NVFP4) {
        options.weight_scale_divisor = 1.0F;
        options.input_scale_divisor = 3.5F;
    }
    auto weight = quantized_weight::make_patterned_weight(
        family.qtype, kOutputRows, family.input_rows, family.seed, options);
    std::fill_n(weight.payload.begin(), static_cast<std::size_t>(weight.code_plane_bytes),
                static_cast<std::uint8_t>(0));
    std::fill_n(weight.payload.begin() + static_cast<std::size_t>(weight.scale_plane_offset),
                static_cast<std::size_t>(weight.scale_plane_bytes), static_cast<std::uint8_t>(0));
    if (family.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        constexpr std::array<std::uint8_t, 6> kCodes{0x30U, 0x38U, 0x3cU,
                                                      0xb0U, 0xb8U, 0xbcU};
        const std::uint16_t scale = f32_to_bf16(1.0F);
        for (std::int32_t row = 0; row < kOutputRows; ++row) {
            weight.payload[static_cast<std::size_t>(row) * family.input_rows +
                           kSparseWeightColumn] =
                kCodes[static_cast<std::size_t>(row) % kCodes.size()];
            const std::size_t offset =
                weight.scale_plane_offset + static_cast<std::size_t>(row) * sizeof(scale);
            weight.payload[offset] = static_cast<std::uint8_t>(scale & 0xffU);
            weight.payload[offset + 1] = static_cast<std::uint8_t>(scale >> 8);
        }
    } else {
        constexpr std::array<std::uint8_t, 6> kCodes{0x01U, 0x02U, 0x03U,
                                                      0x09U, 0x0aU, 0x0bU};
        constexpr std::uint8_t kUnitE4m3 = 0x38U;
        const std::int32_t k_tiles = family.input_rows / 64;
        for (std::int32_t row = 0; row < kOutputRows; ++row) {
            weight.payload[static_cast<std::size_t>(row) * family.input_rows / 2] =
                kCodes[static_cast<std::size_t>(row) % kCodes.size()];
            const std::int32_t row_tile = row / 128;
            const std::int32_t row_inner = row % 128;
            const std::size_t scale_offset =
                weight.scale_plane_offset +
                static_cast<std::size_t>(row_tile * k_tiles) * 512U +
                static_cast<std::size_t>(row_inner % 32) * 16U +
                static_cast<std::size_t>(row_inner / 32) * 4U;
            weight.payload[scale_offset] = kUnitE4m3;
        }
    }
    return weight;
}

quantized_weight::PackedWeight make_zero_weight(const RouteFamily& family) {
    auto weight = make_sparse_projection_weight(family);
    std::fill_n(weight.payload.begin(), static_cast<std::size_t>(weight.code_plane_bytes),
                static_cast<std::uint8_t>(0));
    return weight;
}

int verify_projection(const RouteFamily& family, std::int32_t tokens,
                      std::span<const std::uint16_t> activation,
                      std::span<const std::uint16_t> initial_residual,
                      std::span<const float> signature, std::span<const float> direction,
                      const Weight& weight, GuardedDeviceBuffer& device_activation,
                      const ResidualProjectionView* projection,
                      const ResidualProjectionView* ordering_projection) {
    const std::size_t words = static_cast<std::size_t>(kOutputRows) * tokens;
    const bool use_ordering_projection =
        family.qtype == QType::NVFP4 && tokens >= family.first_quantized_token;
    GuardedDeviceBuffer projected(words * sizeof(std::uint16_t));
    GuardedDeviceBuffer ordering_projected(words * sizeof(std::uint16_t));
    GuardedDeviceBuffer legacy(words * sizeof(std::uint16_t));
    GuardedDeviceBuffer null_projection(words * sizeof(std::uint16_t));
    GuardedDeviceBuffer zero_coefficient(words * sizeof(std::uint16_t));
    projected.copy_from_host(initial_residual.data(), projected.bytes());
    ordering_projected.copy_from_host(initial_residual.data(), ordering_projected.bytes());
    legacy.copy_from_host(initial_residual.data(), legacy.bytes());
    null_projection.copy_from_host(initial_residual.data(), null_projection.bytes());
    zero_coefficient.copy_from_host(initial_residual.data(), zero_coefficient.bytes());
    Tensor x(device_activation.data(), DType::BF16, {family.input_rows, tokens});
    Tensor out(projected.data(), DType::BF16, {kOutputRows, tokens});
    Tensor ordering_out(ordering_projected.data(), DType::BF16, {kOutputRows, tokens});
    Tensor legacy_out(legacy.data(), DType::BF16, {kOutputRows, tokens});
    Tensor null_out(null_projection.data(), DType::BF16, {kOutputRows, tokens});
    Tensor zero_out(zero_coefficient.data(), DType::BF16, {kOutputRows, tokens});

    const std::size_t base_capacity = ops::linear_add_workspace_capacity_bytes(
        family.qtype, kOutputRows, family.input_rows, family.policy, tokens, tokens);
    const std::size_t projected_capacity =
        base_capacity + sizeof(float) * static_cast<std::size_t>(tokens);
    WorkspaceArena workspace(projected_capacity);
    WorkspaceArena ordering_workspace(projected_capacity);
    WorkspaceArena legacy_workspace(std::max<std::size_t>(base_capacity, 256));
    WorkspaceArena null_workspace(std::max<std::size_t>(base_capacity, 256));
    WorkspaceArena zero_workspace(projected_capacity);
    ResidualProjectionView zero_projection = *projection;
    zero_projection.coefficient = 0.0F;

    ops::linear_add(x, weight, out, family.policy, projection, workspace, nullptr);
    if (use_ordering_projection) {
        ops::linear_add(x, weight, ordering_out, family.policy, ordering_projection,
                        ordering_workspace, nullptr);
    }
    ops::linear_add(x, weight, legacy_out, family.policy, legacy_workspace, nullptr);
    ops::linear_add(x, weight, null_out, family.policy, nullptr, null_workspace, nullptr);
    ops::linear_add(x, weight, zero_out, family.policy, &zero_projection, zero_workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "synchronize projection comparisons");

    const std::string label = std::string(family.label) + " T=" + std::to_string(tokens);
    int failures = projected.verify_guards(label);
    if (use_ordering_projection) {
        failures += ordering_projected.verify_guards(label + " ordering projection");
    }
    failures += legacy.verify_guards(label + " legacy");
    failures += null_projection.verify_guards(label + " null");
    failures += zero_coefficient.verify_guards(label + " zero coefficient");
    if (workspace.used() != 0 || workspace.peak_used() != projected_capacity) {
        std::cerr << label << ": projected workspace was not base + exactly 4*T bytes\n";
        ++failures;
    }
    if (use_ordering_projection &&
        (ordering_workspace.used() != 0 ||
         ordering_workspace.peak_used() != projected_capacity)) {
        std::cerr << label << ": ordering projection workspace was not base + exactly 4*T bytes\n";
        ++failures;
    }
    if (zero_workspace.used() != 0 || zero_workspace.peak_used() != projected_capacity) {
        std::cerr << label << ": zero-coefficient workspace was not base + exactly 4*T bytes\n";
        ++failures;
    }

    std::vector<std::uint16_t> actual(words);
    std::vector<std::uint16_t> ordering_actual;
    std::vector<std::uint16_t> legacy_bits(words);
    std::vector<std::uint16_t> null_bits(words);
    std::vector<std::uint16_t> zero_bits(words);
    projected.copy_to_host(actual.data(), projected.bytes());
    if (use_ordering_projection) {
        ordering_actual.resize(words);
        ordering_projected.copy_to_host(ordering_actual.data(), ordering_projected.bytes());
    }
    legacy.copy_to_host(legacy_bits.data(), legacy.bytes());
    null_projection.copy_to_host(null_bits.data(), null_projection.bytes());
    zero_coefficient.copy_to_host(zero_bits.data(), zero_coefficient.bytes());
    if (std::equal(legacy_bits.begin(), legacy_bits.end(), initial_residual.begin())) {
        std::cerr << label << ": sparse fixture produced no observable matmul\n";
        ++failures;
    }
    if (legacy_bits != null_bits || null_workspace.peak_used() != base_capacity) {
        std::cerr << label << ": null projection changed the existing route\n";
        ++failures;
    }
    if (legacy_bits != zero_bits) {
        std::size_t mismatch_count = 0;
        std::size_t first_mismatch = words;
        for (std::size_t index = 0; index < words; ++index) {
            if (legacy_bits[index] == zero_bits[index]) { continue; }
            if (first_mismatch == words) { first_mismatch = index; }
            ++mismatch_count;
        }
        const std::size_t first_token = first_mismatch / kOutputRows;
        const std::size_t first_row = first_mismatch - first_token * kOutputRows;
        std::cerr << label << ": zero coefficient changed " << mismatch_count
                  << " nonzero-matmul BF16 values; first token=" << first_token
                  << " row=" << first_row << " legacy_bits=" << legacy_bits[first_mismatch]
                  << " projected_bits=" << zero_bits[first_mismatch]
                  << " legacy=" << bf16_to_f32(legacy_bits[first_mismatch])
                  << " projected=" << bf16_to_f32(zero_bits[first_mismatch]) << '\n';
        ++failures;
    }

    std::vector<double> actual_values;
    std::vector<double> expected_values;
    actual_values.reserve(words);
    expected_values.reserve(words);
    std::size_t discriminating_elements = 0;
    std::size_t ordering_mismatches = 0;
    std::size_t first_ordering_mismatch = words;
    std::uint16_t first_ordering_actual_bits = 0;
    std::uint16_t first_ordering_expected_bits = 0;
    std::uint16_t first_post_store_bits = 0;
    float first_profiled_matmul = 0.0F;
    float first_ordering_score = 0.0F;
    float first_profiled_correction = 0.0F;
    float first_residual = 0.0F;
    for (std::int32_t token = 0; token < tokens; ++token) {
        const float logical_score = route_score(family, activation, signature, token, tokens);
        const float ordering_score = use_ordering_projection
                                         ? nvfp4_ordering_score(family, activation, token)
                                         : logical_score;
        const ResidualProjectionView* sentinel_projection =
            use_ordering_projection ? ordering_projection : projection;
        const auto& sentinel_actual = use_ordering_projection ? ordering_actual : actual;
        const SparseActivationProfile sparse_activation =
            route_activation_at_sparse_column(family, activation, token, tokens);
        for (std::int32_t row = 0; row < kOutputRows; ++row) {
            const std::size_t index = static_cast<std::size_t>(token) * kOutputRows + row;
            const float weight_value = sparse_weight_value(row);
            const float logical_matmul = weight_value * sparse_activation.logical_value;
            const float emitted_product = weight_value * sparse_activation.emitted_code;
            const float profiled_matmul =
                (emitted_product * sparse_activation.emitted_scale) *
                sparse_activation.output_scale;
            const float profiled_correction =
                sentinel_projection->coefficient * direction[row] * ordering_score;
            const float logical_correction =
                projection->coefficient * direction[row] * logical_score;
            const float residual = bf16_to_f32(initial_residual[index]);
            const std::uint16_t expected_bits =
                f32_to_bf16((profiled_matmul - profiled_correction) + residual);
            const std::uint16_t post_store_bits = f32_to_bf16(
                bf16_to_f32(f32_to_bf16(profiled_matmul + residual)) - profiled_correction);
            if (expected_bits != post_store_bits) {
                ++discriminating_elements;
                if (sentinel_actual[index] != expected_bits) {
                    if (first_ordering_mismatch == words) {
                        first_ordering_mismatch = index;
                        first_ordering_actual_bits = sentinel_actual[index];
                        first_ordering_expected_bits = expected_bits;
                        first_post_store_bits = post_store_bits;
                        first_profiled_matmul = profiled_matmul;
                        first_ordering_score = ordering_score;
                        first_profiled_correction = profiled_correction;
                        first_residual = residual;
                    }
                    ++ordering_mismatches;
                }
            }
            actual_values.push_back(bf16_to_f32(actual[index]));
            expected_values.push_back(
                bf16_to_f32(f32_to_bf16((logical_matmul - logical_correction) + residual)));
        }
    }
    const std::size_t minimum_discriminators =
        static_cast<std::size_t>(tokens) * kMinimumOrderingSentinelsPerToken;
    if (discriminating_elements < minimum_discriminators) {
        std::cerr << label << ": fixture has only " << discriminating_elements
                  << " pre-store ordering sentinels; required at least " << minimum_discriminators
                  << '\n';
        ++failures;
    }
    if (ordering_mismatches != 0) {
        const std::size_t token = first_ordering_mismatch / kOutputRows;
        const std::size_t row = first_ordering_mismatch - token * kOutputRows;
        std::cerr << label << ": " << ordering_mismatches << '/' << discriminating_elements
                  << " ordering sentinels from "
                  << (use_ordering_projection ? "the dedicated NVFP4 ordering projection"
                                              : "the main projection")
                  << " missed the analytical pre-store BF16 result; first token=" << token
                  << " row=" << row << " actual_bits=" << first_ordering_actual_bits
                  << " expected_bits=" << first_ordering_expected_bits
                  << " wrong_post_store_bits=" << first_post_store_bits
                  << " actual=" << bf16_to_f32(first_ordering_actual_bits)
                  << " expected=" << bf16_to_f32(first_ordering_expected_bits)
                  << " wrong_post_store=" << bf16_to_f32(first_post_store_bits)
                  << " profiled_matmul=" << first_profiled_matmul
                  << " score=" << first_ordering_score
                  << " correction=" << first_profiled_correction
                  << " residual=" << first_residual << '\n';
        ++failures;
    }

    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    double max_absolute = 0.0;
    for (std::size_t index = 0; index < actual_values.size(); ++index) {
        max_absolute = std::max(max_absolute,
                                std::abs(actual_values[index] - expected_values[index]));
        dot += actual_values[index] * expected_values[index];
        actual_norm += actual_values[index] * actual_values[index];
        expected_norm += expected_values[index] * expected_values[index];
    }
    const double cosine = dot / std::sqrt(actual_norm * expected_norm);
    if (max_absolute > kMaxAbsoluteError || cosine < kMinimumCosine) {
        std::cerr << label << ": max_abs=" << max_absolute << " cosine=" << cosine << '\n';
        ++failures;
    }
    return failures;
}

int run_numerical_routes() {
    int failures = 0;
    for (const RouteFamily& family : kRouteFamilies) {
        auto host_weight = make_sparse_projection_weight(family);
        GuardedDeviceBuffer device_weight(host_weight.payload.size());
        device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
        const Weight weight = host_weight.device_weight(device_weight.data());
        const auto activation = make_activation(family.input_rows, kTokenCases.back(), family.seed);
        const auto residual = make_residual(kTokenCases.back(), family.seed + 1U);
        const auto signature = make_signature(family.input_rows);
        const auto ordering_signature = make_ordering_signature(family.input_rows);
        const auto direction = make_direction();
        GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer device_signature(signature.size() * sizeof(float));
        GuardedDeviceBuffer device_ordering_signature(ordering_signature.size() * sizeof(float));
        GuardedDeviceBuffer device_direction(direction.size() * sizeof(float));
        device_activation.copy_from_host(activation.data(), device_activation.bytes());
        device_signature.copy_from_host(signature.data(), device_signature.bytes());
        device_ordering_signature.copy_from_host(ordering_signature.data(),
                                                 device_ordering_signature.bytes());
        device_direction.copy_from_host(direction.data(), device_direction.bytes());
        const ResidualProjectionView projection{
            static_cast<const float*>(device_direction.data()),
            static_cast<const float*>(device_signature.data()),
            1.0F,
            static_cast<std::uint32_t>(direction.size()),
            static_cast<std::uint32_t>(signature.size()),
        };
        const ResidualProjectionView ordering_projection{
            static_cast<const float*>(device_direction.data()),
            static_cast<const float*>(device_ordering_signature.data()),
            1.0F,
            static_cast<std::uint32_t>(direction.size()),
            static_cast<std::uint32_t>(ordering_signature.size()),
        };
        for (const std::int32_t tokens : kTokenCases) {
            failures += verify_projection(
                family, tokens,
                std::span<const std::uint16_t>(activation.data(),
                                               static_cast<std::size_t>(family.input_rows) * tokens),
                std::span<const std::uint16_t>(residual.data(),
                                               static_cast<std::size_t>(kOutputRows) * tokens),
                signature, direction, weight, device_activation, &projection,
                &ordering_projection);
        }
        if (family.qtype == QType::NVFP4) {
            failures += verify_projection(
                family, kNvfp4SmallTToken,
                std::span<const std::uint16_t>(
                    activation.data(),
                    static_cast<std::size_t>(family.input_rows) * kNvfp4SmallTToken),
                std::span<const std::uint16_t>(
                    residual.data(), static_cast<std::size_t>(kOutputRows) * kNvfp4SmallTToken),
                signature, direction, weight, device_activation, &projection,
                &ordering_projection);
        }
    }
    return failures;
}

int residual_is_not_projected() {
    const RouteFamily& family = kRouteFamilies.front();
    auto host_weight = make_zero_weight(family);
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());
    std::vector<std::uint16_t> activation(family.input_rows, f32_to_bf16(0.0F));
    const auto residual = make_residual(1, 1301U);
    const auto signature = make_signature(family.input_rows);
    const auto direction = make_direction();
    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_residual(residual.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_signature(signature.size() * sizeof(float));
    GuardedDeviceBuffer device_direction(direction.size() * sizeof(float));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    device_residual.copy_from_host(residual.data(), device_residual.bytes());
    device_signature.copy_from_host(signature.data(), device_signature.bytes());
    device_direction.copy_from_host(direction.data(), device_direction.bytes());
    Tensor x(device_activation.data(), DType::BF16, {family.input_rows, 1});
    Tensor out(device_residual.data(), DType::BF16, {kOutputRows, 1});
    const ResidualProjectionView projection{
        static_cast<const float*>(device_direction.data()),
        static_cast<const float*>(device_signature.data()),
        1.0F,
        static_cast<std::uint32_t>(direction.size()),
        static_cast<std::uint32_t>(signature.size()),
    };
    WorkspaceArena workspace(sizeof(float));
    ops::linear_add(x, weight, out, family.policy, &projection, workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "synchronize residual preservation");
    std::vector<std::uint16_t> actual(residual.size());
    device_residual.copy_to_host(actual.data(), device_residual.bytes());
    if (actual == residual) { return 0; }
    std::cerr << "projection was incorrectly applied to residual input\n";
    return 1;
}

int unsupported_projection_fails_before_launch() {
    constexpr std::int32_t kInputRows = 6144;
    auto host_weight =
        quantized_weight::make_patterned_weight(QType::Q5G64_F16S, kOutputRows, kInputRows, 1321U);
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());
    std::vector<std::uint16_t> activation(kInputRows, f32_to_bf16(0.0F));
    std::vector<std::uint16_t> residual(kOutputRows, f32_to_bf16(1.0F));
    const auto signature = make_signature(kInputRows);
    const auto direction = make_direction();
    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_residual(residual.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_signature(signature.size() * sizeof(float));
    GuardedDeviceBuffer device_direction(direction.size() * sizeof(float));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    device_residual.copy_from_host(residual.data(), device_residual.bytes());
    device_signature.copy_from_host(signature.data(), device_signature.bytes());
    device_direction.copy_from_host(direction.data(), device_direction.bytes());
    Tensor x(device_activation.data(), DType::BF16, {kInputRows, 1});
    Tensor out(device_residual.data(), DType::BF16, {kOutputRows, 1});
    const ResidualProjectionView projection{
        static_cast<const float*>(device_direction.data()),
        static_cast<const float*>(device_signature.data()),
        1.0F,
        static_cast<std::uint32_t>(direction.size()),
        static_cast<std::uint32_t>(signature.size()),
    };
    WorkspaceArena workspace(256);
    try {
        ops::linear_add(x, weight, out, ops::LinearPolicy::A16Only, &projection, workspace,
                        nullptr);
    } catch (const std::invalid_argument&) {
        if (cudaPeekAtLastError() == cudaSuccess && workspace.peak_used() == 0) { return 0; }
        std::cerr << "unsupported projected route performed work before rejection\n";
        return 1;
    }
    std::cerr << "unsupported projected route was accepted\n";
    return 1;
}

int graph_capture_has_no_allocation_nodes() {
    const RouteFamily& family = kRouteFamilies.front();
    constexpr std::int32_t kTokens = 22;
    auto host_weight = make_zero_weight(family);
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());
    const auto activation = make_activation(family.input_rows, kTokens, 1361U);
    const auto residual = make_residual(kTokens, 1367U);
    const auto signature = make_signature(family.input_rows);
    const auto direction = make_direction();
    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_residual(residual.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_signature(signature.size() * sizeof(float));
    GuardedDeviceBuffer device_direction(direction.size() * sizeof(float));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    device_residual.copy_from_host(residual.data(), device_residual.bytes());
    device_signature.copy_from_host(signature.data(), device_signature.bytes());
    device_direction.copy_from_host(direction.data(), device_direction.bytes());
    Tensor x(device_activation.data(), DType::BF16, {family.input_rows, kTokens});
    Tensor out(device_residual.data(), DType::BF16, {kOutputRows, kTokens});
    const ResidualProjectionView projection{
        static_cast<const float*>(device_direction.data()),
        static_cast<const float*>(device_signature.data()),
        1.0F,
        static_cast<std::uint32_t>(direction.size()),
        static_cast<std::uint32_t>(signature.size()),
    };
    const std::size_t base_capacity = ops::linear_add_workspace_capacity_bytes(
        family.qtype, kOutputRows, family.input_rows, family.policy, kTokens, kTokens);
    WorkspaceArena workspace(base_capacity + sizeof(float) * kTokens);
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create projection capture stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin projection capture");
    ops::linear_add(x, weight, out, family.policy, &projection, workspace, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end projection capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate projection graph");
    cuda_check(cudaGraphLaunch(executable, stream), "launch projection graph");
    cuda_check(cudaGraphLaunch(executable, stream), "replay projection graph");
    cuda_check(cudaStreamSynchronize(stream), "synchronize projection graph");

    std::size_t node_count = 0;
    cuda_check(cudaGraphGetNodes(graph, nullptr, &node_count), "count projection graph nodes");
    std::vector<cudaGraphNode_t> nodes(node_count);
    cuda_check(cudaGraphGetNodes(graph, nodes.data(), &node_count), "read projection graph nodes");
    int failures = 0;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        cuda_check(cudaGraphNodeGetType(node, &type), "read projection graph node type");
        if (type == cudaGraphNodeTypeMemAlloc || type == cudaGraphNodeTypeMemFree) {
            std::cerr << "projected linear_add captured a device allocation node\n";
            ++failures;
        }
    }
    cuda_check(cudaGraphExecDestroy(executable), "destroy projection graph executable");
    cuda_check(cudaGraphDestroy(graph), "destroy projection graph");
    cuda_check(cudaStreamDestroy(stream), "destroy projection capture stream");
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        failures += run_numerical_routes();
        failures += residual_is_not_projected();
        failures += unsupported_projection_fails_before_launch();
        failures += graph_capture_has_no_allocation_nodes();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " residual projection Op\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "residual projection Op: " << error.what() << '\n';
        return 1;
    }
}
