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
constexpr std::array<std::int32_t, 6> kTokenCases{1, 16, 22, 25, 256, 1024};
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

std::vector<float> make_direction() {
    const float value = 1.0F / std::sqrt(static_cast<float>(kOutputRows));
    return std::vector<float>(kOutputRows, value);
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

quantized_weight::PackedWeight make_zero_weight(const RouteFamily& family) {
    quantized_weight::PatternedWeightOptions options;
    if (family.qtype == QType::NVFP4) {
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor = 3.5F;
    }
    auto weight = quantized_weight::make_patterned_weight(
        family.qtype, kOutputRows, family.input_rows, family.seed, options);
    std::fill_n(weight.payload.begin(), static_cast<std::size_t>(weight.code_plane_bytes),
                static_cast<std::uint8_t>(0));
    return weight;
}

int verify_projection(const RouteFamily& family, std::int32_t tokens,
                      std::span<const std::uint16_t> activation,
                      std::span<const std::uint16_t> initial_residual,
                      std::span<const float> signature, std::span<const float> direction,
                      const Weight& weight, GuardedDeviceBuffer& device_activation,
                      const ResidualProjectionView* projection) {
    const std::size_t words = static_cast<std::size_t>(kOutputRows) * tokens;
    GuardedDeviceBuffer projected(words * sizeof(std::uint16_t));
    projected.copy_from_host(initial_residual.data(), projected.bytes());
    Tensor x(device_activation.data(), DType::BF16, {family.input_rows, tokens});
    Tensor out(projected.data(), DType::BF16, {kOutputRows, tokens});

    const std::size_t base_capacity = ops::linear_add_workspace_capacity_bytes(
        family.qtype, kOutputRows, family.input_rows, family.policy, tokens, tokens);
    const std::size_t projected_capacity =
        base_capacity + sizeof(float) * static_cast<std::size_t>(tokens);
    WorkspaceArena workspace(projected_capacity);

    // RED: Task 5 adds this projection-bearing overload. Its execution must consume exactly one
    // FP32 score per token in addition to the pre-existing route workspace.
    ops::linear_add(x, weight, out, family.policy, projection, workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "synchronize projected linear_add");

    const std::string label = std::string(family.label) + " T=" + std::to_string(tokens);
    int failures = projected.verify_guards(label);
    if (workspace.used() != 0 || workspace.peak_used() != projected_capacity) {
        std::cerr << label << ": projected workspace was not base + exactly 4*T bytes\n";
        ++failures;
    }

    std::vector<std::uint16_t> actual(words);
    projected.copy_to_host(actual.data(), projected.bytes());
    std::vector<double> actual_values;
    std::vector<double> expected_values;
    actual_values.reserve(words);
    expected_values.reserve(words);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const float score = route_score(family, activation, signature, token, tokens);
        for (std::int32_t row = 0; row < kOutputRows; ++row) {
            const std::size_t index = static_cast<std::size_t>(token) * kOutputRows + row;
            const float expected = bf16_to_f32(f32_to_bf16(
                bf16_to_f32(initial_residual[index]) - projection->coefficient * direction[row] *
                                                          score));
            actual_values.push_back(bf16_to_f32(actual[index]));
            expected_values.push_back(expected);
        }
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

    GuardedDeviceBuffer legacy(words * sizeof(std::uint16_t));
    GuardedDeviceBuffer null_projection(words * sizeof(std::uint16_t));
    legacy.copy_from_host(initial_residual.data(), legacy.bytes());
    null_projection.copy_from_host(initial_residual.data(), null_projection.bytes());
    Tensor legacy_out(legacy.data(), DType::BF16, {kOutputRows, tokens});
    Tensor null_out(null_projection.data(), DType::BF16, {kOutputRows, tokens});
    WorkspaceArena legacy_workspace(std::max<std::size_t>(base_capacity, 256));
    WorkspaceArena null_workspace(std::max<std::size_t>(base_capacity, 256));
    ops::linear_add(x, weight, legacy_out, family.policy, legacy_workspace, nullptr);
    ops::linear_add(x, weight, null_out, family.policy, nullptr, null_workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "synchronize null projection comparison");
    std::vector<std::uint16_t> legacy_bits(words);
    std::vector<std::uint16_t> null_bits(words);
    legacy.copy_to_host(legacy_bits.data(), legacy.bytes());
    null_projection.copy_to_host(null_bits.data(), null_projection.bytes());
    if (legacy_bits != null_bits || null_workspace.peak_used() != base_capacity) {
        std::cerr << label << ": null projection changed the existing route\n";
        ++failures;
    }
    return failures;
}

int run_numerical_routes() {
    int failures = 0;
    for (const RouteFamily& family : kRouteFamilies) {
        auto host_weight = make_zero_weight(family);
        GuardedDeviceBuffer device_weight(host_weight.payload.size());
        device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
        const Weight weight = host_weight.device_weight(device_weight.data());
        const auto activation = make_activation(family.input_rows, kTokenCases.back(), family.seed);
        const auto residual = make_residual(kTokenCases.back(), family.seed + 1U);
        const auto signature = make_signature(family.input_rows);
        const auto direction = make_direction();
        GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer device_signature(signature.size() * sizeof(float));
        GuardedDeviceBuffer device_direction(direction.size() * sizeof(float));
        device_activation.copy_from_host(activation.data(), device_activation.bytes());
        device_signature.copy_from_host(signature.data(), device_signature.bytes());
        device_direction.copy_from_host(direction.data(), device_direction.bytes());
        const ResidualProjectionView projection{
            static_cast<const float*>(device_direction.data()),
            static_cast<const float*>(device_signature.data()),
            1.0F,
            static_cast<std::uint32_t>(direction.size()),
            static_cast<std::uint32_t>(signature.size()),
        };
        for (const std::int32_t tokens : kTokenCases) {
            failures += verify_projection(
                family, tokens,
                std::span<const std::uint16_t>(activation.data(),
                                               static_cast<std::size_t>(family.input_rows) * tokens),
                std::span<const std::uint16_t>(residual.data(),
                                               static_cast<std::size_t>(kOutputRows) * tokens),
                signature, direction, weight, device_activation, &projection);
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
