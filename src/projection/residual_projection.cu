#include "projection/residual_projection_kernels.h"

#include "core/device.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void projection_score_a16_kernel(
    const __nv_bfloat16* __restrict__ activation, const float* __restrict__ signature,
    float* __restrict__ scores, std::int32_t input_rows) {
    __shared__ float warp_sums[Threads / 32];
    const std::int32_t token = static_cast<std::int32_t>(blockIdx.x);
    const auto* token_activation =
        activation + static_cast<std::int64_t>(token) * input_rows;
    float local = 0.0F;
    for (std::int32_t column = static_cast<std::int32_t>(threadIdx.x); column < input_rows;
         column += Threads) {
        local += signature[column] * __bfloat162float(token_activation[column]);
    }
    const float score = block_reduce_sum<Threads>(local, warp_sums);
    if (threadIdx.x == 0) { scores[token] = score; }
}

} // namespace

float* projection_score_a16(const ResidualProjectionView& projection, const Tensor& x,
                            std::int32_t tokens, cudaStream_t stream,
                            WorkspaceArena& workspace) {
    if (tokens <= 0 || tokens != x.ne[1]) {
        throw std::invalid_argument("projection score A16: invalid token count");
    }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("projection score A16: workspace size overflow");
    }
    const DeviceSpan storage = workspace.alloc_bytes(count * sizeof(float), alignof(float));
    auto* scores = static_cast<float*>(storage.data);
    constexpr int kThreads = 256;
    projection_score_a16_kernel<kThreads><<<tokens, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), projection.signature, scores, x.ne[0]);
    CUDA_CHECK(cudaGetLastError());
    return scores;
}

} // namespace ninfer::ops::detail
