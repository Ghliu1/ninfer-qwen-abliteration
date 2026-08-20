#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear_add/fp8/fp8_linear_add_epilogue.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                std::int32_t tokens, const ResidualProjectionView* projection,
                const float* scores, cudaStream_t stream) {
    using Schedule          = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    auto* output            = static_cast<__nv_bfloat16*>(residual.data);
    const Fp8ContiguousOutput destination{output, Geometry::kOutputRows};

    if (projection == nullptr) {
        if constexpr (Schedule::kSharedBytes > 48 * 1024) {
            static const cudaError_t attribute = cudaFuncSetAttribute(
                fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8AddResidualEpilogue,
                               Fp8ContiguousOutput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
            CUDA_CHECK(attribute);
        }
        fp8_mma_kernel<Geometry, Schedule, FullTokens>
            <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
                workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const __nv_bfloat16*>(weight.scales), tokens,
                Fp8AddResidualEpilogue{output, Geometry::kOutputRows}, destination);
    } else {
        if constexpr (Schedule::kSharedBytes > 48 * 1024) {
            static const cudaError_t attribute = cudaFuncSetAttribute(
                fp8_mma_kernel<Geometry, Schedule, FullTokens,
                               Fp8ProjectedAddResidualEpilogue, Fp8ContiguousOutput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
            CUDA_CHECK(attribute);
        }
        fp8_mma_kernel<Geometry, Schedule, FullTokens>
            <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
                workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const __nv_bfloat16*>(weight.scales), tokens,
                Fp8ProjectedAddResidualEpilogue{output, Geometry::kOutputRows,
                                                 projection->direction, scores,
                                                 projection->coefficient},
                destination);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                    std::int32_t tokens, const ResidualProjectionView* projection,
                    const float* scores, cudaStream_t stream) {
    using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    if ((tokens % Schedule::kBlockTokens) == 0) {
        launch_mma<Geometry, true>(weight, residual, workspace, tokens, projection, scores, stream);
    } else {
        launch_mma<Geometry, false>(weight, residual, workspace, tokens, projection, scores,
                                    stream);
    }
}

} // namespace

void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace,
                              const ResidualProjectionView* projection, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
    float* scores = nullptr;
    if (projection == nullptr) {
        launch_fp8_a8_quantize(x, weight, scratch, stream);
    } else {
        const DeviceSpan score_storage = workspace.alloc_bytes(
            sizeof(float) * static_cast<std::size_t>(x.ne[1]), alignof(float));
        scores = static_cast<float*>(score_storage.data);
        launch_fp8_a8_quantize_scored(x, weight, scratch, projection->signature, scores, stream);
    }
    switch (resolve_fp8_problem(weight.n, weight.k)) {
    case Fp8Problem::Residual6144:
        launch_problem<Fp8Residual6144Geometry>(weight, residual, scratch, x.ne[1], projection,
                                                scores, stream);
        return;
    case Fp8Problem::Residual17408:
        launch_problem<Fp8Residual17408Geometry>(weight, residual, scratch, x.ne[1], projection,
                                                 scores, stream);
        return;
    case Fp8Problem::AttnInput:
    case Fp8Problem::GdnInput:
    case Fp8Problem::MlpGateUp:
    case Fp8Problem::Vocabulary:
        break;
    }
    throw std::invalid_argument("fp8 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
