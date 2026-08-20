#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/projection/residual_projection.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Allocates exactly one FP32 score per token from caller-owned transient storage and enqueues the
// BF16-signature dot on the caller stream. The enclosing Op scope owns the returned span.
float* projection_score_a16(const ResidualProjectionView& projection, const Tensor& x,
                            std::int32_t tokens, cudaStream_t stream,
                            WorkspaceArena& workspace);

} // namespace ninfer::ops::detail
