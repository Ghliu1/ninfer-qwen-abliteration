#include "core/device.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <utility>

int main() {
    const char* enabled = std::getenv("NINFER_RUN_QWEN38_CAPACITY_PLAN_TEST");
    if (enabled == nullptr || std::string_view(enabled) != "1") {
        std::cout << "skip: set NINFER_RUN_QWEN38_CAPACITY_PLAN_TEST=1 under the GPU lease\n";
        return 77;
    }

    using Package = ninfer::targets::qwen3_6_27b::Package;
    ninfer::DeviceContext device(0);
    ninfer::EngineOptions options;
    options.max_context                    = 204800;
    options.kv_capacity                    = ninfer::KvCapacityPolicy::explicit_capacity(221184);
    options.max_concurrency                = 2;
    options.prefill_chunk                  = 1024;
    options.kv_cache                       = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend            = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens       = 3;
    options.speculative.proposal_head      = ninfer::ProposalHead::Optimized;
    options.enable_vision                  = true;
    options.max_merged_vision_tokens       = 4096;
    options.use_cuda_graph                 = true;

    auto planner = Package::make_sequence_planner(
        device, options, Package::WeightsProfile::Qwen38Nvfp4);
    auto sequence = std::move(planner).finalize(3456);
    if (sequence.capacity() != 204800 || sequence.kv_capacity() != 221184 ||
        sequence.max_concurrency() != 2) {
        std::cerr << "Qwen3.8 capacity identity changed\n";
        return 1;
    }
    if (sequence.workspace_capacity_bytes() != 277481216ULL) {
        std::cerr << "4096-token Vision workspace is "
                  << sequence.workspace_capacity_bytes() << ", expected 277481216\n";
        return 1;
    }
    if (sequence.request_transient_capacity_bytes() != 41943040ULL) {
        std::cerr << "4096-token Vision transient is "
                  << sequence.request_transient_capacity_bytes() << ", expected 41943040\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
