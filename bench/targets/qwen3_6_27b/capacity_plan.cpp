#include "core/device.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <charconv>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace {

using Json = nlohmann::json;
using Package = ninfer::targets::qwen3_6_27b::Package;

constexpr std::uint64_t kReserveFloorBytes = 1024ULL * 1024ULL * 1024ULL;

struct Options {
    std::uint32_t max_context             = 204800;
    std::uint32_t kv_capacity             = 221184;
    std::uint32_t max_concurrency         = 2;
    std::uint32_t prefill_chunk           = 1024;
    std::uint32_t mtp_draft_tokens        = 3;
    std::uint32_t max_merged_vision_tokens = 4096;
    int device                            = 0;
    ninfer::KvCacheStorage kv_cache       = ninfer::KvCacheStorage::Int8Group64;
    ninfer::ProposalHead proposal_head    = ninfer::ProposalHead::Optimized;
    bool cuda_graph                       = true;
    bool help                             = false;
};

std::string usage(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " [--max-context N] [--kv-capacity N] [--max-concurrency N]"
           " [--kv-dtype bf16|int8] [--prefill-chunk N]"
           " [--mtp-draft-tokens 1..5] [--proposal-head full|optimized]"
           " [--max-merged-vision-tokens N] [--cuda-graph on|off] [--device N]\n";
}

std::uint32_t parse_u32(std::string_view text, std::string_view name) {
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || error != std::errc{} || end != text.data() + text.size() || value == 0 ||
        value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid " + std::string(name) + ": " + std::string(text));
    }
    return static_cast<std::uint32_t>(value);
}

int parse_device(std::string_view text) {
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || error != std::errc{} || end != text.data() + text.size() ||
        value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid device: " + std::string(text));
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        const auto value = [&]() -> std::string_view {
            if (++i >= argc) {
                throw std::invalid_argument("missing value for " + std::string(argument));
            }
            return argv[i];
        };
        if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else if (argument == "--max-context") {
            options.max_context = parse_u32(value(), "max-context");
        } else if (argument == "--kv-capacity") {
            options.kv_capacity = parse_u32(value(), "kv-capacity");
        } else if (argument == "--max-concurrency") {
            options.max_concurrency = parse_u32(value(), "max-concurrency");
        } else if (argument == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value(), "prefill-chunk");
        } else if (argument == "--mtp-draft-tokens") {
            options.mtp_draft_tokens = parse_u32(value(), "mtp-draft-tokens");
        } else if (argument == "--max-merged-vision-tokens") {
            options.max_merged_vision_tokens =
                parse_u32(value(), "max-merged-vision-tokens");
        } else if (argument == "--device") {
            options.device = parse_device(value());
        } else if (argument == "--kv-dtype") {
            const std::string_view name = value();
            if (name == "bf16") {
                options.kv_cache = ninfer::KvCacheStorage::BFloat16;
            } else if (name == "int8") {
                options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
            } else {
                throw std::invalid_argument("--kv-dtype must be bf16 or int8");
            }
        } else if (argument == "--proposal-head") {
            const std::string_view name = value();
            if (name == "full") {
                options.proposal_head = ninfer::ProposalHead::Full;
            } else if (name == "optimized") {
                options.proposal_head = ninfer::ProposalHead::Optimized;
            } else {
                throw std::invalid_argument("--proposal-head must be full or optimized");
            }
        } else if (argument == "--cuda-graph") {
            const std::string_view name = value();
            if (name == "on") {
                options.cuda_graph = true;
            } else if (name == "off") {
                options.cuda_graph = false;
            } else {
                throw std::invalid_argument("--cuda-graph must be on or off");
            }
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.mtp_draft_tokens == 0 || options.mtp_draft_tokens > 5) {
        throw std::invalid_argument("--mtp-draft-tokens must be in [1,5]");
    }
    return options;
}

const char* kv_cache_name(ninfer::KvCacheStorage storage) {
    return storage == ninfer::KvCacheStorage::BFloat16 ? "bf16" : "int8-group64";
}

const char* proposal_head_name(ninfer::ProposalHead head) {
    return head == ninfer::ProposalHead::Full ? "full" : "optimized";
}

std::uint64_t as_u64(std::size_t value) {
    return static_cast<std::uint64_t>(value);
}

Json run(const Options& cli) {
    ninfer::DeviceContext device(cli.device);
    ninfer::EngineOptions options;
    options.device                          = cli.device;
    options.max_context                     = cli.max_context;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(cli.kv_capacity);
    options.max_concurrency                 = cli.max_concurrency;
    options.prefill_chunk                   = cli.prefill_chunk;
    options.kv_cache                        = cli.kv_cache;
    options.speculative.backend             = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens        = cli.mtp_draft_tokens;
    options.speculative.proposal_head       = cli.proposal_head;
    options.enable_vision                   = true;
    options.max_merged_vision_tokens        = cli.max_merged_vision_tokens;
    options.use_cuda_graph                  = cli.cuda_graph;

    auto planner = Package::make_sequence_planner(
        device, options, Package::WeightsProfile::Qwen38Nvfp4);
    const auto& curve = planner.capacity_curve();
    if (cli.kv_capacity % curve.main_page_tokens != 0) {
        throw std::invalid_argument("--kv-capacity must be aligned to the production KV page size");
    }
    const std::uint32_t page_groups = cli.kv_capacity / curve.main_page_tokens;
    if (page_groups < curve.minimum_main_page_groups ||
        page_groups > curve.maximum_main_page_groups) {
        throw std::invalid_argument("--kv-capacity is outside the production planner range");
    }
    auto sequence = std::move(planner).finalize(page_groups);

    std::size_t free_device_bytes  = 0;
    std::size_t total_device_bytes = 0;
    const cudaError_t memory_error = cudaMemGetInfo(&free_device_bytes, &total_device_bytes);
    if (memory_error != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemGetInfo failed: ") +
                                 cudaGetErrorString(memory_error));
    }

    const std::uint64_t reservation = as_u64(sequence.device_reservation_bytes());
    const std::uint64_t free_bytes  = as_u64(free_device_bytes);
    const bool admissible =
        reservation <= free_bytes && kReserveFloorBytes <= free_bytes - reservation;
    return Json{
        {"schema_version", 1},
        {"target", "qwen3.8-27b"},
        {"weights_profile", "qwen38-nvfp4"},
        {"inputs",
         {{"max_context", cli.max_context},
          {"kv_capacity", cli.kv_capacity},
          {"max_concurrency", cli.max_concurrency},
          {"kv_dtype", kv_cache_name(cli.kv_cache)},
          {"prefill_chunk", cli.prefill_chunk},
          {"speculative_backend", "mtp"},
          {"mtp_draft_tokens", cli.mtp_draft_tokens},
          {"proposal_head", proposal_head_name(cli.proposal_head)},
          {"vision", true},
          {"max_merged_vision_tokens", cli.max_merged_vision_tokens},
          {"cuda_graph", cli.cuda_graph},
          {"device", cli.device}}},
        {"memory",
         {{"persistent_bytes", as_u64(sequence.persistent_capacity_bytes())},
          {"workspace_bytes", as_u64(sequence.workspace_capacity_bytes())},
          {"request_transient_bytes", as_u64(sequence.request_transient_capacity_bytes())},
          {"graph_allowance_bytes", as_u64(sequence.graph_allowance_bytes())},
          {"total_device_reservation_bytes", reservation},
          {"free_device_bytes", free_bytes},
          {"total_device_bytes", as_u64(total_device_bytes)},
          {"reserve_floor_bytes", kReserveFloorBytes}}},
        {"admissible", admissible}};
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            std::cout << usage(argv[0]);
            return 0;
        }
        std::cout << run(options).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen38_capacity_plan: " << error.what() << '\n' << usage(argv[0]);
        return 1;
    }
}
