#include "targets/qwen3_6_27b/impl/variant.h"

#include "core/layout.h"
#include "ninfer/engine.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/projection/residual_projection.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace projection_trace {

struct Snapshot {
    std::size_t calls                = 0;
    std::size_t invalid              = 0;
    std::size_t duplicates           = 0;
    std::size_t attention            = 0;
    std::size_t gdn                  = 0;
    std::size_t mlp                  = 0;
    std::size_t unique               = 0;
    std::size_t mtp_linear_calls     = 0;
    std::size_t mtp_silu_calls       = 0;
    std::size_t mtp_residual_calls   = 0;
    std::size_t mtp_projection_calls = 0;
    std::array<std::array<std::size_t, 3>, 64> identity_counts{};
};

std::mutex mutex;
std::array<std::array<std::size_t, 3>, 64> identity_counts{};
std::size_t calls                = 0;
std::size_t invalid              = 0;
std::size_t duplicates           = 0;
std::size_t mtp_linear_calls     = 0;
std::size_t mtp_silu_calls       = 0;
std::size_t mtp_residual_calls   = 0;
std::size_t mtp_projection_calls = 0;
thread_local std::size_t mtp_writer_depth = 0;

void reset() {
    std::lock_guard lock(mutex);
    identity_counts.fill({});
    calls                = 0;
    invalid              = 0;
    duplicates           = 0;
    mtp_linear_calls     = 0;
    mtp_silu_calls       = 0;
    mtp_residual_calls   = 0;
    mtp_projection_calls = 0;
}

void record(std::uint32_t layer, ninfer::ProjectionSite site) {
    std::lock_guard lock(mutex);
    ++calls;
    if (mtp_writer_depth != 0) { ++mtp_projection_calls; }
    if (layer >= 64 ||
        (site != ninfer::ProjectionSite::AttentionOutput &&
         site != ninfer::ProjectionSite::GdnOutput && site != ninfer::ProjectionSite::MlpDown)) {
        ++invalid;
        return;
    }
    const std::size_t site_index = static_cast<std::size_t>(site);
    if (identity_counts[layer][site_index]++ != 0) { ++duplicates; }
}

struct MtpFixtureScope {
    MtpFixtureScope() { ++mtp_writer_depth; }
    ~MtpFixtureScope() { --mtp_writer_depth; }
    MtpFixtureScope(const MtpFixtureScope&)            = delete;
    MtpFixtureScope& operator=(const MtpFixtureScope&) = delete;
};

bool fixture_active() { return mtp_writer_depth != 0; }

void record_mtp_linear() {
    std::lock_guard lock(mutex);
    ++mtp_linear_calls;
}

void record_mtp_silu() {
    std::lock_guard lock(mutex);
    ++mtp_silu_calls;
}

void record_mtp_residual() {
    std::lock_guard lock(mutex);
    ++mtp_residual_calls;
}

Snapshot snapshot() {
    std::lock_guard lock(mutex);
    Snapshot out{.calls                = calls,
                 .invalid              = invalid,
                 .duplicates           = duplicates,
                 .mtp_linear_calls     = mtp_linear_calls,
                 .mtp_silu_calls       = mtp_silu_calls,
                 .mtp_residual_calls   = mtp_residual_calls,
                 .mtp_projection_calls = mtp_projection_calls};
    for (const auto& counts : identity_counts) {
        if (counts[0] != 0) {
            ++out.attention;
            ++out.unique;
        }
        if (counts[1] != 0) {
            ++out.gdn;
            ++out.unique;
        }
        if (counts[2] != 0) {
            ++out.mlp;
            ++out.unique;
        }
    }
    out.identity_counts = identity_counts;
    return out;
}

bool exactly_one_backbone_traversal(const Snapshot& trace) {
    if (trace.calls != 128 || trace.invalid != 0 || trace.duplicates != 0 ||
        trace.unique != 128 || trace.attention != 16 || trace.gdn != 48 || trace.mlp != 64) {
        return false;
    }
    for (std::uint32_t layer = 0; layer < 64; ++layer) {
        const bool attention = layer % 4 == 3;
        if (trace.identity_counts[layer][0] != static_cast<std::size_t>(attention) ||
            trace.identity_counts[layer][1] != static_cast<std::size_t>(!attention) ||
            trace.identity_counts[layer][2] != 1) {
            return false;
        }
    }
    return true;
}

bool contains_only_complete_backbone_traversals(const Snapshot& trace) {
    if (trace.calls < 128 || trace.invalid != 0 || trace.unique != 128 ||
        trace.attention != 16 || trace.gdn != 48 || trace.mlp != 64) {
        return false;
    }
    for (std::uint32_t layer = 0; layer < 64; ++layer) {
        const bool attention = layer % 4 == 3;
        const std::size_t attention_count = trace.identity_counts[layer][0];
        const std::size_t gdn_count       = trace.identity_counts[layer][1];
        if ((attention && (attention_count == 0 || gdn_count != 0)) ||
            (!attention && (gdn_count == 0 || attention_count != 0)) ||
            trace.identity_counts[layer][2] == 0) {
            return false;
        }
    }
    return true;
}

} // namespace projection_trace

extern "C" ninfer::ResidualProjectionView
__real__ZNK6ninfer23ResidualProjectionTable4viewEjNS_14ProjectionSiteE(
    const ninfer::ResidualProjectionTable*, std::uint32_t, ninfer::ProjectionSite);

extern "C" ninfer::ResidualProjectionView
__wrap__ZNK6ninfer23ResidualProjectionTable4viewEjNS_14ProjectionSiteE(
    const ninfer::ResidualProjectionTable* table, std::uint32_t layer,
    ninfer::ProjectionSite site) {
    projection_trace::record(layer, site);
    return __real__ZNK6ninfer23ResidualProjectionTable4viewEjNS_14ProjectionSiteE(table, layer,
                                                                                  site);
}

extern "C" void
__real__ZN6ninfer3ops6linearERKNS_6TensorERKNS_6WeightERS1_P11CUstream_st(
    const ninfer::Tensor&, const ninfer::Weight&, ninfer::Tensor&, cudaStream_t);

extern "C" void
__wrap__ZN6ninfer3ops6linearERKNS_6TensorERKNS_6WeightERS1_P11CUstream_st(
    const ninfer::Tensor& input, const ninfer::Weight& weight, ninfer::Tensor& output,
    cudaStream_t stream) {
    if (projection_trace::fixture_active()) {
        projection_trace::record_mtp_linear();
        return;
    }
    __real__ZN6ninfer3ops6linearERKNS_6TensorERKNS_6WeightERS1_P11CUstream_st(input, weight,
                                                                              output, stream);
}

extern "C" void
__real__ZN6ninfer3ops8silu_mulERKNS_6TensorES3_RS1_P11CUstream_st(
    const ninfer::Tensor&, const ninfer::Tensor&, ninfer::Tensor&, cudaStream_t);

extern "C" void
__wrap__ZN6ninfer3ops8silu_mulERKNS_6TensorES3_RS1_P11CUstream_st(
    const ninfer::Tensor& gate, const ninfer::Tensor& value, ninfer::Tensor& output,
    cudaStream_t stream) {
    if (projection_trace::fixture_active()) {
        projection_trace::record_mtp_silu();
        return;
    }
    __real__ZN6ninfer3ops8silu_mulERKNS_6TensorES3_RS1_P11CUstream_st(gate, value, output, stream);
}

extern "C" void
__real__ZN6ninfer3ops12residual_addERKNS_6TensorERS1_P11CUstream_st(
    const ninfer::Tensor&, ninfer::Tensor&, cudaStream_t);

extern "C" void
__wrap__ZN6ninfer3ops12residual_addERKNS_6TensorERS1_P11CUstream_st(
    const ninfer::Tensor& delta, ninfer::Tensor& residual, cudaStream_t stream) {
    if (projection_trace::fixture_active()) {
        projection_trace::record_mtp_residual();
        return;
    }
    __real__ZN6ninfer3ops12residual_addERKNS_6TensorERS1_P11CUstream_st(delta, residual, stream);
}

namespace {

using ninfer::ProjectionSite;
using ninfer::ResidualProjectionTable;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::WorkspaceArena;
using ninfer::ops::LinearPolicy;
using ninfer::targets::qwen3_6::TextPhase;
using ninfer::targets::qwen3_6_27b::detail::Variant;
using WeightsProfile = Variant::WeightsProfile;

using AttentionWriter = void (*)(const Tensor&, const Weight&, Tensor&, std::uint32_t,
                                 const ResidualProjectionTable*, TextPhase, WorkspaceArena&,
                                 cudaStream_t);
using GdnWriter = void (*)(const Tensor&, const Weight&, Tensor&, std::uint32_t,
                           const ResidualProjectionTable*, TextPhase, WorkspaceArena&,
                           cudaStream_t);
using MlpWriter = void (*)(const Tensor&, const Variant::PostMixerWeights&, Tensor&,
                           std::uint32_t, const ResidualProjectionTable*, TextPhase,
                           WorkspaceArena&, cudaStream_t);
using MtpWriter = void (*)(const Tensor&, const Variant::MtpPostMixerWeights&, Tensor&,
                           WorkspaceArena&, cudaStream_t);

static_assert(std::is_same_v<
              decltype(static_cast<AttentionWriter>(&Variant::attention_output_projection)),
              AttentionWriter>);
static_assert(std::is_same_v<
              decltype(static_cast<GdnWriter>(&Variant::gdn_output_projection)), GdnWriter>);
static_assert(
    std::is_same_v<decltype(static_cast<MlpWriter>(&Variant::post_mixer)), MlpWriter>);
// The draft MTP writer deliberately retains the legacy signature: it has neither a layer identity
// nor access to the backbone projection table.
static_assert(
    std::is_same_v<decltype(static_cast<MtpWriter>(&Variant::mtp_post_mixer)), MtpWriter>);

constexpr std::array<std::int32_t, 6> kTokens{1, 16, 22, 25, 256, 1024};
constexpr std::int32_t kHidden       = 5120;
constexpr std::int32_t kMixerInput   = 6144;
constexpr std::int32_t kMlpInput     = 17408;

std::size_t score_bytes(std::int32_t tokens) {
    return sizeof(float) * static_cast<std::size_t>(tokens);
}

std::size_t expected_post_mixer_workspace(ninfer::QType qtype, LinearPolicy policy,
                                          std::int32_t first, std::int32_t last,
                                          bool projected) {
    ninfer::WorkspaceLayoutBuilder layout;
    (void)layout.alloc(ninfer::DType::BF16, {kMlpInput, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ninfer::ops::linear_swiglu_workspace_capacity_bytes(
            qtype, 2 * kMlpInput, kHidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        std::size_t bytes = ninfer::ops::linear_add_workspace_capacity_bytes(
            qtype, kHidden, kMlpInput, policy, first, last);
        if (projected) { bytes += score_bytes(last); }
        (void)layout.alloc_bytes(bytes);
    }
    return layout.peak_bytes(1);
}

int verify_claim_inventory() {
    std::set<std::pair<std::uint32_t, ProjectionSite>> claims;
    std::size_t duplicates = 0;
    std::size_t attention  = 0;
    std::size_t gdn        = 0;
    std::size_t mlp        = 0;

    for (std::uint32_t layer = 0; layer < 64; ++layer) {
        const ProjectionSite mixer = Variant::mixer_projection_site(layer);
        const ProjectionSite expected =
            layer % 4 == 3 ? ProjectionSite::AttentionOutput : ProjectionSite::GdnOutput;
        if (mixer != expected) {
            std::cerr << "layer " << layer << " resolved the wrong mixer projection site\n";
            return 1;
        }
        attention += mixer == ProjectionSite::AttentionOutput ? 1U : 0U;
        gdn += mixer == ProjectionSite::GdnOutput ? 1U : 0U;
        mlp += 1;
        if (!claims.emplace(layer, mixer).second) { ++duplicates; }
        if (!claims.emplace(layer, ProjectionSite::MlpDown).second) { ++duplicates; }
    }

    if (claims.size() != 128 || duplicates != 0 || attention != 16 || gdn != 48 || mlp != 64) {
        std::cerr << "backbone claim inventory is incomplete: claims=" << claims.size()
                  << " duplicates=" << duplicates << " attention=" << attention
                  << " gdn=" << gdn << " mlp=" << mlp << '\n';
        return 1;
    }

    try {
        (void)Variant::mixer_projection_site(64);
    } catch (const std::out_of_range&) {
        return 0;
    }
    std::cerr << "out-of-range layer identity was accepted\n";
    return 1;
}

int verify_writer_site_validation() {
    Tensor tensor;
    Weight weight;
    std::array<std::byte, 1> storage{};
    WorkspaceArena workspace(ninfer::DeviceSpan{storage.data(), storage.size()});
    int failures = 0;

    try {
        Variant::attention_output_projection(tensor, weight, tensor, 0, nullptr,
                                             TextPhase::Prefill, workspace, nullptr);
        std::cerr << "attention writer accepted a GDN layer identity\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    }
    try {
        Variant::gdn_output_projection(tensor, weight, tensor, 3, nullptr, TextPhase::Prefill,
                                       workspace, nullptr);
        std::cerr << "GDN writer accepted an attention layer identity\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    }
    try {
        Variant::post_mixer(tensor, Variant::PostMixerWeights{}, tensor, 64, nullptr,
                            TextPhase::Prefill, workspace, nullptr);
        std::cerr << "MLP writer accepted an out-of-range layer identity\n";
        ++failures;
    } catch (const std::out_of_range&) {
    }
    return failures;
}

int verify_mtp_writer_isolation() {
    std::array<std::byte, 2> tensor_storage{};
    Tensor hidden(tensor_storage.data(), ninfer::DType::BF16, {kHidden, 1});
    Tensor residual(tensor_storage.data(), ninfer::DType::BF16, {kHidden, 1});
    std::vector<std::byte> workspace_storage(128 * 1024);
    WorkspaceArena workspace(
        ninfer::DeviceSpan{workspace_storage.data(), workspace_storage.size()});

    projection_trace::reset();
    {
        projection_trace::MtpFixtureScope fixture;
        Variant::mtp_post_mixer(hidden, Variant::MtpPostMixerWeights{}, residual, workspace,
                                nullptr);
    }
    const projection_trace::Snapshot trace = projection_trace::snapshot();
    if (trace.calls != 0 || trace.mtp_projection_calls != 0 || trace.mtp_linear_calls != 2 ||
        trace.mtp_silu_calls != 1 || trace.mtp_residual_calls != 1) {
        std::cerr << "MTP writer fixture is not isolated: projection_calls=" << trace.calls
                  << " nested_projection_calls=" << trace.mtp_projection_calls
                  << " linear_calls=" << trace.mtp_linear_calls
                  << " silu_calls=" << trace.mtp_silu_calls
                  << " residual_calls=" << trace.mtp_residual_calls << '\n';
        return 1;
    }
    return 0;
}

int verify_projected_workspace() {
    int failures = 0;
    for (const TextPhase phase : {TextPhase::Prefill, TextPhase::Verify}) {
        for (const std::int32_t tokens : kTokens) {
            const std::size_t attention_base = ninfer::ops::linear_add_workspace_capacity_bytes(
                ninfer::QType::FP8_E4M3FN_ROW_BF16S, kHidden, kMixerInput,
                LinearPolicy::AllowA8, tokens, tokens);
            const std::size_t attention =
                Variant::attention_output_projection_workspace_capacity_bytes(
                    WeightsProfile::Qwen38Nvfp4, phase, tokens, tokens);
            if (attention != attention_base + score_bytes(tokens)) {
                std::cerr << "attention projection workspace mismatch at T=" << tokens
                          << " phase=" << static_cast<int>(phase) << ": got " << attention
                          << " expected " << attention_base + score_bytes(tokens) << '\n';
                ++failures;
            }

            const std::size_t gdn_base = ninfer::ops::linear_add_workspace_capacity_bytes(
                ninfer::QType::FP8_E4M3FN_ROW_BF16S, kHidden, kMixerInput,
                LinearPolicy::AllowA8, tokens, tokens);
            const std::size_t gdn = Variant::gdn_output_projection_workspace_capacity_bytes(
                WeightsProfile::Qwen38Nvfp4, phase, tokens, tokens);
            if (gdn != gdn_base + score_bytes(tokens)) {
                std::cerr << "GDN projection workspace mismatch at T=" << tokens
                          << " phase=" << static_cast<int>(phase) << ": got " << gdn
                          << " expected " << gdn_base + score_bytes(tokens) << '\n';
                ++failures;
            }

            const std::size_t mlp = Variant::post_mixer_workspace_capacity_bytes(
                WeightsProfile::Qwen38Nvfp4, phase, tokens, tokens);
            const std::size_t expected_mlp = std::max(
                expected_post_mixer_workspace(ninfer::QType::NVFP4, LinearPolicy::AllowA4,
                                              tokens, tokens, true),
                expected_post_mixer_workspace(ninfer::QType::FP8_E4M3FN_ROW_BF16S,
                                              LinearPolicy::AllowA8, tokens, tokens, true));
            if (mlp != expected_mlp) {
                std::cerr << "MLP projection workspace mismatch at T=" << tokens
                          << " phase=" << static_cast<int>(phase) << ": got " << mlp
                          << " expected " << expected_mlp << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

int verify_legacy_workspace_unchanged() {
    int failures = 0;
    for (const std::int32_t tokens : kTokens) {
        const std::size_t expected = ninfer::ops::linear_add_workspace_capacity_bytes(
            ninfer::QType::NVFP4, kHidden, kMixerInput, LinearPolicy::AllowA4, tokens, tokens);
        const std::size_t actual = Variant::attention_output_projection_workspace_capacity_bytes(
            WeightsProfile::Qwen36Nvfp4, TextPhase::Prefill, tokens, tokens);
        if (actual != expected) {
            std::cerr << "Qwen3.6 workspace changed at T=" << tokens << ": got " << actual
                      << " expected " << expected << '\n';
            ++failures;
        }
    }
    return failures;
}

bool real_integration_enabled() {
    const char* value = std::getenv("NINFER_RUN_QWEN38_PROJECTION_INTEGRATION");
    return value != nullptr && std::string_view(value) == "1";
}

std::filesystem::path configured_path(const char* environment, const char* fallback) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    return fallback;
}

ninfer::EngineOptions real_engine_options(bool use_cuda_graph) {
    ninfer::EngineOptions options;
    options.artifact_path = configured_path(
        "NINFER_QWEN3_8_27B_NVFP4_WEIGHTS",
        "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
        "qwen3_8_27b_nvfp4.ninfer");
    options.refusal_projection_path = configured_path(
        "NINFER_QWEN3_8_27B_PROJECTION",
        "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
        "qwen3_8_27b_refusal_projection.ninferproj");
    options.max_context               = 2048;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.kv_cache                  = ninfer::KvCacheStorage::Int8Group64;
    options.prefill_chunk             = 256;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision             = true;
    options.use_cuda_graph            = use_cuda_graph;
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t output_tokens, bool allow_prefix_reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = output_tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = allow_prefix_reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

bool configured_mtp3(const ninfer::GenerationResult& result) {
    return result.speculative.enabled &&
           result.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
           result.speculative.draft_window == 3;
}

bool used_mtp3(const ninfer::GenerationResult& result) {
    return configured_mtp3(result) && result.speculative.rounds != 0;
}

void print_tokens(std::string_view label, const std::vector<ninfer::TokenId>& tokens) {
    std::cerr << label << "=[";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) { std::cerr << ','; }
        std::cerr << tokens[index];
    }
    std::cerr << "]\n";
}

std::vector<std::uint8_t> gradient_ppm(int width = 64, int height = 64) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < width * height; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::PromptInput vision_prompt() {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "projection-integration.ppm";

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    message.parts.push_back(ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                                .text = "Describe this image briefly.",
                                                .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

int verify_real_projection_integration() {
    if (!real_integration_enabled()) {
        std::cout << "note: real Qwen3.8 projection integration was not requested\n";
        return 0;
    }

    const ninfer::EngineOptions graph_options = real_engine_options(true);
    if (!std::filesystem::is_regular_file(graph_options.artifact_path) ||
        !std::filesystem::is_regular_file(graph_options.refusal_projection_path)) {
        std::cerr << "real Qwen3.8 artifact or projection sidecar is missing\n";
        return 1;
    }

    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    std::vector<ninfer::TokenId> graph_tokens;
    std::vector<ninfer::TokenId> continuation;
    {
        ninfer::EngineOptions inventory_options = real_engine_options(false);
        inventory_options.speculative            = {};
        inventory_options.enable_vision          = false;
        ninfer::Engine inventory_engine(inventory_options);
        projection_trace::reset();
        const ninfer::GenerationResult inventory = inventory_engine.generate(
            inventory_engine.prepare_tokens(prompt), greedy_options(1, false));
        const projection_trace::Snapshot trace = projection_trace::snapshot();
        if (inventory.generated_token_ids.size() != 1 || inventory.speculative.enabled ||
            !projection_trace::exactly_one_backbone_traversal(trace)) {
            std::cerr << "isolated real backbone traversal is not exact: calls=" << trace.calls
                      << " invalid=" << trace.invalid << " duplicates=" << trace.duplicates
                      << " unique=" << trace.unique << " attention=" << trace.attention
                      << " gdn=" << trace.gdn << " mlp=" << trace.mlp << '\n';
            return 1;
        }
    }

    projection_trace::reset();
    {
        ninfer::Engine graph_engine(graph_options);
        const ninfer::MemorySummary memory = graph_engine.memory_summary();
        if (memory.cuda_graph_allowance_bytes == 0 ||
            memory.kv_cache != ninfer::KvCacheStorage::Int8Group64) {
            std::cerr << "graph-enabled Qwen3.8 engine has the wrong graph/KV profile\n";
            return 1;
        }

        projection_trace::reset();
        const ninfer::GenerationResult first = graph_engine.generate(
            graph_engine.prepare_tokens(prompt), greedy_options(6, false));
        if (first.generated_token_ids.size() != 6 || !used_mtp3(first)) {
            std::cerr << "projected graph/MTP request did not complete: outputs="
                      << first.generated_token_ids.size()
                      << " backend=" << static_cast<int>(first.speculative.backend)
                      << " window=" << first.speculative.draft_window
                      << " rounds=" << first.speculative.rounds << '\n';
            return 1;
        }
        graph_tokens = first.generated_token_ids;

        const projection_trace::Snapshot trace = projection_trace::snapshot();
        if (!projection_trace::contains_only_complete_backbone_traversals(trace)) {
            std::cerr << "real writer trace is incomplete: calls=" << trace.calls
                      << " invalid=" << trace.invalid << " duplicates=" << trace.duplicates
                      << " unique=" << trace.unique
                      << " attention=" << trace.attention << " gdn=" << trace.gdn
                      << " mlp=" << trace.mlp << '\n';
            return 1;
        }

        std::vector<ninfer::TokenId> exact_frontier = prompt;
        exact_frontier.insert(exact_frontier.end(), graph_tokens.begin(), graph_tokens.end() - 1);
        const ninfer::GenerationResult resumed = graph_engine.generate(
            graph_engine.prepare_tokens(exact_frontier), greedy_options(2, true));
        if (resumed.reused_prompt_tokens != exact_frontier.size() ||
            resumed.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier ||
            resumed.generated_token_ids.size() != 2 ||
            resumed.generated_token_ids.front() != graph_tokens.back() ||
            !configured_mtp3(resumed)) {
            std::cerr << "projected zero-suffix replay did not resume the retained frontier: "
                      << "reused=" << resumed.reused_prompt_tokens
                      << " expected=" << exact_frontier.size()
                      << " path=" << static_cast<int>(resumed.prefix_reuse_path)
                      << " backend=" << static_cast<int>(resumed.speculative.backend)
                      << " window=" << resumed.speculative.draft_window
                      << " rounds=" << resumed.speculative.rounds << '\n';
            print_tokens("source", graph_tokens);
            print_tokens("resumed", resumed.generated_token_ids);
            return 1;
        }

        continuation = prompt;
        continuation.insert(continuation.end(), graph_tokens.begin(), graph_tokens.end());
        continuation.push_back(198);
        const ninfer::GenerationResult reused = graph_engine.generate(
            graph_engine.prepare_tokens(continuation), greedy_options(4, true));
        const std::uint32_t expected_reuse =
            static_cast<std::uint32_t>(prompt.size() + graph_tokens.size());
        if (reused.reused_prompt_tokens != expected_reuse ||
            reused.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier ||
            reused.generated_token_ids.size() != 4 || !used_mtp3(reused)) {
            std::cerr << "projected prefix hit did not complete: reused="
                      << reused.reused_prompt_tokens << " expected=" << expected_reuse
                      << " hit_outputs=" << reused.generated_token_ids.size()
                      << " hit_path=" << static_cast<int>(reused.prefix_reuse_path)
                      << " hit_rounds=" << reused.speculative.rounds << '\n';
            print_tokens("source", graph_tokens);
            print_tokens("hit", reused.generated_token_ids);
            return 1;
        }

        const ninfer::GenerationResult vision = graph_engine.generate(
            graph_engine.prepare(vision_prompt()), greedy_options(5, false));
        if (!vision.prompt.has_media || vision.generated_token_ids.size() != 5 ||
            !used_mtp3(vision)) {
            std::cerr << "projected Vision/MTP request did not complete: media="
                      << vision.prompt.has_media << " outputs="
                      << vision.generated_token_ids.size()
                      << " rounds=" << vision.speculative.rounds << '\n';
            return 1;
        }
    }

    ninfer::Engine eager_engine(real_engine_options(false));
    const ninfer::MemorySummary eager_memory = eager_engine.memory_summary();
    if (eager_memory.kv_cache != ninfer::KvCacheStorage::Int8Group64) {
        std::cerr << "eager Qwen3.8 engine is not using INT8 group-64 KV\n";
        return 1;
    }
    const ninfer::GenerationResult eager =
        eager_engine.generate(eager_engine.prepare_tokens(prompt), greedy_options(6, false));
    if (eager.generated_token_ids != graph_tokens || !used_mtp3(eager)) {
        std::cerr << "projected eager and CUDA Graph greedy outputs diverged\n";
        print_tokens("graph", graph_tokens);
        print_tokens("eager", eager.generated_token_ids);
        return 1;
    }
    const ninfer::GenerationResult miss = eager_engine.generate(
        eager_engine.prepare_tokens(continuation), greedy_options(4, false));
    if (miss.reused_prompt_tokens != 0 ||
        miss.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        miss.generated_token_ids.size() != 4 || !used_mtp3(miss)) {
        std::cerr << "projected fresh prefix miss did not complete: reused="
                  << miss.reused_prompt_tokens
                  << " path=" << static_cast<int>(miss.prefix_reuse_path)
                  << " outputs=" << miss.generated_token_ids.size()
                  << " rounds=" << miss.speculative.rounds << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    int failures = 0;
    failures += verify_claim_inventory();
    failures += verify_writer_site_validation();
    failures += verify_mtp_writer_isolation();
    failures += verify_projected_workspace();
    failures += verify_legacy_workspace_unchanged();
    failures += verify_real_projection_integration();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.8 projection integration checks failed\n";
        return 1;
    }
    std::cout << "ok: 128 unique backbone claims, MTP isolated, projected workspace exact\n";
    return 0;
}
