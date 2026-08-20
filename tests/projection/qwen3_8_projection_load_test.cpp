#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <ninfer/projection/residual_projection.h>
#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {

using ninfer::ProjectionError;
using ninfer::artifact::Binder;
using ninfer::artifact::Reader;
using ninfer::targets::qwen3_6_27b::Package;

std::filesystem::path configured_path(const char* environment, const char* fallback) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    return fallback;
}

std::filesystem::path real_artifact_path() {
    return configured_path(
        "NINFER_QWEN3_8_27B_NVFP4_WEIGHTS",
        "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
        "qwen3_8_27b_nvfp4.ninfer");
}

std::filesystem::path real_sidecar_path() {
    return configured_path(
        "NINFER_QWEN3_8_27B_PROJECTION",
        "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
        "qwen3_8_27b_refusal_projection.ninferproj");
}

int verify_real_plan_load(const std::filesystem::path& artifact_path,
                          const std::filesystem::path& sidecar_path) {
    using Json = ninfer::test::artifact_fixture::Json;
    auto decoy_path = ninfer::test::artifact_fixture::write_fixture(
        {{"identity", {{"model_id", "qwen3.8-27b"}, {"weights_id", "nvfp4"}}},
         {"objects", Json::array({{{"name", "fixture/resource"},
                                    {"kind", "resource"},
                                    {"encoding", "raw-bytes-v1"},
                                    {"offset", 0},
                                    {"bytes", 1}}})}},
        "qwen38_decoy_path");
    Reader reader(artifact_path);
    const auto profile = Package::resolve_weights(reader.identity());
    if (profile != Package::WeightsProfile::Qwen38Nvfp4) {
        std::cerr << "approved artifact resolved to the wrong weights profile\n";
        return 1;
    }
    Binder binder(reader);
    ninfer::EngineOptions options;
    options.artifact_path = decoy_path.path;
    options.refusal_projection_path = sidecar_path;
    auto plan = Package::plan_load(binder, options, profile);
    if (plan.materialization().object_count != 1124 ||
        plan.materialization().device_capacity_bytes == 0) {
        std::cerr << "real Qwen3.8 plan_load produced an incomplete materialization plan\n";
        return 1;
    }
    const auto before_move = plan.refusal_projection(3, ninfer::ProjectionSite::AttentionOutput);
    auto moved = std::move(plan);
    const auto after_move = moved.refusal_projection(3, ninfer::ProjectionSite::AttentionOutput);
    if (before_move.direction != after_move.direction ||
        before_move.signature != after_move.signature ||
        after_move.direction_count != 5120 || after_move.signature_count != 6144) {
        std::cerr << "Package::LoadPlan did not retain stable projection ownership\n";
        return 1;
    }
    return 0;
}

int verify_changed_artifact_rejected(const std::filesystem::path& sidecar_path) {
    using Json = ninfer::test::artifact_fixture::Json;
    auto changed = ninfer::test::artifact_fixture::write_fixture(
        {{"identity", {{"model_id", "qwen3.8-27b"}, {"weights_id", "nvfp4"}}},
         {"objects", Json::array({{{"name", "fixture/resource"},
                                    {"kind", "resource"},
                                    {"encoding", "raw-bytes-v1"},
                                    {"offset", 0},
                                    {"bytes", 1}}})}},
        "changed_qwen38");
    Reader reader(changed.path);
    Binder binder(reader);
    ninfer::EngineOptions options;
    options.artifact_path = changed.path;
    options.refusal_projection_path = sidecar_path;
    try {
        (void)Package::plan_load(binder, options, Package::WeightsProfile::Qwen38Nvfp4);
    } catch (const ProjectionError& error) {
        if (std::string_view(error.what()).find("artifact SHA-256") != std::string_view::npos) {
            return 0;
        }
        std::cerr << "changed artifact failed for the wrong projection reason: " << error.what()
                  << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "changed artifact reached binding before authentication: " << error.what()
                  << '\n';
        return 1;
    }
    std::cerr << "changed artifact bytes were accepted\n";
    return 1;
}

int verify_conversion_report_required(const std::filesystem::path& sidecar_path) {
    const std::filesystem::path temporary =
        std::filesystem::temp_directory_path() / "ninfer_projection_report_identity";
    std::filesystem::create_directories(temporary);
    const std::filesystem::path linked_sidecar = temporary / "projection.ninferproj";
    const std::filesystem::path forged_report = temporary / "projection.conversion.json";
    std::error_code ignored;
    std::filesystem::remove(linked_sidecar, ignored);
    std::filesystem::create_symlink(sidecar_path, linked_sidecar);
    {
        std::ofstream output(forged_report, std::ios::binary | std::ios::trunc);
        output << "{\"schema_version\":1}";
    }
    int result = 1;
    try {
        (void)ninfer::ResidualProjectionTable::load(
            linked_sidecar,
            "bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32");
        std::cerr << "forged conversion report was accepted\n";
    } catch (const ProjectionError& error) {
        if (std::string_view(error.what()).find("conversion report") != std::string_view::npos) {
            result = 0;
        } else {
            std::cerr << "forged report failed for the wrong reason: " << error.what() << '\n';
        }
    }
    std::filesystem::remove_all(temporary, ignored);
    return result;
}

} // namespace

int main() {
    const std::filesystem::path artifact = real_artifact_path();
    const std::filesystem::path sidecar = real_sidecar_path();
    if (!std::filesystem::is_regular_file(artifact) ||
        !std::filesystem::is_regular_file(sidecar)) {
        std::cerr << "skip: approved Qwen3.8 artifact and sidecar are required\n";
        return 77;
    }
    int failures = 0;
    failures += verify_real_plan_load(artifact, sidecar);
    failures += verify_changed_artifact_rejected(sidecar);
    failures += verify_conversion_report_required(sidecar);
    return failures == 0 ? 0 : 1;
}
