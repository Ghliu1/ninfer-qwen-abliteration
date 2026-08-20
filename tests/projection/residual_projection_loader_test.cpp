#include <ninfer/projection/residual_projection.h>

#include "../../apps/cli/options.h"
#include "serve/serve_options.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using ninfer::ProjectionError;
using ninfer::ProjectionSite;
using ninfer::ResidualProjectionTable;
using ninfer::targets::qwen3_6_27b::detail::WeightsProfile;

constexpr std::string_view kModelSha =
    "bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32";

int check(bool condition, std::string_view message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

template <class Function>
int expect_projection_error(Function&& function, std::string_view label) {
    try {
        std::forward<Function>(function)();
    } catch (const ProjectionError&) {
        return 0;
    } catch (const std::exception& error) {
        std::cerr << label << " threw the wrong exception: " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << " was accepted\n";
    return 1;
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        path_ = std::filesystem::temp_directory_path() /
                ("ninfer-residual-projection-" + std::to_string(std::rand()));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

std::filesystem::path make_fixture(const TemporaryDirectory& temporary, std::string_view mode) {
    const std::filesystem::path output = temporary.path() / (std::string(mode) + ".ninferproj");
    const std::filesystem::path maker =
        std::filesystem::path(NINFER_SOURCE_DIR) /
        "tests/projection/make_residual_projection_fixture.py";
    const std::string command = std::string("PYTHONPATH=\"") + NINFER_SOURCE_DIR + "\" \"" +
                                NINFER_PYTHON_EXECUTABLE + "\" \"" + maker.string() +
                                "\" \"" + output.string() + "\" " + std::string(mode);
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("projection fixture generator failed for " + std::string(mode));
    }
    return output;
}

std::vector<char*> argv(std::vector<std::string>& arguments) {
    std::vector<char*> result;
    result.reserve(arguments.size());
    for (std::string& argument : arguments) { result.push_back(argument.data()); }
    return result;
}

int verify_valid_fixture(const TemporaryDirectory& temporary,
                         const std::filesystem::path& valid) {
    int failures = 0;
    ResidualProjectionTable table = ResidualProjectionTable::load(valid, kModelSha);
    table.verify_all_claimed();

    std::vector<const float*> addresses;
    addresses.reserve(256);
    std::size_t record_index = 0;
    for (std::uint32_t layer = 0; layer < 64; ++layer) {
        const bool attention = layer % 4 == 3;
        const ProjectionSite mixer =
            attention ? ProjectionSite::AttentionOutput : ProjectionSite::GdnOutput;
        const auto mixer_view = table.view(layer, mixer);
        failures += check(mixer_view.direction_count == 5120 &&
                              mixer_view.signature_count == 6144 &&
                              mixer_view.coefficient == static_cast<float>(record_index) + 0.5F,
                          "mixer projection view does not match the exact claim map");
        addresses.push_back(mixer_view.direction);
        addresses.push_back(mixer_view.signature);
        ++record_index;
        const auto mlp_view = table.view(layer, ProjectionSite::MlpDown);
        failures += check(mlp_view.direction_count == 5120 &&
                              mlp_view.signature_count == 17408 &&
                              mlp_view.coefficient == static_cast<float>(record_index) + 0.5F,
                          "MLP projection view does not match the exact claim map");
        addresses.push_back(mlp_view.direction);
        addresses.push_back(mlp_view.signature);
        ++record_index;
        failures += expect_projection_error(
            [&] {
                (void)table.view(layer, attention ? ProjectionSite::GdnOutput
                                                  : ProjectionSite::AttentionOutput);
            },
            "unclaimed mixer site");
    }
    failures += expect_projection_error(
        [&] { (void)table.view(64, ProjectionSite::MlpDown); }, "out-of-range layer");

    const auto layer_three = table.view(3, ProjectionSite::AttentionOutput);
    float direction_first = 0.0F;
    float signature_first = 0.0F;
    if (cudaMemcpy(&direction_first, layer_three.direction, sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(&signature_first, layer_three.signature, sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::cerr << "projection payload was not uploaded to device memory\n";
        ++failures;
    }
    failures += check(direction_first == 1.0F && signature_first == 6.25F,
                      "device payload does not match fixture bytes");

    ResidualProjectionTable moved(std::move(table));
    record_index = 0;
    for (std::uint32_t layer = 0; layer < 64; ++layer) {
        const ProjectionSite mixer = layer % 4 == 3 ? ProjectionSite::AttentionOutput
                                                    : ProjectionSite::GdnOutput;
        const auto mixer_view = moved.view(layer, mixer);
        failures += check(mixer_view.direction == addresses[record_index * 2] &&
                              mixer_view.signature == addresses[record_index * 2 + 1],
                          "device addresses changed when projection ownership moved");
        ++record_index;
        const auto mlp_view = moved.view(layer, ProjectionSite::MlpDown);
        failures += check(mlp_view.direction == addresses[record_index * 2] &&
                              mlp_view.signature == addresses[record_index * 2 + 1],
                          "MLP device addresses changed when projection ownership moved");
        ++record_index;
    }

    using ninfer::targets::qwen3_6_27b::detail::load_refusal_projection;
    using ninfer::targets::qwen3_6_27b::detail::qwen38_projection_required_by_build;
    auto bound = load_refusal_projection(WeightsProfile::Qwen38Nvfp4, valid);
    failures += check(bound.has_value(), "Qwen3.8/NVFP4 did not own the supplied sidecar");
    failures += expect_projection_error(
        [&] { (void)load_refusal_projection(WeightsProfile::Qwen38GroupwiseInt, valid); },
        "Qwen3.8 groupwise projection path");
    failures += expect_projection_error(
        [&] { (void)load_refusal_projection(WeightsProfile::Qwen36Nvfp4, valid); },
        "Qwen3.6 projection path");
    if (qwen38_projection_required_by_build()) {
        failures += expect_projection_error(
            [&] { (void)load_refusal_projection(WeightsProfile::Qwen38Nvfp4, {}); },
            "required Qwen3.8 projection omission");
    } else {
        failures += check(!load_refusal_projection(WeightsProfile::Qwen38Nvfp4, {}).has_value(),
                          "baseline build did not allow an omitted projection");
    }

    std::vector<std::string> cli_arguments{
        "ninfer", "model.ninfer", "--prompt", "hello", "--refusal-projection", valid.string()};
    auto cli_argv = argv(cli_arguments);
    const auto cli = ninfer::cli::parse_options(static_cast<int>(cli_argv.size()), cli_argv.data());
    failures += check(cli.refusal_projection_path == valid,
                      "CLI projection path did not reach parsed options");
    failures += check(ninfer::cli::usage_text("ninfer").find("--refusal-projection") !=
                          std::string::npos,
                      "CLI help omits --refusal-projection");

    std::vector<std::string> serve_arguments{
        "ninfer-serve", "model.ninfer", "--refusal-projection", valid.string()};
    auto serve_argv = argv(serve_arguments);
    const auto serve = ninfer::serve::parse_serve_options(static_cast<int>(serve_argv.size()),
                                                           serve_argv.data());
    failures += check(serve.refusal_projection_path == valid,
                      "server projection path did not reach parsed options");
    failures += check(ninfer::serve::serve_usage_text("ninfer-serve").find(
                          "--refusal-projection") != std::string::npos,
                      "server help omits --refusal-projection");
    return failures;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "skip: CUDA device is required\n";
        return 77;
    }

    TemporaryDirectory temporary;
    const std::filesystem::path valid = make_fixture(temporary, "valid");
    int failures = verify_valid_fixture(temporary, valid);
    for (const std::string_view mode : {
             "missing",          "lambda",          "override",       "duplicate",
             "wrong_layer",      "wrong_site",      "wrong_order",    "overlap",
             "unaligned",        "out_of_range",    "wrong_count",    "wrong_format",
             "wrong_artifact_hash", "wrong_direction_hash", "bad_manifest_hash",
             "wrong_version",    "bad_magic",       "oversized_manifest",
             "oversized_payload",
         }) {
        const std::filesystem::path malformed = make_fixture(temporary, mode);
        failures += expect_projection_error(
            [&] { (void)ResidualProjectionTable::load(malformed, kModelSha); }, mode);
        std::filesystem::remove(malformed);
    }
    failures += expect_projection_error(
        [&] { (void)ResidualProjectionTable::load(valid, std::string(64, '0')); },
        "mismatched expected model SHA-256");
    failures += expect_projection_error(
        [&] { (void)ResidualProjectionTable::load(valid, "not-a-sha256"); },
        "malformed expected model SHA-256");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
