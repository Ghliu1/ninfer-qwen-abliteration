#include <ninfer/projection/residual_projection.h>

#include "../../apps/cli/options.h"
#include "artifact/reader.h"
#include "serve/serve_options.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace projection_open_swap {

std::array<char, PATH_MAX> watched_path{};
std::array<char, PATH_MAX> replacement_link{};
std::atomic<bool> armed{false};
std::atomic<bool> fired{false};
std::atomic<int> swap_error{0};

void arm(const std::filesystem::path& watched, const std::filesystem::path& replacement) {
    const std::string watched_text = watched.string();
    const std::string replacement_text = replacement.string();
    if (watched_text.size() >= watched_path.size() ||
        replacement_text.size() >= replacement_link.size()) {
        throw std::runtime_error("projection swap test path exceeds PATH_MAX");
    }
    std::fill(watched_path.begin(), watched_path.end(), '\0');
    std::fill(replacement_link.begin(), replacement_link.end(), '\0');
    std::copy(watched_text.begin(), watched_text.end(), watched_path.begin());
    std::copy(replacement_text.begin(), replacement_text.end(), replacement_link.begin());
    swap_error.store(0);
    fired.store(false);
    armed.store(true);
}

void after_open(const char* pathname, int descriptor) noexcept {
    if (descriptor < 0 || pathname == nullptr || !armed.load() ||
        std::strcmp(pathname, watched_path.data()) != 0 || !armed.exchange(false)) {
        return;
    }
    if (::rename(replacement_link.data(), watched_path.data()) != 0) {
        swap_error.store(errno);
    }
    fired.store(true);
}

int raw_open(const char* pathname, int flags, mode_t mode) noexcept {
    const int descriptor = static_cast<int>(
        ::syscall(SYS_openat, AT_FDCWD, pathname, flags, mode));
    after_open(pathname, descriptor);
    return descriptor;
}

} // namespace projection_open_swap

extern "C" int open(const char* pathname, int flags, ...) {
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        va_list arguments;
        va_start(arguments, flags);
        mode = static_cast<mode_t>(va_arg(arguments, int));
        va_end(arguments);
    }
    return projection_open_swap::raw_open(pathname, flags, mode);
}

extern "C" int open64(const char* pathname, int flags, ...) {
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        va_list arguments;
        va_start(arguments, flags);
        mode = static_cast<mode_t>(va_arg(arguments, int));
        va_end(arguments);
    }
    return projection_open_swap::raw_open(pathname, flags, mode);
}

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

std::filesystem::path real_sidecar_path() {
    if (const char* value = std::getenv("NINFER_QWEN3_8_27B_PROJECTION");
        value != nullptr && *value != '\0') {
        return value;
    }
    return "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
           "qwen3_8_27b_refusal_projection.ninferproj";
}

std::filesystem::path real_artifact_path() {
    if (const char* value = std::getenv("NINFER_QWEN3_8_27B_NVFP4_WEIGHTS");
        value != nullptr && *value != '\0') {
        return value;
    }
    return "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
           "qwen3_8_27b_nvfp4.ninfer";
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

template <class Function>
int expect_projection_error_containing(Function&& function, std::string_view label,
                                       std::string_view expected) {
    try {
        std::forward<Function>(function)();
    } catch (const ProjectionError& error) {
        if (std::string_view(error.what()).find(expected) != std::string_view::npos) { return 0; }
        std::cerr << label << " failed for the wrong reason: " << error.what() << '\n';
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

std::filesystem::path conversion_report_path(std::filesystem::path sidecar) {
    sidecar.replace_extension(".conversion.json");
    return sidecar;
}

int verify_swap_fired(std::string_view label) {
    if (!projection_open_swap::fired.load()) {
        std::cerr << label << " did not intercept the intended file open\n";
        return 1;
    }
    if (projection_open_swap::swap_error.load() != 0) {
        std::cerr << label << " could not replace the symlink: "
                  << std::strerror(projection_open_swap::swap_error.load()) << '\n';
        return 1;
    }
    return 0;
}

int verify_sidecar_symlink_swap_rejected(const TemporaryDirectory& temporary,
                                         const std::filesystem::path& approved_sidecar) {
    const std::filesystem::path consumed_sidecar = make_fixture(temporary, "valid");
    const std::filesystem::path attack_path = temporary.path() / "sidecar_swap.ninferproj";
    const std::filesystem::path replacement = temporary.path() / "sidecar_swap.next";
    std::filesystem::create_symlink(consumed_sidecar, attack_path);
    std::filesystem::create_symlink(approved_sidecar, replacement);
    std::filesystem::create_symlink(conversion_report_path(approved_sidecar),
                                    conversion_report_path(attack_path));
    projection_open_swap::arm(attack_path, replacement);
    int failures = expect_projection_error_containing(
        [&] { (void)ResidualProjectionTable::load(attack_path, kModelSha); },
        "sidecar symlink replacement", "approved sidecar");
    failures += verify_swap_fired("sidecar symlink replacement");
    return failures;
}

int verify_report_symlink_swap_uses_opened_bytes(const TemporaryDirectory& temporary,
                                                 const std::filesystem::path& approved_sidecar) {
    const std::filesystem::path attack_sidecar = temporary.path() / "report_swap.ninferproj";
    const std::filesystem::path attack_report = conversion_report_path(attack_sidecar);
    const std::filesystem::path replacement = temporary.path() / "report_swap.next";
    const std::filesystem::path forged_report = temporary.path() / "forged_report.json";
    const std::filesystem::path approved_report = conversion_report_path(approved_sidecar);
    std::filesystem::create_symlink(approved_sidecar, attack_sidecar);
    std::filesystem::create_symlink(approved_report, attack_report);
    {
        const auto bytes = std::filesystem::file_size(approved_report);
        std::string forged(static_cast<std::size_t>(bytes), ' ');
        constexpr std::string_view invalid = R"({"schema_version":1})";
        std::copy(invalid.begin(), invalid.end(), forged.begin());
        std::ofstream output(forged_report, std::ios::binary | std::ios::trunc);
        output.write(forged.data(), static_cast<std::streamsize>(forged.size()));
    }
    std::filesystem::create_symlink(forged_report, replacement);
    projection_open_swap::arm(attack_report, replacement);
    int failures = 0;
    try {
        auto table = ResidualProjectionTable::load(attack_sidecar, kModelSha);
        const auto view = table.view(3, ProjectionSite::AttentionOutput);
        failures += check(view.direction_count == 5120 && view.signature_count == 6144,
                          "opened conversion report did not retain the approved identity");
    } catch (const std::exception& error) {
        std::cerr << "report symlink replacement redirected consumed bytes: " << error.what()
                  << '\n';
        ++failures;
    }
    failures += verify_swap_fired("report symlink replacement");
    return failures;
}

std::vector<char*> argv(std::vector<std::string>& arguments) {
    std::vector<char*> result;
    result.reserve(arguments.size());
    for (std::string& argument : arguments) { result.push_back(argument.data()); }
    return result;
}

int verify_valid_fixture(const std::filesystem::path& valid,
                         const ninfer::artifact::Reader& artifact) {
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
                              std::isfinite(mixer_view.coefficient),
                          "mixer projection view does not match the exact claim map");
        addresses.push_back(mixer_view.direction);
        addresses.push_back(mixer_view.signature);
        ++record_index;
        const auto mlp_view = table.view(layer, ProjectionSite::MlpDown);
        failures += check(mlp_view.direction_count == 5120 &&
                              mlp_view.signature_count == 17408 &&
                              std::isfinite(mlp_view.coefficient),
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
    if (cudaMemcpy(&direction_first, layer_three.direction, sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::cerr << "projection payload was not uploaded to device memory\n";
        ++failures;
    }
    failures += check(std::isfinite(direction_first), "device direction payload is not finite");

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
    failures += expect_projection_error(
        [&] { (void)load_refusal_projection(WeightsProfile::Qwen38GroupwiseInt, valid, artifact); },
        "Qwen3.8 groupwise projection path");
    failures += expect_projection_error(
        [&] { (void)load_refusal_projection(WeightsProfile::Qwen36Nvfp4, valid, artifact); },
        "Qwen3.6 projection path");
    if (qwen38_projection_required_by_build()) {
        failures += expect_projection_error(
            [&] { (void)load_refusal_projection(WeightsProfile::Qwen38Nvfp4, {}, artifact); },
            "required Qwen3.8 projection omission");
    } else {
        failures += check(
            !load_refusal_projection(WeightsProfile::Qwen38Nvfp4, {}, artifact).has_value(),
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

    const std::filesystem::path valid = real_sidecar_path();
    const std::filesystem::path artifact_path = real_artifact_path();
    if (!std::filesystem::is_regular_file(valid) ||
        !std::filesystem::is_regular_file(artifact_path)) {
        std::cerr << "skip: approved Qwen3.8 artifact and projection sidecar are required\n";
        return 77;
    }
    ninfer::artifact::Reader artifact(artifact_path);
    TemporaryDirectory temporary;
    int failures = verify_valid_fixture(valid, artifact);
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
    for (const std::string_view mode : {
             "finite_direction_mutation",
             "finite_signature_mutation",
             "arbitrary_weight_hash",
         }) {
        const std::filesystem::path corrupted = make_fixture(temporary, mode);
        failures += expect_projection_error_containing(
            [&] { (void)ResidualProjectionTable::load(corrupted, kModelSha); }, mode,
            "approved sidecar");
    }
    for (const std::string_view mode : {"zero_direction", "nonunit_direction"}) {
        const std::filesystem::path corrupted = make_fixture(temporary, mode);
        failures += expect_projection_error_containing(
            [&] { (void)ResidualProjectionTable::load(corrupted, kModelSha); }, mode,
            "unit length");
    }
    const std::filesystem::path literal_duplicate =
        make_fixture(temporary, "literal_duplicate_key");
    failures += expect_projection_error_containing(
        [&] { (void)ResidualProjectionTable::load(literal_duplicate, kModelSha); },
        "literal duplicate JSON key", "duplicate object key");
    failures += expect_projection_error(
        [&] { (void)ResidualProjectionTable::load(valid, std::string(64, '0')); },
        "mismatched expected model SHA-256");
    failures += expect_projection_error(
        [&] { (void)ResidualProjectionTable::load(valid, "not-a-sha256"); },
        "malformed expected model SHA-256");
    failures += verify_sidecar_symlink_swap_rejected(temporary, valid);
    failures += verify_report_symlink_swap_uses_opened_bytes(temporary, valid);

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
