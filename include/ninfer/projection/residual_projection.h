#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace ninfer {

class ProjectionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class ProjectionSite : std::uint8_t {
    AttentionOutput,
    GdnOutput,
    MlpDown,
};

struct ResidualProjectionView {
    const float* direction = nullptr;
    const float* signature = nullptr;
    float coefficient      = 0.0F;
    std::uint32_t direction_count = 0;
    std::uint32_t signature_count = 0;
};

class ResidualProjectionTable {
public:
    static ResidualProjectionTable load(const std::filesystem::path& path,
                                        std::string_view expected_model_sha256);

    ~ResidualProjectionTable();
    ResidualProjectionTable(ResidualProjectionTable&&) noexcept;
    ResidualProjectionTable& operator=(ResidualProjectionTable&&) noexcept;

    ResidualProjectionTable(const ResidualProjectionTable&)            = delete;
    ResidualProjectionTable& operator=(const ResidualProjectionTable&) = delete;

    [[nodiscard]] ResidualProjectionView view(std::uint32_t layer,
                                              ProjectionSite site) const;
    void verify_all_claimed() const;

private:
    class Impl;
    explicit ResidualProjectionTable(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
