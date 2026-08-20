#include <ninfer/projection/residual_projection.h>

#include "core/arena.h"
#include "projection/sha256.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kHeaderBytes       = 124;
constexpr std::uint32_t kSchemaVersion   = 1;
constexpr std::uint64_t kPayloadAlignment = 4096;
constexpr std::uint64_t kMaximumManifestBytes = 1ULL << 20;
constexpr std::uint64_t kMaximumPayloadBytes  = 16ULL << 20;
constexpr std::uint64_t kMaximumReportBytes   = 64ULL << 10;
constexpr std::size_t kLayers            = 64;
constexpr std::size_t kWriterCount       = 128;
constexpr std::uint32_t kDirectionCount  = 5120;
constexpr std::string_view kApprovedDirectionSha256 =
    "9de12cbe71f38baf2f6b4a21dfcb2b13bd6416ab4785214afce27c7543f05c1d";
constexpr std::string_view kApprovedSidecarSha256 =
    "1b536e5bbab61df1fc4bc3b9347818282327f47c887494e572d89740cc4d395e";
constexpr std::string_view kApprovedReportSha256 =
    "77e66b40b88ba4f0d74aa7350f6bb3d25cc366dcb4e750cd822c1bfde65e78cd";
constexpr std::array<char, 8> kMagic{'N', 'I', 'N', 'F', 'R', 'P', '1', '\0'};
constexpr std::array<std::string_view, 10> kRecordFields{
    "coefficient",          "direction_count",      "direction_offset",
    "key",                  "layer",                "signature_count",
    "signature_offset",     "site",                 "weight_format",
    "weight_payload_sha256",
};

using Digest = projection_internal::Sha256Digest;
using projection_internal::sha256;

struct Header {
    std::uint32_t version = 0;
    std::uint64_t manifest_bytes = 0;
    std::uint64_t payload_bytes  = 0;
    Digest manifest_sha{};
    Digest artifact_sha{};
    Digest direction_sha{};
};

struct ParsedRecord {
    std::uint32_t layer = 0;
    ProjectionSite site = ProjectionSite::AttentionOutput;
    float coefficient = 0.0F;
    std::uint64_t direction_offset = 0;
    std::uint32_t direction_count = 0;
    std::uint64_t signature_offset = 0;
    std::uint32_t signature_count = 0;
};

struct PayloadRange {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

std::uint32_t load_le32(const std::uint8_t* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint64_t load_le64(const std::uint8_t* bytes) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (8U * index);
    }
    return value;
}

std::uint8_t hex_nibble(char value) {
    if (value >= '0' && value <= '9') { return static_cast<std::uint8_t>(value - '0'); }
    if (value >= 'a' && value <= 'f') { return static_cast<std::uint8_t>(value - 'a' + 10); }
    if (value >= 'A' && value <= 'F') { return static_cast<std::uint8_t>(value - 'A' + 10); }
    throw ProjectionError("projection SHA-256 is not hexadecimal");
}

Digest parse_digest(std::string_view value) {
    if (value.size() != 64) {
        throw ProjectionError("projection SHA-256 must contain exactly 64 hexadecimal characters");
    }
    Digest digest{};
    for (std::size_t index = 0; index < digest.size(); ++index) {
        digest[index] = static_cast<std::uint8_t>((hex_nibble(value[2 * index]) << 4U) |
                                                  hex_nibble(value[2 * index + 1]));
    }
    return digest;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, const char* label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw ProjectionError(std::string(label) + " overflows u64");
    }
    return left + right;
}

std::uint64_t align_up(std::uint64_t value) {
    const std::uint64_t mask = kPayloadAlignment - 1;
    return checked_add(value, mask, "projection alignment") & ~mask;
}

Header parse_header(const std::array<std::uint8_t, kHeaderBytes>& bytes) {
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        throw ProjectionError("projection sidecar magic is not NINFRP1");
    }
    Header header;
    header.version        = load_le32(bytes.data() + 8);
    header.manifest_bytes = load_le64(bytes.data() + 12);
    header.payload_bytes  = load_le64(bytes.data() + 20);
    std::copy_n(bytes.data() + 28, 32, header.manifest_sha.begin());
    std::copy_n(bytes.data() + 60, 32, header.artifact_sha.begin());
    std::copy_n(bytes.data() + 92, 32, header.direction_sha.begin());
    return header;
}

std::string expected_key(std::uint32_t layer, ProjectionSite site) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    switch (site) {
    case ProjectionSite::AttentionOutput:
        return prefix + "self_attn.o_proj";
    case ProjectionSite::GdnOutput:
        return prefix + "linear_attn.out_proj";
    case ProjectionSite::MlpDown:
        return prefix + "mlp.down_proj";
    }
    throw ProjectionError("projection site is invalid");
}

std::string_view site_name(ProjectionSite site) {
    switch (site) {
    case ProjectionSite::AttentionOutput:
        return "attention_output";
    case ProjectionSite::GdnOutput:
        return "gdn_output";
    case ProjectionSite::MlpDown:
        return "mlp_down";
    }
    throw ProjectionError("projection site is invalid");
}

std::size_t lookup_index(std::uint32_t layer, ProjectionSite site) {
    if (layer >= kLayers) { throw ProjectionError("projection layer is outside [0,63]"); }
    std::size_t site_index = 0;
    switch (site) {
    case ProjectionSite::AttentionOutput:
        site_index = 0;
        break;
    case ProjectionSite::GdnOutput:
        site_index = 1;
        break;
    case ProjectionSite::MlpDown:
        site_index = 2;
        break;
    default:
        throw ProjectionError("projection site is invalid");
    }
    return static_cast<std::size_t>(layer) * 3 + site_index;
}

const Json& member(const Json& record, std::string_view name) {
    const auto found = record.find(std::string(name));
    if (found == record.end()) {
        throw ProjectionError("projection record is missing member " + std::string(name));
    }
    return *found;
}

std::string string_member(const Json& record, std::string_view name) {
    const Json& value = member(record, name);
    if (!value.is_string()) {
        throw ProjectionError("projection record member " + std::string(name) + " must be a string");
    }
    return value.get<std::string>();
}

std::uint64_t unsigned_member(const Json& record, std::string_view name) {
    const Json& value = member(record, name);
    if (!value.is_number_unsigned()) {
        throw ProjectionError("projection record member " + std::string(name) +
                              " must be a nonnegative integer");
    }
    return value.get<std::uint64_t>();
}

std::uint32_t narrow_u32(std::uint64_t value, std::string_view name) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw ProjectionError("projection record member " + std::string(name) + " exceeds u32");
    }
    return static_cast<std::uint32_t>(value);
}

float coefficient_member(const Json& record) {
    const Json& value = member(record, "coefficient");
    if (!value.is_number()) { throw ProjectionError("projection coefficient must be numeric"); }
    const double coefficient = value.get<double>();
    if (!std::isfinite(coefficient) ||
        std::abs(coefficient) > static_cast<double>(std::numeric_limits<float>::max())) {
        throw ProjectionError("projection coefficient must be finite FP32");
    }
    return static_cast<float>(coefficient);
}

void validate_record_fields(const Json& record) {
    if (!record.is_object() || record.size() != kRecordFields.size()) {
        throw ProjectionError("projection record has missing or extra members");
    }
    std::set<std::string_view> expected(kRecordFields.begin(), kRecordFields.end());
    for (const auto& [name, unused] : record.items()) {
        (void)unused;
        if (!expected.erase(name)) {
            throw ProjectionError("projection record has unknown member " + name);
        }
    }
    if (!expected.empty()) { throw ProjectionError("projection record has missing members"); }
}

PayloadRange validate_range(std::uint64_t offset, std::uint32_t count,
                            std::uint64_t payload_bytes, std::string_view label) {
    if (offset % kPayloadAlignment != 0) {
        throw ProjectionError(std::string(label) + " offset is not 4096-byte aligned");
    }
    const std::uint64_t bytes = static_cast<std::uint64_t>(count) * sizeof(float);
    const std::uint64_t end = checked_add(offset, bytes, "projection payload span");
    if (end > payload_bytes) {
        throw ProjectionError(std::string(label) + " extends beyond the projection payload");
    }
    return PayloadRange{offset, end};
}

Json parse_manifest(const std::string& manifest) {
    std::vector<std::unordered_set<std::string>> object_keys;
    const Json::parser_callback_t reject_duplicates =
        [&](int, Json::parse_event_t event, Json& parsed) {
            if (event == Json::parse_event_t::object_start) {
                object_keys.emplace_back();
            } else if (event == Json::parse_event_t::key) {
                if (object_keys.empty() || !object_keys.back().insert(parsed.get<std::string>()).second) {
                    throw ProjectionError("projection manifest contains a duplicate object key");
                }
            } else if (event == Json::parse_event_t::object_end) {
                if (object_keys.empty()) {
                    throw ProjectionError("projection manifest object nesting is invalid");
                }
                object_keys.pop_back();
            }
            return true;
        };
    try {
        return Json::parse(manifest, reject_duplicates, true, false);
    } catch (const ProjectionError&) {
        throw;
    } catch (const Json::exception& error) {
        throw ProjectionError("invalid projection manifest JSON: " + std::string(error.what()));
    }
}

std::vector<ParsedRecord> validate_manifest(const Json& manifest, std::uint64_t payload_bytes) {
    if (!manifest.is_array()) { throw ProjectionError("projection manifest must be an array"); }
    if (manifest.size() != kWriterCount) {
        throw ProjectionError("projection manifest must contain exactly 128 writers");
    }
    std::vector<ParsedRecord> records;
    records.reserve(kWriterCount);
    std::vector<PayloadRange> ranges;
    ranges.reserve(kWriterCount * 2);
    std::unordered_set<std::string> keys;
    keys.reserve(kWriterCount);
    for (std::size_t index = 0; index < manifest.size(); ++index) {
        const Json& value = manifest[index];
        validate_record_fields(value);
        const std::uint32_t layer = narrow_u32(unsigned_member(value, "layer"), "layer");
        if (layer >= kLayers) { throw ProjectionError("projection record layer is outside [0,63]"); }
        const ProjectionSite site = index % 2 == 1
                                        ? ProjectionSite::MlpDown
                                        : (layer % 4 == 3 ? ProjectionSite::AttentionOutput
                                                          : ProjectionSite::GdnOutput);
        const std::string key = string_member(value, "key");
        if (!keys.insert(key).second) { throw ProjectionError("duplicate projection writer claim"); }
        if (key != expected_key(layer, site) || layer != index / 2) {
            throw ProjectionError("projection writer inventory order or key is invalid");
        }
        if (string_member(value, "site") != site_name(site)) {
            throw ProjectionError("projection writer site does not match its key");
        }
        const std::uint32_t direction_count =
            narrow_u32(unsigned_member(value, "direction_count"), "direction_count");
        const std::uint32_t signature_count =
            narrow_u32(unsigned_member(value, "signature_count"), "signature_count");
        if (direction_count != kDirectionCount) {
            throw ProjectionError("projection direction_count must be 5120");
        }
        const std::uint32_t expected_signature_count =
            site == ProjectionSite::MlpDown ? 17408 : 6144;
        if (signature_count != expected_signature_count) {
            throw ProjectionError("projection signature_count does not match its site");
        }
        const std::string expected_format =
            site != ProjectionSite::MlpDown || layer >= 56 ? "fp8_row" : "nvfp4_block";
        if (string_member(value, "weight_format") != expected_format) {
            throw ProjectionError("projection weight_format does not match the fixed writer");
        }
        (void)parse_digest(string_member(value, "weight_payload_sha256"));
        ParsedRecord record{
            .layer = layer,
            .site = site,
            .coefficient = coefficient_member(value),
            .direction_offset = unsigned_member(value, "direction_offset"),
            .direction_count = direction_count,
            .signature_offset = unsigned_member(value, "signature_offset"),
            .signature_count = signature_count,
        };
        ranges.push_back(validate_range(record.direction_offset, record.direction_count,
                                        payload_bytes, "projection direction"));
        ranges.push_back(validate_range(record.signature_offset, record.signature_count,
                                        payload_bytes, "projection signature"));
        records.push_back(record);
    }
    std::sort(ranges.begin(), ranges.end(), [](const PayloadRange& left, const PayloadRange& right) {
        return left.begin < right.begin;
    });
    std::uint64_t previous_end = 0;
    for (const PayloadRange& range : ranges) {
        if (range.begin < previous_end) { throw ProjectionError("projection payload spans overlap"); }
        previous_end = range.end;
    }
    return records;
}

std::vector<std::uint8_t> read_bounded_file(const std::filesystem::path& path,
                                            std::uint64_t maximum_bytes,
                                            std::string_view label) {
    class FileDescriptor {
    public:
        explicit FileDescriptor(const std::filesystem::path& source)
            : descriptor_(::open(source.c_str(), O_RDONLY | O_CLOEXEC)) {}

        ~FileDescriptor() {
            if (descriptor_ >= 0) { (void)::close(descriptor_); }
        }

        FileDescriptor(const FileDescriptor&)            = delete;
        FileDescriptor& operator=(const FileDescriptor&) = delete;

        [[nodiscard]] int get() const noexcept { return descriptor_; }

    private:
        int descriptor_ = -1;
    };

    FileDescriptor input(path);
    if (input.get() < 0) { throw ProjectionError("could not open " + std::string(label)); }

    std::vector<std::uint8_t> encoded;
    std::array<std::uint8_t, 64ULL << 10> buffer{};
    while (true) {
        ssize_t count = -1;
        do {
            count = ::read(input.get(), buffer.data(), buffer.size());
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            throw ProjectionError("could not read complete " + std::string(label));
        }
        if (count == 0) { break; }
        const auto bytes = static_cast<std::size_t>(count);
        if (encoded.size() > maximum_bytes || bytes > maximum_bytes - encoded.size()) {
            throw ProjectionError(std::string(label) + " exceeds the bounded schema");
        }
        encoded.insert(encoded.end(), buffer.begin(), buffer.begin() + bytes);
    }
    return encoded;
}

void validate_payload(const PinnedHostBuffer& payload,
                      const std::vector<ParsedRecord>& records) {
    const auto* bytes = static_cast<const std::uint8_t*>(payload.data());
    for (const ParsedRecord& record : records) {
        for (const auto [offset, count] : {
                 std::pair{record.direction_offset, record.direction_count},
                 std::pair{record.signature_offset, record.signature_count},
             }) {
            const float* values = reinterpret_cast<const float*>(bytes + offset);
            for (std::uint32_t index = 0; index < count; ++index) {
                if (!std::isfinite(values[index])) {
                    throw ProjectionError("projection payload contains a non-finite FP32 value");
                }
            }
        }
        const float* direction = reinterpret_cast<const float*>(bytes + record.direction_offset);
        double squared_norm = 0.0;
        for (std::uint32_t index = 0; index < record.direction_count; ++index) {
            squared_norm += static_cast<double>(direction[index]) * direction[index];
        }
        const double norm = std::sqrt(squared_norm);
        if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1e-5) {
            throw ProjectionError("projection direction must have unit length");
        }
    }
}

void require_exact_fields(const Json& value, std::initializer_list<std::string_view> fields,
                          std::string_view label) {
    if (!value.is_object() || value.size() != fields.size()) {
        throw ProjectionError(std::string(label) + " has missing or extra members");
    }
    std::set<std::string_view> remaining(fields.begin(), fields.end());
    for (const auto& [name, unused] : value.items()) {
        (void)unused;
        if (!remaining.erase(name)) {
            throw ProjectionError(std::string(label) + " has unknown member " + name);
        }
    }
    if (!remaining.empty()) {
        throw ProjectionError(std::string(label) + " has missing members");
    }
}

void require_unsigned_value(const Json& value, std::string_view name, std::uint64_t expected) {
    const Json& actual = member(value, name);
    if (!actual.is_number_unsigned() || actual.get<std::uint64_t>() != expected) {
        throw ProjectionError("projection conversion report member " + std::string(name) +
                              " is not the approved value");
    }
}

void require_string_value(const Json& value, std::string_view name, std::string_view expected) {
    if (string_member(value, name) != expected) {
        throw ProjectionError("projection conversion report member " + std::string(name) +
                              " is not the approved value");
    }
}

void validate_report_identity(const Json& report, std::string_view expected_model_sha256) {
    require_exact_fields(report,
                         {"artifact", "decoder_revision", "directions", "max_signature_error",
                          "records", "schema_version", "sidecar", "sites",
                          "unclaimed_records", "weight_formats"},
                         "projection conversion report");
    const Json& artifact = member(report, "artifact");
    const Json& directions = member(report, "directions");
    const Json& sidecar = member(report, "sidecar");
    require_exact_fields(artifact, {"bytes", "path", "sha256"},
                         "projection conversion report artifact");
    require_exact_fields(directions, {"bytes", "path", "sha256"},
                         "projection conversion report directions");
    require_exact_fields(sidecar, {"bytes", "path", "sha256"},
                         "projection conversion report sidecar");
    (void)string_member(artifact, "path");
    (void)string_member(directions, "path");
    (void)string_member(sidecar, "path");
    require_unsigned_value(artifact, "bytes", 21492695040ULL);
    require_string_value(artifact, "sha256", expected_model_sha256);
    require_unsigned_value(directions, "bytes", 2643584);
    require_string_value(directions, "sha256", kApprovedDirectionSha256);
    require_unsigned_value(sidecar, "bytes", 8695808);
    require_string_value(sidecar, "sha256", kApprovedSidecarSha256);
    require_unsigned_value(report, "schema_version", 1);
    require_unsigned_value(report, "records", 128);
    require_unsigned_value(report, "unclaimed_records", 0);
    require_string_value(report, "decoder_revision",
                         "a05746aa:tools.artifact.layouts.fp8-row/nvfp4");
    const Json& max_error = member(report, "max_signature_error");
    if (!max_error.is_number() || max_error.get<double>() != 0.0) {
        throw ProjectionError("projection conversion report signature error is not zero");
    }
    const Json& sites = member(report, "sites");
    require_exact_fields(sites, {"attention_output", "gdn_output", "mlp_down"},
                         "projection conversion report sites");
    require_unsigned_value(sites, "attention_output", 16);
    require_unsigned_value(sites, "gdn_output", 48);
    require_unsigned_value(sites, "mlp_down", 64);
    const Json& formats = member(report, "weight_formats");
    require_exact_fields(formats, {"fp8_row", "nvfp4_block"},
                         "projection conversion report weight formats");
    require_unsigned_value(formats, "fp8_row", 72);
    require_unsigned_value(formats, "nvfp4_block", 56);
}

void authenticate_conversion_report(const std::filesystem::path& sidecar_path,
                                    std::string_view expected_model_sha256) {
    std::filesystem::path report_path = sidecar_path;
    report_path.replace_extension(".conversion.json");
    const std::vector<std::uint8_t> encoded_bytes =
        read_bounded_file(report_path, kMaximumReportBytes,
                          "approved projection conversion report");
    if (encoded_bytes.empty()) {
        throw ProjectionError("approved projection conversion report is missing or unbounded");
    }
    if (sha256(encoded_bytes) != parse_digest(kApprovedReportSha256)) {
        throw ProjectionError("projection conversion report is not the approved report");
    }
    const std::string encoded(reinterpret_cast<const char*>(encoded_bytes.data()),
                              encoded_bytes.size());
    validate_report_identity(parse_manifest(encoded), expected_model_sha256);
}

} // namespace

class ResidualProjectionTable::Impl {
public:
    Impl(std::size_t payload_bytes, std::vector<ParsedRecord> parsed_records)
        : host(payload_bytes), device(payload_bytes), records(std::move(parsed_records)) {
        lookup.fill(-1);
        for (std::size_t index = 0; index < records.size(); ++index) {
            const ParsedRecord& record = records[index];
            const std::size_t slot = lookup_index(record.layer, record.site);
            if (lookup[slot] != -1) { throw ProjectionError("duplicate projection lookup claim"); }
            lookup[slot] = static_cast<std::int16_t>(index);
        }
    }

    PinnedHostBuffer host;
    DeviceBuffer device;
    std::vector<ParsedRecord> records;
    std::array<std::int16_t, kLayers * 3> lookup{};
};

ResidualProjectionTable::ResidualProjectionTable(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ResidualProjectionTable::~ResidualProjectionTable() = default;
ResidualProjectionTable::ResidualProjectionTable(ResidualProjectionTable&&) noexcept = default;
ResidualProjectionTable&
ResidualProjectionTable::operator=(ResidualProjectionTable&&) noexcept = default;

ResidualProjectionTable
ResidualProjectionTable::load(const std::filesystem::path& path,
                              std::string_view expected_model_sha256) {
    if (path.empty()) { throw ProjectionError("projection sidecar path must not be empty"); }
    constexpr std::uint64_t kMaximumSidecarBytes =
        kHeaderBytes + kMaximumManifestBytes + kPayloadAlignment + kMaximumPayloadBytes;
    const std::vector<std::uint8_t> encoded =
        read_bounded_file(path, kMaximumSidecarBytes, "projection sidecar");
    const std::uint64_t file_bytes = encoded.size();
    if (file_bytes < kHeaderBytes) {
        throw ProjectionError("projection sidecar is shorter than its header");
    }
    std::array<std::uint8_t, kHeaderBytes> raw_header{};
    std::copy_n(encoded.data(), raw_header.size(), raw_header.data());
    const Header header = parse_header(raw_header);
    if (header.version != kSchemaVersion) {
        throw ProjectionError("unsupported projection sidecar schema version");
    }
    if (header.manifest_bytes == 0 || header.manifest_bytes > kMaximumManifestBytes) {
        throw ProjectionError("projection manifest length is outside the bounded schema");
    }
    if (header.payload_bytes == 0 || header.payload_bytes > kMaximumPayloadBytes) {
        throw ProjectionError("projection payload length is outside the bounded schema");
    }
    const std::uint64_t manifest_end = checked_add(kHeaderBytes, header.manifest_bytes,
                                                   "projection manifest range");
    const std::uint64_t payload_offset = align_up(manifest_end);
    if (checked_add(payload_offset, header.payload_bytes, "projection file range") != file_bytes) {
        throw ProjectionError("projection payload range does not match the file length");
    }
    if (header.artifact_sha != parse_digest(expected_model_sha256)) {
        throw ProjectionError("projection artifact SHA-256 does not match the expected model");
    }
    if (header.direction_sha != parse_digest(kApprovedDirectionSha256)) {
        throw ProjectionError("projection direction SHA-256 is not the approved source");
    }

    const std::string manifest(
        reinterpret_cast<const char*>(encoded.data() + kHeaderBytes),
        static_cast<std::size_t>(header.manifest_bytes));
    const auto manifest_span = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(manifest.data()), manifest.size());
    if (sha256(manifest_span) != header.manifest_sha) {
        throw ProjectionError("projection manifest SHA-256 mismatch");
    }
    Json manifest_value = parse_manifest(manifest);
    std::vector<ParsedRecord> records = validate_manifest(manifest_value, header.payload_bytes);

    auto impl = std::make_unique<Impl>(static_cast<std::size_t>(header.payload_bytes),
                                      std::move(records));
    std::memcpy(impl->host.data(), encoded.data() + static_cast<std::size_t>(payload_offset),
                impl->host.size());
    validate_payload(impl->host, impl->records);
    if (sha256(encoded) != parse_digest(kApprovedSidecarSha256)) {
        throw ProjectionError("projection file is not the approved sidecar");
    }
    authenticate_conversion_report(path, expected_model_sha256);
    impl->device.copy_from_host(impl->host.data(), impl->host.size());

    ResidualProjectionTable table(std::move(impl));
    table.verify_all_claimed();
    return table;
}

ResidualProjectionView ResidualProjectionTable::view(std::uint32_t layer,
                                                     ProjectionSite site) const {
    if (impl_ == nullptr) { throw ProjectionError("projection table is empty"); }
    const std::int16_t record_index = impl_->lookup[lookup_index(layer, site)];
    if (record_index < 0) { throw ProjectionError("projection writer is not claimed"); }
    const ParsedRecord& record = impl_->records[static_cast<std::size_t>(record_index)];
    const auto* base = static_cast<const std::uint8_t*>(impl_->device.p);
    return ResidualProjectionView{
        .direction = reinterpret_cast<const float*>(base + record.direction_offset),
        .signature = reinterpret_cast<const float*>(base + record.signature_offset),
        .coefficient = record.coefficient,
        .direction_count = record.direction_count,
        .signature_count = record.signature_count,
    };
}

void ResidualProjectionTable::verify_all_claimed() const {
    if (impl_ == nullptr) { throw ProjectionError("projection table is empty"); }
    std::size_t claims = 0;
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        const ProjectionSite mixer = layer % 4 == 3 ? ProjectionSite::AttentionOutput
                                                    : ProjectionSite::GdnOutput;
        (void)view(layer, mixer);
        (void)view(layer, ProjectionSite::MlpDown);
        claims += 2;
        const ProjectionSite unclaimed = layer % 4 == 3 ? ProjectionSite::GdnOutput
                                                        : ProjectionSite::AttentionOutput;
        if (impl_->lookup[lookup_index(layer, unclaimed)] != -1) {
            throw ProjectionError("projection table claims an impossible mixer site");
        }
    }
    if (claims != kWriterCount || impl_->records.size() != kWriterCount) {
        throw ProjectionError("projection table does not contain exactly 128 writers");
    }
}

} // namespace ninfer
