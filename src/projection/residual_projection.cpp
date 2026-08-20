#include <ninfer/projection/residual_projection.h>

#include "core/arena.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <span>
#include <string>
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
constexpr std::size_t kLayers            = 64;
constexpr std::size_t kWriterCount       = 128;
constexpr std::uint32_t kDirectionCount  = 5120;
constexpr std::string_view kApprovedDirectionSha256 =
    "9de12cbe71f38baf2f6b4a21dfcb2b13bd6416ab4785214afce27c7543f05c1d";
constexpr std::array<char, 8> kMagic{'N', 'I', 'N', 'F', 'R', 'P', '1', '\0'};
constexpr std::array<std::string_view, 10> kRecordFields{
    "coefficient",          "direction_count",      "direction_offset",
    "key",                  "layer",                "signature_count",
    "signature_offset",     "site",                 "weight_format",
    "weight_payload_sha256",
};

using Digest = std::array<std::uint8_t, 32>;

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

std::uint32_t load_be32(const std::uint8_t* bytes) {
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) |
           static_cast<std::uint32_t>(bytes[3]);
}

void store_be32(std::uint32_t value, std::uint8_t* bytes) {
    bytes[0] = static_cast<std::uint8_t>(value >> 24U);
    bytes[1] = static_cast<std::uint8_t>(value >> 16U);
    bytes[2] = static_cast<std::uint8_t>(value >> 8U);
    bytes[3] = static_cast<std::uint8_t>(value);
}

void sha256_block(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) {
    constexpr std::array<std::uint32_t, 64> round{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
        0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
        0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
        0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
        words[index] = load_be32(block + 4 * index);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
        const std::uint32_t s0 = std::rotr(words[index - 15], 7) ^
                                 std::rotr(words[index - 15], 18) ^
                                 (words[index - 15] >> 3U);
        const std::uint32_t s1 = std::rotr(words[index - 2], 17) ^
                                 std::rotr(words[index - 2], 19) ^
                                 (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
        const std::uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const std::uint32_t choose = (e & f) ^ (~e & g);
        const std::uint32_t t1 = h + sum1 + choose + round[index] + words[index];
        const std::uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

Digest sha256(std::span<const std::uint8_t> input) {
    if (input.size() > std::numeric_limits<std::uint64_t>::max() / 8ULL) {
        throw ProjectionError("projection manifest is too large to hash");
    }
    std::array<std::uint32_t, 8> state{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                       0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                       0x1f83d9abU, 0x5be0cd19U};
    std::size_t offset = 0;
    while (input.size() - offset >= 64) {
        sha256_block(state, input.data() + offset);
        offset += 64;
    }
    std::array<std::uint8_t, 128> tail{};
    const std::size_t remaining = input.size() - offset;
    std::copy_n(input.data() + offset, remaining, tail.data());
    tail[remaining] = 0x80U;
    const std::size_t tail_bytes = remaining < 56 ? 64 : 128;
    const std::uint64_t bits = static_cast<std::uint64_t>(input.size()) * 8ULL;
    for (std::size_t index = 0; index < 8; ++index) {
        tail[tail_bytes - 1 - index] = static_cast<std::uint8_t>(bits >> (8U * index));
    }
    sha256_block(state, tail.data());
    if (tail_bytes == 128) { sha256_block(state, tail.data() + 64); }
    Digest result{};
    for (std::size_t index = 0; index < state.size(); ++index) {
        store_be32(state[index], result.data() + 4 * index);
    }
    return result;
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

void read_exact(std::ifstream& input, void* destination, std::size_t bytes, const char* label) {
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw ProjectionError(std::string(label) + " exceeds streamsize");
    }
    input.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (!input || input.gcount() != static_cast<std::streamsize>(bytes)) {
        throw ProjectionError(std::string("could not read complete ") + label);
    }
}

void validate_finite_payload(const PinnedHostBuffer& payload,
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
    }
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
    std::error_code file_error;
    const std::uint64_t file_bytes = std::filesystem::file_size(path, file_error);
    if (file_error) {
        throw ProjectionError("could not stat projection sidecar: " + file_error.message());
    }
    if (file_bytes < kHeaderBytes) {
        throw ProjectionError("projection sidecar is shorter than its header");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw ProjectionError("could not open projection sidecar"); }
    std::array<std::uint8_t, kHeaderBytes> raw_header{};
    read_exact(input, raw_header.data(), raw_header.size(), "projection header");
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

    std::string manifest(static_cast<std::size_t>(header.manifest_bytes), '\0');
    read_exact(input, manifest.data(), manifest.size(), "projection manifest");
    const auto manifest_span = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(manifest.data()), manifest.size());
    if (sha256(manifest_span) != header.manifest_sha) {
        throw ProjectionError("projection manifest SHA-256 mismatch");
    }
    Json manifest_value = parse_manifest(manifest);
    std::vector<ParsedRecord> records = validate_manifest(manifest_value, header.payload_bytes);

    auto impl = std::make_unique<Impl>(static_cast<std::size_t>(header.payload_bytes),
                                      std::move(records));
    input.seekg(static_cast<std::streamoff>(payload_offset), std::ios::beg);
    if (!input) { throw ProjectionError("could not seek to projection payload"); }
    read_exact(input, impl->host.data(), impl->host.size(), "projection payload");
    validate_finite_payload(impl->host, impl->records);
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
