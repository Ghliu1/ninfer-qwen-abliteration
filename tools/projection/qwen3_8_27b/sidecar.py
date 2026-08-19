"""Container and pinned-source helpers for a Qwen3.8 projection sidecar.

The sidecar is deliberately independent of the C++ runtime.  It stores the
normalized output directions and FP32 signatures for the exact served weight
payloads, while this module owns only the framing, source-file validation, and
the fixed writer inventory.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


MAGIC = b"NINFRP1\0"
SCHEMA_VERSION = 1
HEADER = struct.Struct("<8sIQQ32s32s32s")
PAYLOAD_ALIGNMENT = 4096
FP32_BYTES = 4
DIRECTION_COUNT = 5120
WRITER_COUNT = 128
APPROVED_ARTIFACT_SHA256 = (
    "bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32"
)
APPROVED_DIRECTION_SHA256 = (
    "9de12cbe71f38baf2f6b4a21dfcb2b13bd6416ab4785214afce27c7543f05c1d"
)

_RECORD_FIELDS = frozenset(
    {
        "key",
        "layer",
        "site",
        "coefficient",
        "direction_offset",
        "direction_count",
        "signature_offset",
        "signature_count",
        "weight_format",
        "weight_payload_sha256",
    }
)
_SITES = frozenset({"attention_output", "gdn_output", "mlp_down"})
_FORMATS = frozenset({"fp8_row", "nvfp4_block"})


class SidecarError(ValueError):
    """The sidecar or one of its pinned source inputs is invalid."""


def align_up(value: int, alignment: int = PAYLOAD_ALIGNMENT) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("alignment requires a nonnegative value and positive alignment")
    return (value + alignment - 1) // alignment * alignment


def expected_writer_keys() -> tuple[str, ...]:
    """Return the exact 128 output-writing module keys in writer order."""

    keys: list[str] = []
    for layer in range(64):
        mixer = (
            "self_attn.o_proj"
            if layer % 4 == 3
            else "linear_attn.out_proj"
        )
        keys.extend(
            (
                f"model.layers.{layer}.{mixer}",
                f"model.layers.{layer}.mlp.down_proj",
            )
        )
    return tuple(keys)


def direction_key_for_writer(key: str) -> str:
    """Map the runtime writer key to the pinned safetensors key."""

    prefix = "model."
    if not key.startswith(prefix):
        raise SidecarError(f"unsupported writer key: {key}")
    return "model.language_model." + key[len(prefix) :]


def artifact_name_for_writer(key: str) -> str:
    """Map a writer key to its exact physical NInfer object name."""

    parts = key.split(".")
    if len(parts) != 5 or parts[0] != "model" or parts[1] != "layers":
        raise SidecarError(f"unsupported writer key: {key}")
    try:
        layer = int(parts[2])
    except ValueError as exc:
        raise SidecarError(f"unsupported writer layer: {key}") from exc
    suffix = ".".join(parts[3:])
    if suffix == "linear_attn.out_proj":
        return f"text/layers/{layer}/gdn/output"
    if suffix == "self_attn.o_proj":
        return f"text/layers/{layer}/attention/output"
    if suffix == "mlp.down_proj":
        return f"text/layers/{layer}/mlp/down"
    raise SidecarError(f"unsupported writer suffix: {key}")


def site_for_writer(key: str) -> str:
    if key.endswith("linear_attn.out_proj"):
        return "gdn_output"
    if key.endswith("self_attn.o_proj"):
        return "attention_output"
    if key.endswith("mlp.down_proj"):
        return "mlp_down"
    raise SidecarError(f"unsupported writer key: {key}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_bytes(value: str, field: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise SidecarError(f"{field} must be a 64-character SHA-256 hex digest")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise SidecarError(f"{field} must be a 64-character SHA-256 hex digest") from exc


@dataclass(frozen=True, slots=True)
class SidecarRecord:
    key: str
    layer: int
    site: str
    coefficient: float
    direction_offset: int
    direction_count: int
    signature_offset: int
    signature_count: int
    weight_format: str
    weight_payload_sha256: str

    @classmethod
    def from_json(cls, value: object) -> "SidecarRecord":
        if not isinstance(value, dict) or frozenset(value) != _RECORD_FIELDS:
            raise SidecarError("projection record has missing or extra members")
        key = value["key"]
        if not isinstance(key, str) or not key:
            raise SidecarError("projection record key must be a nonempty string")
        layer = value["layer"]
        if type(layer) is not int or layer < 0:
            raise SidecarError("projection record layer must be a nonnegative integer")
        site = value["site"]
        if site not in _SITES:
            raise SidecarError(f"unsupported projection site: {site!r}")
        coefficient = value["coefficient"]
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
            raise SidecarError("projection coefficient must be numeric")
        coefficient = float(coefficient)
        if not math.isfinite(coefficient):
            raise SidecarError("projection coefficient must be finite")
        fields = (
            "direction_offset",
            "direction_count",
            "signature_offset",
            "signature_count",
        )
        ints: list[int] = []
        for field in fields:
            item = value[field]
            if type(item) is not int or item < 0:
                raise SidecarError(f"{field} must be a nonnegative integer")
            ints.append(item)
        if ints[1] != DIRECTION_COUNT:
            raise SidecarError(f"direction_count must be {DIRECTION_COUNT}")
        expected_signature_count = 17408 if site == "mlp_down" else 6144
        if ints[3] != expected_signature_count:
            raise SidecarError(
                f"signature_count for {site} must be {expected_signature_count}"
            )
        weight_format = value["weight_format"]
        if weight_format not in _FORMATS:
            raise SidecarError(f"unsupported projection weight format: {weight_format!r}")
        payload_sha = value["weight_payload_sha256"]
        _digest_bytes(payload_sha, "weight_payload_sha256")
        return cls(
            key,
            layer,
            site,
            coefficient,
            ints[0],
            ints[1],
            ints[2],
            ints[3],
            weight_format,
            payload_sha,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "key": self.key,
            "layer": self.layer,
            "site": self.site,
            "coefficient": self.coefficient,
            "direction_offset": self.direction_offset,
            "direction_count": self.direction_count,
            "signature_offset": self.signature_offset,
            "signature_count": self.signature_count,
            "weight_format": self.weight_format,
            "weight_payload_sha256": self.weight_payload_sha256,
        }


class SidecarReader:
    """Structurally validate and read a sidecar without materializing its payload."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mapping: mmap.mmap | None = None
        try:
            self._file.seek(0, os.SEEK_END)
            self.file_bytes = self._file.tell()
            self._file.seek(0)
            if self.file_bytes < HEADER.size:
                raise SidecarError("sidecar is shorter than its header")
            raw_header = self._file.read(HEADER.size)
            (
                magic,
                version,
                manifest_bytes,
                payload_bytes,
                manifest_sha,
                artifact_sha,
                direction_sha,
            ) = HEADER.unpack(raw_header)
            if magic != MAGIC:
                raise SidecarError("sidecar magic is not NINFRP1")
            if version != SCHEMA_VERSION:
                raise SidecarError(f"unsupported sidecar schema version: {version}")
            if manifest_bytes <= 0:
                raise SidecarError("sidecar manifest must not be empty")
            manifest_end = HEADER.size + manifest_bytes
            payload_offset = align_up(manifest_end)
            if payload_offset + payload_bytes != self.file_bytes:
                raise SidecarError("sidecar payload range does not match file length")
            self._file.seek(HEADER.size)
            manifest_raw = self._file.read(manifest_bytes)
            if hashlib.sha256(manifest_raw).digest() != manifest_sha:
                raise SidecarError("sidecar manifest SHA-256 mismatch")
            try:
                manifest_value = json.loads(manifest_raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SidecarError(f"invalid sidecar manifest JSON: {exc}") from exc
            if not isinstance(manifest_value, list):
                raise SidecarError("sidecar manifest must be an array")
            records = tuple(SidecarRecord.from_json(item) for item in manifest_value)
            self.manifest_bytes = manifest_bytes
            self.payload_offset = payload_offset
            self.payload_bytes = payload_bytes
            self.manifest = records
            self.artifact_sha256 = artifact_sha.hex()
            self.direction_sha256 = direction_sha.hex()
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._validate_ranges()
        except BaseException:
            if self._mapping is not None:
                self._mapping.close()
            self._file.close()
            raise

    def _validate_ranges(self) -> None:
        ranges: list[tuple[int, int, str]] = []
        for record in self.manifest:
            for offset, count, name in (
                (record.direction_offset, record.direction_count, "direction"),
                (record.signature_offset, record.signature_count, "signature"),
            ):
                if offset % PAYLOAD_ALIGNMENT:
                    raise SidecarError(
                        f"{record.key} {name} offset is not "
                        f"{PAYLOAD_ALIGNMENT}-byte aligned"
                    )
                if offset % FP32_BYTES:
                    raise SidecarError(f"{record.key} {name} offset is not FP32-aligned")
                end = offset + count * FP32_BYTES
                if end > self.payload_bytes:
                    raise SidecarError(f"{record.key} {name} extends beyond payload")
                ranges.append((offset, end, f"{record.key}:{name}"))
        ranges.sort()
        previous_end = 0
        for begin, end, label in ranges:
            if begin < previous_end:
                raise SidecarError(f"sidecar payload ranges overlap at {label}")
            previous_end = end

    def payload_view(self, offset: int, count: int) -> memoryview:
        if self._mapping is None:
            raise RuntimeError("sidecar is closed")
        if offset < 0 or count < 0 or offset + count > self.payload_bytes:
            raise SidecarError("sidecar payload view is outside the payload")
        begin = self.payload_offset + offset
        return memoryview(self._mapping)[begin : begin + count]

    def record_payload(self, record: SidecarRecord, field: str) -> memoryview:
        if field == "direction":
            return self.payload_view(
                record.direction_offset, record.direction_count * FP32_BYTES
            )
        if field == "signature":
            return self.payload_view(
                record.signature_offset, record.signature_count * FP32_BYTES
            )
        raise ValueError(f"unknown sidecar payload field: {field}")

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "SidecarReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _canonical_manifest(records: Sequence[SidecarRecord]) -> bytes:
    return json.dumps(
        [record.to_json() for record in records],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_source(payload: bytes | bytearray | memoryview | Path) -> tuple[int, Iterator[bytes]]:
    if isinstance(payload, Path):
        size = payload.stat().st_size

        def chunks() -> Iterator[bytes]:
            with payload.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    yield chunk

        return size, chunks()
    raw = memoryview(payload).cast("B")
    return raw.nbytes, iter((raw.tobytes(),))


def write_sidecar(
    path: str | Path,
    records: Sequence[SidecarRecord],
    payload: bytes | bytearray | memoryview | Path,
    artifact_sha256: str,
    direction_sha256: str,
) -> SidecarReader:
    """Write a sidecar via a same-directory fsync + atomic rename."""

    output = Path(path)
    raw_records = tuple(records)
    if len(raw_records) != WRITER_COUNT:
        raise SidecarError(f"sidecar must contain {WRITER_COUNT} records")
    # Validate the dataclass values through the same strict JSON contract used
    # by the reader.  This rejects NaN/Infinity before a temporary file exists,
    # so a failed publication cannot replace an existing final sidecar.
    records = tuple(
        SidecarRecord.from_json(record.to_json()) for record in raw_records
    )
    if len({record.key for record in records}) != len(records):
        raise SidecarError("sidecar contains duplicate projection records")
    manifest = _canonical_manifest(records)
    payload_size, chunks = _payload_source(payload)
    for record in records:
        for offset, count, name in (
            (record.direction_offset, record.direction_count, "direction"),
            (record.signature_offset, record.signature_count, "signature"),
        ):
            if offset % PAYLOAD_ALIGNMENT:
                raise SidecarError(
                    f"{record.key} {name} offset is not "
                    f"{PAYLOAD_ALIGNMENT}-byte aligned"
                )
            if offset % FP32_BYTES:
                raise SidecarError(f"{record.key} {name} offset is not FP32-aligned")
            if offset + count * FP32_BYTES > payload_size:
                raise SidecarError(f"{record.key} {name} extends beyond payload")
    header = HEADER.pack(
        MAGIC,
        SCHEMA_VERSION,
        len(manifest),
        payload_size,
        hashlib.sha256(manifest).digest(),
        _digest_bytes(artifact_sha256, "artifact_sha256"),
        _digest_bytes(direction_sha256, "direction_sha256"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(header)
            stream.write(manifest)
            stream.write(b"\x00" * (align_up(HEADER.size + len(manifest)) - HEADER.size - len(manifest)))
            for chunk in chunks:
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        # Reopen and fully validate the sibling before making it visible.  If
        # this fails, the existing final path is left byte-identical.
        temporary_reader = SidecarReader(temporary)
        temporary_reader.close()
        os.replace(temporary, output)
        try:
            directory_fd = os.open(output.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return SidecarReader(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, value: object) -> None:
    """Write canonical metadata JSON with the same durable publication rule."""

    output = Path(path)
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        try:
            directory_fd = os.open(output.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class DirectionSource:
    path: Path
    sha256: str
    keys: tuple[str, ...]
    metadata: dict[str, str]
    offsets: dict[str, tuple[int, int]]
    coefficient_offsets: tuple[int, int]
    payload: bytes

    def raw_direction(self, key: str) -> memoryview:
        try:
            begin, end = self.offsets[key]
        except KeyError as exc:
            raise SidecarError(f"direction source is missing {key}") from exc
        return memoryview(self.payload)[begin:end]

    def coefficients(self) -> tuple[float, ...]:
        begin, end = self.coefficient_offsets
        if end - begin != WRITER_COUNT * FP32_BYTES:
            raise SidecarError("direction source coefficient payload has wrong size")
        return struct.unpack("<128f", self.payload[begin:end])


def load_direction_source(path: str | Path) -> DirectionSource:
    """Read and strictly validate the pinned safetensors direction contract."""

    source_path = Path(path)
    raw = source_path.read_bytes()
    if len(raw) < 8:
        raise SidecarError("direction source is shorter than its safetensors header")
    header_bytes = struct.unpack("<Q", raw[:8])[0]
    if header_bytes <= 0 or 8 + header_bytes > len(raw):
        raise SidecarError("direction source header range is invalid")
    try:
        header = json.loads(raw[8 : 8 + header_bytes].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SidecarError(f"invalid direction source header: {exc}") from exc
    if not isinstance(header, dict):
        raise SidecarError("direction source header must be an object")
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise SidecarError("direction source is missing __metadata__")
    expected_meta = {
        "source_base": "Qwen/Qwen3.8-27B",
        "source_abl": "Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored",
        "modules": "128",
    }
    for field, expected in expected_meta.items():
        if metadata.get(field) != expected:
            raise SidecarError(f"direction metadata {field} does not match the pinned source")
    order_raw = metadata.get("coef_order")
    if not isinstance(order_raw, str):
        raise SidecarError("direction metadata is missing coef_order")
    try:
        order_value = json.loads(order_raw)
    except json.JSONDecodeError as exc:
        raise SidecarError(f"direction coef_order is not JSON: {exc}") from exc
    expected_keys = expected_writer_keys()
    expected_direction_keys = tuple(direction_key_for_writer(key) for key in expected_keys)
    if not isinstance(order_value, list) or tuple(order_value) != tuple(sorted(order_value)):
        # The pinned file uses safetensors' lexical key order for its coefficient
        # table.  Requiring that exact sequence prevents implicit alphabetical
        # remapping from becoming a source of silent coefficient drift.
        raise SidecarError("direction coef_order is not the pinned lexical order")
    if tuple(order_value) != tuple(sorted(expected_direction_keys)):
        raise SidecarError("direction coef_order does not match pinned safetensors order")
    blob = raw[8 + header_bytes :]
    offsets: dict[str, tuple[int, int]] = {}
    ranges: list[tuple[int, int, str]] = []
    for key, descriptor in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise SidecarError(f"direction descriptor for {key!r} is malformed")
        if descriptor["dtype"] != "F32" or descriptor["shape"] != [DIRECTION_COUNT]:
            if key != "__coefs__":
                raise SidecarError(f"direction tensor {key!r} is not F32[5120]")
        offsets_value = descriptor["data_offsets"]
        if (
            not isinstance(offsets_value, list)
            or len(offsets_value) != 2
            or any(type(item) is not int for item in offsets_value)
        ):
            raise SidecarError(f"direction descriptor range for {key!r} is malformed")
        begin, end = offsets_value
        if begin < 0 or end < begin or end > len(blob):
            raise SidecarError(f"direction descriptor range for {key!r} is invalid")
        expected_bytes = (WRITER_COUNT if key == "__coefs__" else DIRECTION_COUNT) * FP32_BYTES
        if descriptor["dtype"] != "F32" or descriptor["shape"] != [expected_bytes // FP32_BYTES]:
            raise SidecarError(f"direction descriptor for {key!r} has wrong shape or dtype")
        if end - begin != expected_bytes:
            raise SidecarError(f"direction descriptor range for {key!r} has wrong size")
        ranges.append((begin, end, key))
        if key == "__coefs__":
            coefficient_offsets = (begin, end)
        else:
            offsets[key] = (begin, end)
    if set(offsets) != set(expected_direction_keys):
        raise SidecarError("direction source tensor inventory does not match pinned keys")
    if len(ranges) != WRITER_COUNT + 1 or "coefficient_offsets" not in locals():
        raise SidecarError("direction source must contain 128 directions and __coefs__")
    ranges.sort()
    previous_end = 0
    for begin, end, key in ranges:
        if begin != previous_end:
            raise SidecarError(f"direction source payload ranges overlap at {key}")
        previous_end = end
    if previous_end != len(blob):
        raise SidecarError("direction source payload ranges do not cover the complete payload")
    return DirectionSource(
        source_path,
        hashlib.sha256(raw).hexdigest(),
        tuple(sorted(offsets)),
        {str(key): str(value) for key, value in metadata.items()},
        offsets,
        coefficient_offsets,
        blob,
    )


__all__ = [
    "DIRECTION_COUNT",
    "APPROVED_ARTIFACT_SHA256",
    "APPROVED_DIRECTION_SHA256",
    "DirectionSource",
    "HEADER",
    "MAGIC",
    "PAYLOAD_ALIGNMENT",
    "SCHEMA_VERSION",
    "SidecarError",
    "SidecarReader",
    "SidecarRecord",
    "WRITER_COUNT",
    "align_up",
    "artifact_name_for_writer",
    "atomic_write_json",
    "direction_key_for_writer",
    "expected_writer_keys",
    "load_direction_source",
    "sha256_file",
    "site_for_writer",
    "write_sidecar",
]
