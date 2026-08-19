from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from tools.projection.qwen3_8_27b.sidecar import (
    SidecarError,
    HEADER,
    SidecarReader,
    SidecarRecord,
    align_up,
    direction_key_for_writer,
    expected_writer_keys,
    site_for_writer,
    write_sidecar,
)
from tools.projection.qwen3_8_27b.verify_sidecar import verify_sidecar
from tools.artifact.container import Artifact, TensorObject
from tools.projection.qwen3_8_27b.build_sidecar import _validate_artifact_object

build_module = importlib.import_module("tools.projection.qwen3_8_27b.build_sidecar")
verify_module = importlib.import_module("tools.projection.qwen3_8_27b.verify_sidecar")


APPROVED_ARTIFACT_SHA256 = (
    "bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32"
)
APPROVED_DIRECTION_SHA256 = (
    "9de12cbe71f38baf2f6b4a21dfcb2b13bd6416ab4785214afce27c7543f05c1d"
)


def test_expected_writer_keys_are_complete() -> None:
    keys = expected_writer_keys()
    assert len(keys) == 128
    assert keys[0] == "model.layers.0.linear_attn.out_proj"
    assert "model.layers.3.self_attn.o_proj" in keys
    assert "model.layers.63.mlp.down_proj" in keys
    assert len(set(keys)) == 128


def test_builder_alignment_helper_writes_payload() -> None:
    stream = io.BytesIO()
    assert build_module._append_aligned(stream, 0, b"abc") == 3
    assert stream.getvalue() == b"abc"


def test_sidecar_rejects_unclaimed_record(tmp_path: Path) -> None:
    # This fixture is intentionally only a malformed manifest: inventory rejection
    # must happen before a verifier attempts to decode any source payload.
    sidecar = tmp_path / "fixture.ninferproj"
    manifest = [
        {
            "key": "model.layers.64.mlp.down_proj",
            "layer": 64,
            "site": "mlp_down",
            "coefficient": 1.0,
            "direction_offset": 0,
            "direction_count": 5120,
            "signature_offset": 24576,
            "signature_count": 17408,
            "weight_format": "nvfp4_block",
            "weight_payload_sha256": "0" * 64,
        }
    ]
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    header = struct.pack(
        "<8sIQQ32s32s32s",
        b"NINFRP1\0",
        1,
        len(manifest_bytes),
        94208,
        hashlib.sha256(manifest_bytes).digest(),
        bytes.fromhex(APPROVED_ARTIFACT_SHA256),
        bytes.fromhex(APPROVED_DIRECTION_SHA256),
    )
    sidecar.write_bytes(header + manifest_bytes + b"\x00" * (4096 - ((len(header) + len(manifest_bytes)) % 4096)) + b"\x00" * 94208)

    with pytest.raises(SidecarError, match="unclaimed projection record"):
        verify_sidecar(sidecar, fixture_artifact())


def fixture_artifact() -> object:
    """A verifier input that is never reached for inventory failures."""

    return object()


def test_builder_rejects_nvfp4_mixer_descriptor() -> None:
    """Mixer outputs are pinned FP8 row-scale in the Qwen3.8 artifact."""

    class FakeArtifact:
        def find(self, name: str) -> TensorObject:
            assert name == "text/layers/0/gdn/output"
            return TensorObject(
                name=name,
                shape=(5120, 6144),
                format="NVFP4",
                layout="blockscale-k16-m128x4-v1",
                offset=0,
                bytes=1,
            )

    with pytest.raises(SidecarError, match="artifact descriptor mismatch"):
        _validate_artifact_object(FakeArtifact(), expected_writer_keys()[0])


@pytest.mark.skipif(
    not all(
        Path(path).is_file()
        for path in (
            os.environ.get(
                "QWEN38_ARTIFACT",
                "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
                "qwen3_8_27b_nvfp4.ninfer",
            ),
            os.environ.get(
                "QWEN38_DIRECTIONS",
                "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
                "refusal_dirs_qwen38.safetensors",
            ),
            os.environ.get(
                "QWEN38_SIDECAR",
                "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
                "qwen3_8_27b_refusal_projection.ninferproj",
            ),
        )
    ),
    reason="pinned Qwen3.8 artifact, directions, and sidecar are not available",
)
def test_verifier_rejects_corrupted_signature(tmp_path: Path) -> None:
    artifact = Path(
        os.environ.get(
            "QWEN38_ARTIFACT",
            "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
            "qwen3_8_27b_nvfp4.ninfer",
        )
    )
    directions = Path(
        os.environ.get(
            "QWEN38_DIRECTIONS",
            "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
            "refusal_dirs_qwen38.safetensors",
        )
    )
    source_sidecar = Path(
        os.environ.get(
            "QWEN38_SIDECAR",
            "/mnt/h/OpenClawLab/Models/ninfer/qwen38-27b-nvfp4-projected/"
            "qwen3_8_27b_refusal_projection.ninferproj",
        )
    )
    corrupted = tmp_path / "corrupted.ninferproj"
    payload = bytearray(source_sidecar.read_bytes())
    with SidecarReader(source_sidecar) as reader:
        record = reader.manifest[0]
        signature_byte = reader.payload_offset + record.signature_offset
    original = struct.unpack_from("<f", payload, signature_byte)[0]
    struct.pack_into("<f", payload, signature_byte, original + 1.0)
    corrupted.write_bytes(payload)

    with pytest.raises(SidecarError, match="signature mismatch"):
        verify_sidecar(
            corrupted,
            artifact,
            directions,
            report_path=tmp_path / "corrupted.conversion.json",
        )


def test_sidecar_writer_publishes_canonical_manifest_and_reopens(tmp_path: Path) -> None:
    records = []
    payload = bytearray()
    for index, key in enumerate(expected_writer_keys()):
        direction_offset = len(payload)
        payload.extend(b"\x00" * (5120 * 4))
        signature_count = 6144 if site_for_writer(key) != "mlp_down" else 17408
        signature_offset = len(payload)
        payload.extend(b"\x00" * (signature_count * 4))
        records.append(
            SidecarRecord(
                key,
                int(key.split(".")[2]),
                site_for_writer(key),
                0.5 + index / 1000,
                direction_offset,
                5120,
                signature_offset,
                signature_count,
                "fp8_row" if signature_count == 6144 else "nvfp4_block",
                "0" * 64,
            )
        )
    output = tmp_path / "roundtrip.ninferproj"
    reader = write_sidecar(output, records, bytes(payload), "a" * 64, "b" * 64)
    try:
        assert reader.file_bytes == output.stat().st_size
        assert reader.artifact_sha256 == "a" * 64
        assert reader.direction_sha256 == "b" * 64
        assert [record.key for record in reader.manifest] == list(expected_writer_keys())
    finally:
        reader.close()
    assert not list(tmp_path.glob("*.tmp"))
    with output.open("rb") as stream:
        header = HEADER.unpack(stream.read(HEADER.size))
        manifest = stream.read(header[2])
    assert hashlib.sha256(manifest).digest() == header[4]


def _fixture_records_payload(
    *,
    direction_words: tuple[float, ...] | None = None,
) -> tuple[list[SidecarRecord], bytes]:
    records: list[SidecarRecord] = []
    payload = bytearray()
    if direction_words is None:
        direction_bytes = b"\x00" * (5120 * 4)
    else:
        direction_bytes = struct.pack("<5120f", *direction_words)
    for index, key in enumerate(expected_writer_keys()):
        direction_offset = align_up(len(payload))
        payload.extend(b"\x00" * (direction_offset - len(payload)))
        payload.extend(direction_bytes)
        signature_count = 6144 if site_for_writer(key) != "mlp_down" else 17408
        signature_offset = align_up(len(payload))
        payload.extend(b"\x00" * (signature_offset - len(payload)))
        payload.extend(b"\x00" * (signature_count * 4))
        records.append(
            SidecarRecord(
                key,
                int(key.split(".")[2]),
                site_for_writer(key),
                0.5 + index / 1000,
                direction_offset,
                5120,
                signature_offset,
                signature_count,
                "fp8_row" if signature_count == 6144 else "nvfp4_block",
                "0" * 64,
            )
        )
    return records, bytes(payload)


def _write_raw_sidecar(
    path: Path,
    records: list[dict[str, object]] | list[SidecarRecord],
    payload: bytes,
    *,
    artifact_sha256: str = APPROVED_ARTIFACT_SHA256,
    direction_sha256: str = APPROVED_DIRECTION_SHA256,
) -> Path:
    manifest_values = [
        value.to_json() if isinstance(value, SidecarRecord) else value
        for value in records
    ]
    manifest = json.dumps(
        manifest_values,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload_offset = align_up(HEADER.size + len(manifest))
    header = HEADER.pack(
        b"NINFRP1\0",
        1,
        len(manifest),
        len(payload),
        hashlib.sha256(manifest).digest(),
        bytes.fromhex(artifact_sha256),
        bytes.fromhex(direction_sha256),
    )
    path.write_bytes(
        header
        + manifest
        + b"\x00" * (payload_offset - len(header) - len(manifest))
        + payload
    )
    return path


def test_sidecar_rejects_nonfinite_coefficient_without_replacing_existing(
    tmp_path: Path,
) -> None:
    records, payload = _fixture_records_payload()
    output = tmp_path / "existing.ninferproj"
    reader = write_sidecar(
        output,
        records,
        payload,
        APPROVED_ARTIFACT_SHA256,
        APPROVED_DIRECTION_SHA256,
    )
    reader.close()
    previous = output.read_bytes()
    corrupted = list(records)
    corrupted[0] = replace(corrupted[0], coefficient=float("nan"))

    with pytest.raises(SidecarError, match="finite"):
        write_sidecar(
            output,
            corrupted,
            payload,
            APPROVED_ARTIFACT_SHA256,
            APPROVED_DIRECTION_SHA256,
        )
    assert output.read_bytes() == previous


def test_sidecar_validates_temp_before_replacing_existing_on_overlap(
    tmp_path: Path,
) -> None:
    records, payload = _fixture_records_payload()
    output = tmp_path / "existing.ninferproj"
    reader = write_sidecar(
        output,
        records,
        payload,
        APPROVED_ARTIFACT_SHA256,
        APPROVED_DIRECTION_SHA256,
    )
    reader.close()
    previous = output.read_bytes()
    corrupted = list(records)
    corrupted[1] = replace(corrupted[1], direction_offset=corrupted[0].direction_offset)

    with pytest.raises(SidecarError, match="overlap"):
        write_sidecar(
            output,
            corrupted,
            payload,
            APPROVED_ARTIFACT_SHA256,
            APPROVED_DIRECTION_SHA256,
        )
    assert output.read_bytes() == previous


def test_sidecar_reader_rejects_unknown_manifest_field(tmp_path: Path) -> None:
    records, payload = _fixture_records_payload()
    manifest = [record.to_json() for record in records]
    manifest[0]["unexpected"] = True
    sidecar = _write_raw_sidecar(tmp_path / "unknown-field.ninferproj", manifest, payload)

    with pytest.raises(SidecarError, match="missing or extra members"):
        SidecarReader(sidecar)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("overlap", "overlap"),
        ("unaligned", "aligned"),
        ("out_of_range", "beyond payload"),
    ),
)
def test_sidecar_reader_rejects_corrupt_payload_spans(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    records, payload = _fixture_records_payload()
    manifest = [record.to_json() for record in records]
    if mutation == "overlap":
        manifest[1]["direction_offset"] = manifest[0]["direction_offset"]
    elif mutation == "unaligned":
        manifest[0]["direction_offset"] = 1
    else:
        manifest[0]["signature_offset"] = len(payload)
    sidecar = _write_raw_sidecar(tmp_path / f"{mutation}.ninferproj", manifest, payload)

    with pytest.raises(SidecarError, match=message):
        SidecarReader(sidecar)


def test_verifier_rejects_wrong_writer_inventory_before_artifact_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, payload = _fixture_records_payload()
    manifest = [record.to_json() for record in records]
    manifest[0]["key"] = "model.layers.64.mlp.down_proj"
    sidecar = _write_raw_sidecar(tmp_path / "wrong-writer.ninferproj", manifest, payload)
    monkeypatch.setattr(
        verify_module.Artifact,
        "open",
        lambda _path: pytest.fail("artifact must not open for an unclaimed writer"),
    )

    with pytest.raises(SidecarError, match="unclaimed projection record"):
        verify_sidecar(sidecar, tmp_path / "unused.ninfer", tmp_path / "unused.safetensors")


@pytest.mark.parametrize("field", ("artifact", "directions"))
def test_verifier_rejects_unapproved_header_hash_before_artifact_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    records, payload = _fixture_records_payload()
    kwargs = {"artifact_sha256": APPROVED_ARTIFACT_SHA256, "direction_sha256": APPROVED_DIRECTION_SHA256}
    kwargs["direction_sha256" if field == "directions" else "artifact_sha256"] = "0" * 64
    sidecar = _write_raw_sidecar(tmp_path / f"wrong-{field}-hash.ninferproj", records, payload, **kwargs)
    monkeypatch.setattr(
        verify_module.Artifact,
        "open",
        lambda _path: pytest.fail("artifact must not open for an unapproved input hash"),
    )

    with pytest.raises(SidecarError, match="approved"):
        verify_sidecar(sidecar, tmp_path / "unused.ninfer", tmp_path / "unused.safetensors")


def test_builder_rejects_wrong_artifact_hash_before_useful_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSource:
        sha256 = APPROVED_DIRECTION_SHA256
        keys = tuple(sorted(direction_key_for_writer(key) for key in expected_writer_keys()))

        @staticmethod
        def coefficients() -> tuple[float, ...]:
            return (1.0,) * 128

    monkeypatch.setattr(build_module, "load_direction_source", lambda _path: FakeSource())
    monkeypatch.setattr(build_module, "sha256_file", lambda _path: "0" * 64)
    monkeypatch.setattr(
        build_module.Artifact,
        "open",
        lambda _path: pytest.fail("artifact must not open for a wrong approved hash"),
    )

    with pytest.raises(SidecarError, match="approved Qwen3.8 artifact"):
        build_module.build_sidecar(
            tmp_path / "artifact.ninfer",
            tmp_path / "directions.safetensors",
            tmp_path / "output.ninferproj",
            run_guard=False,
        )


def test_builder_rejects_wrong_direction_hash_before_useful_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSource:
        sha256 = "0" * 64
        keys = tuple(sorted(direction_key_for_writer(key) for key in expected_writer_keys()))

        @staticmethod
        def coefficients() -> tuple[float, ...]:
            return (1.0,) * 128

    def digest(path: Path) -> str:
        return APPROVED_ARTIFACT_SHA256 if path.name == "artifact.ninfer" else "0" * 64

    monkeypatch.setattr(build_module, "load_direction_source", lambda _path: FakeSource())
    monkeypatch.setattr(build_module, "sha256_file", digest)
    monkeypatch.setattr(
        build_module.Artifact,
        "open",
        lambda _path: pytest.fail("artifact must not open for a wrong approved direction hash"),
    )

    with pytest.raises(SidecarError, match="approved Qwen3.8 source"):
        build_module.build_sidecar(
            tmp_path / "artifact.ninfer",
            tmp_path / "directions.safetensors",
            tmp_path / "output.ninferproj",
            run_guard=False,
        )


def test_verifier_rejects_one_bit_stored_direction_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direction_words = (1.0,) + (0.0,) * 5119
    records, payload = _fixture_records_payload(direction_words=direction_words)
    payload_hash = hashlib.sha256(b"\x00").hexdigest()
    records = [replace(record, weight_payload_sha256=payload_hash) for record in records]
    sidecar = _write_raw_sidecar(tmp_path / "direction-bit.ninferproj", records, payload)
    raw = bytearray(sidecar.read_bytes())
    with SidecarReader(sidecar) as reader:
        mutation_byte = reader.payload_offset + reader.manifest[0].direction_offset
    raw[mutation_byte] ^= 1
    sidecar.write_bytes(raw)

    class FakeSource:
        path = tmp_path / "directions.safetensors"
        sha256 = APPROVED_DIRECTION_SHA256
        keys = tuple(sorted(direction_key_for_writer(key) for key in expected_writer_keys()))

        @staticmethod
        def coefficients() -> tuple[float, ...]:
            return tuple(record.coefficient for record in records)

        @staticmethod
        def raw_direction(_key: str) -> memoryview:
            return memoryview(struct.pack("<5120f", *direction_words))

    class FakeArtifact(Artifact):
        def __init__(self) -> None:
            self.path = tmp_path / "artifact.ninfer"
            self.identity = type(
                "Identity", (), {"model_id": "qwen3.8-27b", "weights_id": "nvfp4"}
            )()

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        verify_module,
        "sha256_file",
        lambda path: (
            APPROVED_DIRECTION_SHA256
            if Path(path).name == "directions.safetensors"
            else APPROVED_ARTIFACT_SHA256
        ),
    )
    monkeypatch.setattr(verify_module, "load_direction_source", lambda _path: FakeSource())
    monkeypatch.setattr(
        verify_module,
        "_expected_object",
        lambda _artifact, key: TensorObject(
            name=key,
            shape=(5120, 6144 if site_for_writer(key) != "mlp_down" else 17408),
            format=(
                "FP8_E4M3FN_ROW_BF16S"
                if site_for_writer(key) != "mlp_down"
                else "NVFP4"
            ),
            layout="test",
            offset=0,
            bytes=1,
        ),
    )
    monkeypatch.setattr(
        verify_module,
        "_recompute_signature",
        lambda _obj, _payload, _direction: __import__("torch").zeros(_obj.shape[1]),
    )
    FakeArtifact.payload = lambda _self, _obj: memoryview(b"\x00")

    with pytest.raises(SidecarError, match="direction identity"):
        verify_sidecar(sidecar, FakeArtifact(), FakeSource.path)


def test_verifier_rejects_coefficient_mutation_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direction_words = (1.0,) + (0.0,) * 5119
    records, payload = _fixture_records_payload(direction_words=direction_words)
    payload_hash = hashlib.sha256(b"\x00").hexdigest()
    records = [replace(record, weight_payload_sha256=payload_hash) for record in records]
    mutated = [record.to_json() for record in records]
    mutated[0]["coefficient"] = float(mutated[0]["coefficient"]) + 0.125
    sidecar = _write_raw_sidecar(tmp_path / "coefficient.ninferproj", mutated, payload)

    class FakeSource:
        path = tmp_path / "directions.safetensors"
        sha256 = APPROVED_DIRECTION_SHA256
        keys = tuple(sorted(direction_key_for_writer(key) for key in expected_writer_keys()))

        @staticmethod
        def coefficients() -> tuple[float, ...]:
            return tuple(record.coefficient for record in records)

        @staticmethod
        def raw_direction(_key: str) -> memoryview:
            return memoryview(struct.pack("<5120f", *direction_words))

    class FakeArtifact(Artifact):
        def __init__(self) -> None:
            self.path = tmp_path / "artifact.ninfer"
            self.identity = type(
                "Identity", (), {"model_id": "qwen3.8-27b", "weights_id": "nvfp4"}
            )()

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        verify_module,
        "sha256_file",
        lambda path: (
            APPROVED_DIRECTION_SHA256
            if Path(path).name == "directions.safetensors"
            else APPROVED_ARTIFACT_SHA256
        ),
    )
    monkeypatch.setattr(verify_module, "load_direction_source", lambda _path: FakeSource())
    monkeypatch.setattr(
        verify_module,
        "_expected_object",
        lambda _artifact, key: TensorObject(
            name=key,
            shape=(5120, 6144 if site_for_writer(key) != "mlp_down" else 17408),
            format=(
                "FP8_E4M3FN_ROW_BF16S"
                if site_for_writer(key) != "mlp_down"
                else "NVFP4"
            ),
            layout="test",
            offset=0,
            bytes=1,
        ),
    )
    FakeArtifact.payload = lambda _self, _obj: memoryview(b"\x00")

    with pytest.raises(SidecarError, match="coefficient order/value mismatch"):
        verify_sidecar(sidecar, FakeArtifact(), FakeSource.path)
