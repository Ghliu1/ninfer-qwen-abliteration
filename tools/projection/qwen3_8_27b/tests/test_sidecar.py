from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from tools.projection.qwen3_8_27b.sidecar import (
    SidecarError,
    HEADER,
    SidecarReader,
    SidecarRecord,
    expected_writer_keys,
    site_for_writer,
    write_sidecar,
)
from tools.projection.qwen3_8_27b.verify_sidecar import verify_sidecar
from tools.artifact.container import TensorObject
from tools.projection.qwen3_8_27b.build_sidecar import _validate_artifact_object


def test_expected_writer_keys_are_complete() -> None:
    keys = expected_writer_keys()
    assert len(keys) == 128
    assert keys[0] == "model.layers.0.linear_attn.out_proj"
    assert "model.layers.3.self_attn.o_proj" in keys
    assert "model.layers.63.mlp.down_proj" in keys
    assert len(set(keys)) == 128


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
        b"a" * 32,
        b"b" * 32,
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
