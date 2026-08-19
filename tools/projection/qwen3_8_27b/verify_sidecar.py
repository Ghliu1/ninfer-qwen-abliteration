"""Independently verify every Qwen3.8 refusal-projection signature."""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path

import torch

from tools.artifact.container import Artifact, TensorObject
from tools.artifact.layouts import (
    decode_fp8_row_scaled_words,
    decode_nvfp4_words,
    encoded_size,
)
from tools.artifact.numeric import decode_e2m1_word, decode_e4m3fn_word
from tools.convert.qwen3_8_27b import inventory_nvfp4 as inventory

from .sidecar import (
    APPROVED_ARTIFACT_SHA256,
    APPROVED_DIRECTION_SHA256,
    DIRECTION_COUNT,
    SidecarError,
    SidecarReader,
    atomic_write_json,
    load_direction_source,
    sha256_file,
)


DECODER_REVISION = "a05746aa:tools.artifact.layouts.fp8-row/nvfp4"
STORAGE_GUARD = Path("/home/georg/openclaw-scripts/maintenance/storage-guard.py")
STORAGE_ROOT = Path("/mnt/h/OpenClawLab")
STORAGE_RESERVE_BYTES = 64 * 1024 * 1024
ROW_CHUNK = 128

_E2M1_LUT = torch.tensor(
    [decode_e2m1_word(index) for index in range(16)], dtype=torch.float32
)
_E4M3_LUT = torch.tensor(
    [decode_e4m3fn_word(index) for index in range(256)], dtype=torch.float32
)
_EXPECTED_TENSOR_SPECS = {spec.name: spec for spec in inventory.TENSOR_SPECS}


def _expected_writer_keys_independent() -> tuple[str, ...]:
    keys: list[str] = []
    for layer in range(64):
        mixer = "self_attn.o_proj" if layer % 4 == 3 else "linear_attn.out_proj"
        keys.extend((f"model.layers.{layer}.{mixer}", f"model.layers.{layer}.mlp.down_proj"))
    return tuple(keys)


def _direction_key(key: str) -> str:
    if not key.startswith("model."):
        raise SidecarError(f"unsupported projection record key: {key}")
    return "model.language_model." + key[len("model.") :]


def _artifact_name(key: str) -> str:
    parts = key.split(".")
    if len(parts) != 5 or parts[:2] != ["model", "layers"]:
        raise SidecarError(f"unsupported projection record key: {key}")
    try:
        layer = int(parts[2])
    except ValueError as exc:
        raise SidecarError(f"unsupported projection layer: {key}") from exc
    suffix = ".".join(parts[3:])
    if suffix == "linear_attn.out_proj":
        return f"text/layers/{layer}/gdn/output"
    if suffix == "self_attn.o_proj":
        return f"text/layers/{layer}/attention/output"
    if suffix == "mlp.down_proj":
        return f"text/layers/{layer}/mlp/down"
    raise SidecarError(f"unsupported projection record key: {key}")


def _site(key: str) -> str:
    if key.endswith("linear_attn.out_proj"):
        return "gdn_output"
    if key.endswith("self_attn.o_proj"):
        return "attention_output"
    if key.endswith("mlp.down_proj"):
        return "mlp_down"
    raise SidecarError(f"unsupported projection record key: {key}")


def _expected_object(artifact: Artifact, key: str) -> TensorObject:
    name = _artifact_name(key)
    try:
        obj = artifact.find(name)
    except KeyError as exc:
        raise SidecarError(f"artifact is missing projection tensor {name}") from exc
    if not isinstance(obj, TensorObject):
        raise SidecarError(f"artifact projection object is not a tensor: {name}")
    expected = _EXPECTED_TENSOR_SPECS.get(name)
    if expected is None:
        raise SidecarError(f"artifact contract is missing projection tensor {name}")
    if (
        tuple(obj.shape) != tuple(expected.shape)
        or obj.format != expected.format
        or obj.layout != expected.layout
        or obj.bytes != encoded_size(expected.layout, expected.format, expected.shape)
    ):
        raise SidecarError(
            f"artifact descriptor mismatch for {name}: "
            f"got shape={obj.shape} format={obj.format} layout={obj.layout} "
            f"bytes={obj.bytes}; "
            f"expected shape={expected.shape} format={expected.format} "
            f"layout={expected.layout} "
            f"bytes={encoded_size(expected.layout, expected.format, expected.shape)}"
        )
    return obj


def _read_f32(view: memoryview, count: int, label: str) -> torch.Tensor:
    if view.nbytes != count * 4:
        raise SidecarError(f"{label} has {view.nbytes} bytes; expected {count * 4}")
    tensor = torch.frombuffer(bytearray(view), dtype=torch.float32).clone()
    if tensor.numel() != count or not bool(torch.isfinite(tensor).all()):
        raise SidecarError(f"{label} is not finite FP32")
    return tensor


def _canonical_f32_bytes(value: torch.Tensor) -> bytes:
    """Serialize a CPU FP32 tensor in the sidecar's little-endian form."""

    return value.detach().contiguous().to(torch.float32).numpy().astype(
        "<f4", copy=False
    ).tobytes()


def _recompute_fp8(payload, shape: tuple[int, int], direction: torch.Tensor) -> torch.Tensor:
    codes, row_scales = decode_fp8_row_scaled_words(payload, shape)
    rows, columns = shape
    result = torch.zeros(columns, dtype=torch.float32)
    for begin in range(0, rows, ROW_CHUNK):
        end = min(rows, begin + ROW_CHUNK)
        decoded = codes[begin:end].view(torch.float8_e4m3fn).float()
        decoded.mul_(row_scales[begin:end].float().unsqueeze(1))
        result.add_(decoded.transpose(0, 1).matmul(direction[begin:end]))
        del decoded
    del codes, row_scales
    return result


def _recompute_nvfp4(payload, shape: tuple[int, int], direction: torch.Tensor) -> torch.Tensor:
    packed, scales, divisor = decode_nvfp4_words(payload, shape)
    rows, columns = shape
    result = torch.zeros(columns, dtype=torch.float32)
    inverse_divisor = torch.reciprocal(divisor.float())
    for begin in range(0, rows, ROW_CHUNK):
        end = min(rows, begin + ROW_CHUNK)
        chunk = packed[begin:end]
        low = chunk & 0x0F
        high = chunk >> 4
        codes = torch.stack((low, high), dim=2).reshape(end - begin, columns)
        decoded = _E2M1_LUT[codes.long()]
        scale = _E4M3_LUT[scales[begin:end].long()]
        scale.mul_(inverse_divisor)
        decoded.mul_(scale.repeat_interleave(16, dim=1))
        result.add_(decoded.transpose(0, 1).matmul(direction[begin:end]))
        del chunk, low, high, codes, decoded, scale
    del packed, scales, divisor, inverse_divisor
    return result


def _recompute_signature(obj: TensorObject, payload, direction: torch.Tensor) -> torch.Tensor:
    if obj.format == "FP8_E4M3FN_ROW_BF16S":
        return _recompute_fp8(payload, tuple(obj.shape), direction)
    if obj.format == "NVFP4":
        return _recompute_nvfp4(payload, tuple(obj.shape), direction)
    raise SidecarError(f"unsupported artifact projection format: {obj.format}")


def _run_storage_guard(
    storage_root: Path,
    guard_path: Path,
    reserve_bytes: int = STORAGE_RESERVE_BYTES,
) -> str:
    import subprocess
    import sys

    if not guard_path.is_file():
        raise SidecarError(f"maintained storage guard is missing: {guard_path}")
    result = subprocess.run(
        [
            sys.executable,
            str(guard_path),
            "check",
            "--operation",
            "model-download",
            "--path",
            str(storage_root),
            "--download-bytes",
            "0",
            "--extracted-bytes",
            str(reserve_bytes),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or '"decision": "allow"' not in output:
        raise SidecarError(f"storage guard denied report write: {output}")
    print(output)
    return output


def verify_sidecar(
    sidecar_path: str | Path,
    artifact: str | Path | Artifact,
    directions_path: str | Path | None = None,
    *,
    report_path: str | Path | None = None,
    run_guard: bool = False,
    storage_root: str | Path = STORAGE_ROOT,
    storage_guard: str | Path = STORAGE_GUARD,
) -> dict[str, object]:
    """Verify inventory, source hashes, decodes, and all 128 signatures."""

    sidecar_path = Path(sidecar_path)
    with SidecarReader(sidecar_path) as reader:
        if reader.artifact_sha256 != APPROVED_ARTIFACT_SHA256:
            raise SidecarError("sidecar artifact SHA-256 is not the approved Qwen3.8 artifact")
        if reader.direction_sha256 != APPROVED_DIRECTION_SHA256:
            raise SidecarError("sidecar direction SHA-256 is not the approved Qwen3.8 source")
        expected_keys = _expected_writer_keys_independent()
        actual_keys = [record.key for record in reader.manifest]
        expected_set = set(expected_keys)
        actual_set = set(actual_keys)
        for key in actual_keys:
            if key not in expected_set:
                raise SidecarError(f"unclaimed projection record: {key}")
        missing = [key for key in expected_keys if key not in actual_set]
        if missing:
            raise SidecarError(f"missing projection record: {missing[0]}")
        if len(actual_keys) != len(actual_set):
            raise SidecarError("duplicate projection record")
        if len(reader.manifest) != 128:
            raise SidecarError("projection inventory must contain 128 records")
        if actual_keys != list(expected_keys):
            raise SidecarError("projection inventory order mismatch")

        if directions_path is None:
            candidate = sidecar_path.parent / "refusal_dirs_qwen38.safetensors"
            directions_path = candidate
        directions_path = Path(directions_path)
        direction_digest = sha256_file(directions_path)
        if direction_digest != APPROVED_DIRECTION_SHA256:
            raise SidecarError("direction SHA-256 is not the approved Qwen3.8 source")
        source = load_direction_source(directions_path)
        if source.sha256 != direction_digest:
            raise SidecarError("direction source changed while it was being loaded")

        if isinstance(artifact, Artifact):
            artifact_context = artifact
            owns_artifact = False
            artifact_path = artifact_context.path
            artifact_digest = sha256_file(artifact_path)
        else:
            artifact_path = Path(artifact)
            artifact_digest = sha256_file(artifact_path)
            if artifact_digest != APPROVED_ARTIFACT_SHA256:
                raise SidecarError("artifact SHA-256 is not the approved Qwen3.8 artifact")
            artifact_context = Artifact.open(artifact_path)
            owns_artifact = True
        if artifact_digest != APPROVED_ARTIFACT_SHA256:
            raise SidecarError("artifact SHA-256 is not the approved Qwen3.8 artifact")
        try:
            if artifact_context.identity.model_id != "qwen3.8-27b" or artifact_context.identity.weights_id != "nvfp4":
                raise SidecarError("artifact identity does not match qwen3.8-27b/nvfp4")
            if artifact_digest != reader.artifact_sha256:
                raise SidecarError("sidecar artifact SHA-256 does not match the supplied artifact")
            if source.sha256 != reader.direction_sha256:
                raise SidecarError("sidecar direction SHA-256 does not match the supplied directions")
            coefficients = dict(zip(source.keys, source.coefficients(), strict=True))
            max_error = 0.0
            format_counts: dict[str, int] = {"fp8_row": 0, "nvfp4_block": 0}
            site_counts: dict[str, int] = {
                "attention_output": 0,
                "gdn_output": 0,
                "mlp_down": 0,
            }
            for record in reader.manifest:
                obj = _expected_object(artifact_context, record.key)
                expected_format = "fp8_row" if obj.format == "FP8_E4M3FN_ROW_BF16S" else "nvfp4_block"
                if record.weight_format != expected_format:
                    raise SidecarError(f"weight format metadata mismatch for {record.key}")
                if record.site != _site(record.key):
                    raise SidecarError(f"site metadata mismatch for {record.key}")
                if record.layer != int(record.key.split(".")[2]):
                    raise SidecarError(f"layer metadata mismatch for {record.key}")
                payload = artifact_context.payload(obj)
                payload_hash = hashlib.sha256(payload).hexdigest()
                if payload_hash != record.weight_payload_sha256:
                    raise SidecarError(f"weight payload SHA-256 mismatch for {record.key}")
                direction_key = _direction_key(record.key)
                try:
                    source_coefficient = coefficients[direction_key]
                except KeyError as exc:
                    raise SidecarError(f"direction source is missing {direction_key}") from exc
                if record.coefficient != float(source_coefficient):
                    raise SidecarError(f"coefficient order/value mismatch for {record.key}")
                raw_direction = _read_f32(
                    source.raw_direction(direction_key), DIRECTION_COUNT, f"direction {direction_key}"
                )
                norm = torch.linalg.vector_norm(raw_direction)
                if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
                    raise SidecarError(f"direction {direction_key} has zero or invalid norm")
                expected_direction = raw_direction / norm
                stored_direction_view = reader.record_payload(record, "direction")
                try:
                    if bytes(stored_direction_view) != _canonical_f32_bytes(expected_direction):
                        raise SidecarError(f"direction identity mismatch for {record.key}")
                    stored_direction = _read_f32(
                        stored_direction_view,
                        DIRECTION_COUNT,
                        f"sidecar direction {record.key}",
                    )
                finally:
                    stored_direction_view.release()
                stored_norm = torch.linalg.vector_norm(stored_direction)
                if not bool(torch.isfinite(stored_norm)) or abs(float(stored_norm) - 1.0) > 1e-5:
                    raise SidecarError(f"sidecar direction is not unit length for {record.key}")
                expected_signature = _recompute_signature(obj, payload, stored_direction)
                stored_signature = _read_f32(
                    reader.record_payload(record, "signature"),
                    int(obj.shape[1]),
                    f"sidecar signature {record.key}",
                )
                try:
                    torch.testing.assert_close(
                        stored_signature,
                        expected_signature,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                except AssertionError as exc:
                    raise SidecarError(f"signature mismatch for {record.key}: {exc}") from exc
                max_error = max(
                    max_error,
                    float(torch.max(torch.abs(stored_signature - expected_signature))),
                )
                format_counts[record.weight_format] += 1
                site_counts[record.site] += 1
                del payload, raw_direction, expected_direction, stored_direction
            result = {
                "schema_version": 1,
                "artifact": {
                    "path": str(artifact_path),
                    "bytes": artifact_path.stat().st_size,
                    "sha256": artifact_digest,
                },
                "directions": {
                    "path": str(source.path),
                    "bytes": source.path.stat().st_size,
                    "sha256": source.sha256,
                },
                "sidecar": {
                    "path": str(sidecar_path),
                    "bytes": sidecar_path.stat().st_size,
                    "sha256": sha256_file(sidecar_path),
                },
                "records": 128,
                "unclaimed_records": 0,
                "weight_formats": format_counts,
                "sites": site_counts,
                "max_signature_error": max_error,
                "decoder_revision": DECODER_REVISION,
            }
        finally:
            exported_payload = locals().get("payload")
            if isinstance(exported_payload, memoryview):
                exported_payload.release()
            gc.collect()
            if owns_artifact:
                artifact_context.close()
    if report_path is None:
        report_path = sidecar_path.with_suffix(".conversion.json")
    if run_guard:
        _run_storage_guard(Path(storage_root), Path(storage_guard))
    atomic_write_json(report_path, result)
    # Reopen the metadata-only report to ensure the published bytes are valid JSON.
    import json

    with Path(report_path).open("r", encoding="utf-8") as stream:
        json.load(stream)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--directions", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-storage-guard", action="store_true")
    parser.add_argument("--storage-root", type=Path, default=STORAGE_ROOT)
    parser.add_argument("--storage-guard", type=Path, default=STORAGE_GUARD)
    args = parser.parse_args()
    result = verify_sidecar(
        args.sidecar,
        args.artifact,
        args.directions,
        report_path=args.report,
        run_guard=not args.skip_storage_guard,
        storage_root=args.storage_root,
        storage_guard=args.storage_guard,
    )
    print(
        "verified sidecar: records={records} max_signature_error={max_signature_error:.6g} "
        "report={report}".format(report=args.report or args.sidecar.with_suffix(".conversion.json"), **result)
    )


if __name__ == "__main__":
    main()
