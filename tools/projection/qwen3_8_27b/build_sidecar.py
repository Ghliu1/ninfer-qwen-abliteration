"""Build the exact Qwen3.8 refusal-projection sidecar."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
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
    DIRECTION_COUNT,
    SidecarError,
    SidecarRecord,
    align_up,
    artifact_name_for_writer,
    direction_key_for_writer,
    expected_writer_keys,
    load_direction_source,
    sha256_file,
    site_for_writer,
    write_sidecar,
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


def _validate_artifact_object(artifact: Artifact, writer_key: str) -> TensorObject:
    object_name = artifact_name_for_writer(writer_key)
    try:
        obj = artifact.find(object_name)
    except KeyError as exc:
        raise SidecarError(f"artifact is missing projection tensor {object_name}") from exc
    if not isinstance(obj, TensorObject):
        raise SidecarError(f"artifact projection object is not a tensor: {object_name}")
    expected = _EXPECTED_TENSOR_SPECS.get(object_name)
    if expected is None:
        raise SidecarError(f"artifact contract is missing projection tensor {object_name}")
    if (
        tuple(obj.shape) != tuple(expected.shape)
        or obj.format != expected.format
        or obj.layout != expected.layout
        or obj.bytes != encoded_size(expected.layout, expected.format, expected.shape)
    ):
        raise SidecarError(
            f"artifact descriptor mismatch for {object_name}: "
            f"got shape={obj.shape} format={obj.format} layout={obj.layout} "
            f"bytes={obj.bytes}; "
            f"expected shape={expected.shape} format={expected.format} "
            f"layout={expected.layout} "
            f"bytes={encoded_size(expected.layout, expected.format, expected.shape)}"
        )
    return obj


def _normalized_direction(source, direction_key: str) -> torch.Tensor:
    raw = torch.frombuffer(bytearray(source.raw_direction(direction_key)), dtype=torch.float32).clone()
    if raw.numel() != DIRECTION_COUNT or not bool(torch.isfinite(raw).all()):
        raise SidecarError(f"direction {direction_key} is not finite F32[5120]")
    norm = torch.linalg.vector_norm(raw)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise SidecarError(f"direction {direction_key} has zero or invalid norm")
    return raw / norm


def _signature_fp8(payload, shape: tuple[int, int], direction: torch.Tensor) -> torch.Tensor:
    codes, row_scales = decode_fp8_row_scaled_words(payload, shape)
    rows, columns = shape
    signature = torch.zeros(columns, dtype=torch.float32)
    for begin in range(0, rows, ROW_CHUNK):
        end = min(rows, begin + ROW_CHUNK)
        decoded = codes[begin:end].view(torch.float8_e4m3fn).float()
        decoded.mul_(row_scales[begin:end].float().unsqueeze(1))
        signature.add_(decoded.transpose(0, 1).matmul(direction[begin:end]))
        del decoded
    del codes, row_scales
    return signature


def _signature_nvfp4(payload, shape: tuple[int, int], direction: torch.Tensor) -> torch.Tensor:
    packed, scales, divisor = decode_nvfp4_words(payload, shape)
    rows, columns = shape
    signature = torch.zeros(columns, dtype=torch.float32)
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
        signature.add_(decoded.transpose(0, 1).matmul(direction[begin:end]))
        del chunk, low, high, codes, decoded, scale
    del packed, scales, divisor, inverse_divisor
    return signature


def compute_signature(obj: TensorObject, payload, direction: torch.Tensor) -> torch.Tensor:
    """Decode one served payload and compute its independent FP32 signature."""

    if tuple(direction.shape) != (DIRECTION_COUNT,):
        raise SidecarError("projection direction must have shape [5120]")
    if obj.format == "FP8_E4M3FN_ROW_BF16S":
        return _signature_fp8(payload, tuple(obj.shape), direction)
    if obj.format == "NVFP4":
        return _signature_nvfp4(payload, tuple(obj.shape), direction)
    raise SidecarError(f"unsupported projection decoder format: {obj.format}")


def _validate_storage_guard(
    storage_root: Path = STORAGE_ROOT,
    guard_path: Path = STORAGE_GUARD,
    reserve_bytes: int = STORAGE_RESERVE_BYTES,
) -> str:
    if reserve_bytes < 0:
        raise SidecarError("storage guard reservation must be nonnegative")
    if not guard_path.is_file():
        raise SidecarError(f"maintained storage guard is missing: {guard_path}")
    command = [
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
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or '"decision": "allow"' not in output:
        raise SidecarError(f"storage guard denied sidecar write: {output}")
    print(output)
    return output


def _append_aligned(stream, offset: int, data: bytes) -> int:
    aligned = align_up(offset, PAYLOAD_ALIGNMENT)
    if aligned > offset:
        stream.write(b"\x00" * (aligned - offset))
    stream.write(data)
    return aligned + len(data)


def build_sidecar(
    artifact_path: str | Path,
    directions_path: str | Path,
    output_path: str | Path,
    *,
    run_guard: bool = True,
    storage_root: str | Path = STORAGE_ROOT,
    storage_guard: str | Path = STORAGE_GUARD,
) -> dict[str, object]:
    """Build and reopen the sidecar, streaming one artifact weight at a time."""

    artifact_path = Path(artifact_path)
    directions_path = Path(directions_path)
    output_path = Path(output_path)
    source = load_direction_source(directions_path)
    coefficients = dict(zip(source.keys, source.coefficients(), strict=True))
    artifact_digest = sha256_file(artifact_path)
    if run_guard:
        _validate_storage_guard(Path(storage_root), Path(storage_guard))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_temp = output_path.with_name(f".{output_path.name}.payload.tmp")
    records: list[SidecarRecord] = []
    payload_offset = 0
    try:
        with Artifact.open(artifact_path) as artifact:
            if artifact.identity.model_id != "qwen3.8-27b" or artifact.identity.weights_id != "nvfp4":
                raise SidecarError("artifact identity does not match qwen3.8-27b/nvfp4")
            with payload_temp.open("wb") as payload_stream:
                for writer_key in expected_writer_keys():
                    direction_key = direction_key_for_writer(writer_key)
                    obj = _validate_artifact_object(artifact, writer_key)
                    payload_view = artifact.payload(obj)
                    try:
                        payload_hash = hashlib.sha256(payload_view).hexdigest()
                        direction = _normalized_direction(source, direction_key)
                        direction_bytes = direction.numpy().tobytes()
                        direction_offset = align_up(payload_offset, PAYLOAD_ALIGNMENT)
                        payload_offset = _append_aligned(payload_stream, payload_offset, direction_bytes)
                        signature = compute_signature(obj, payload_view, direction)
                    finally:
                        payload_view.release()
                    signature_bytes = signature.numpy().tobytes()
                    signature_offset = align_up(payload_offset, PAYLOAD_ALIGNMENT)
                    payload_offset = _append_aligned(payload_stream, payload_offset, signature_bytes)
                    records.append(
                        SidecarRecord(
                            key=writer_key,
                            layer=int(writer_key.split(".")[2]),
                            site=site_for_writer(writer_key),
                            coefficient=float(coefficients[direction_key]),
                            direction_offset=direction_offset,
                            direction_count=DIRECTION_COUNT,
                            signature_offset=signature_offset,
                            signature_count=int(obj.shape[1]),
                            weight_format=(
                                "fp8_row"
                                if obj.format == "FP8_E4M3FN_ROW_BF16S"
                                else "nvfp4_block"
                            ),
                            weight_payload_sha256=payload_hash,
                        )
                    )
                    del direction, signature, direction_bytes, signature_bytes
                payload_stream.flush()
                os.fsync(payload_stream.fileno())
        reader = write_sidecar(
            output_path,
            records,
            payload_temp,
            artifact_digest,
            source.sha256,
        )
        reader.close()
        return {
            "artifact_sha256": artifact_digest,
            "direction_sha256": source.sha256,
            "records": len(records),
            "payload_bytes": payload_offset,
            "sidecar_bytes": output_path.stat().st_size,
            "decoder_revision": DECODER_REVISION,
        }
    finally:
        payload_temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--directions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-storage-guard", action="store_true")
    parser.add_argument("--storage-root", type=Path, default=STORAGE_ROOT)
    parser.add_argument("--storage-guard", type=Path, default=STORAGE_GUARD)
    args = parser.parse_args()
    result = build_sidecar(
        args.artifact,
        args.directions,
        args.output,
        run_guard=not args.skip_storage_guard,
        storage_root=args.storage_root,
        storage_guard=args.storage_guard,
    )
    print(
        "built sidecar: records={records} sidecar_bytes={sidecar_bytes} "
        "artifact_sha256={artifact_sha256} direction_sha256={direction_sha256}".format(
            **result
        )
    )


if __name__ == "__main__":
    main()
