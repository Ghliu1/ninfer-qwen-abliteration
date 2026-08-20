from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from tools.projection.qwen3_8_27b.sidecar import (
    APPROVED_ARTIFACT_SHA256,
    APPROVED_DIRECTION_SHA256,
    HEADER,
    PAYLOAD_ALIGNMENT,
    SidecarRecord,
    align_up,
    expected_writer_keys,
    site_for_writer,
)


def records_and_payload() -> tuple[list[SidecarRecord], bytes]:
    records: list[SidecarRecord] = []
    payload = bytearray()
    for index, key in enumerate(expected_writer_keys()):
        layer = int(key.split(".")[2])
        site = site_for_writer(key)
        direction_offset = align_up(len(payload))
        payload.extend(b"\0" * (direction_offset - len(payload)))
        payload.extend(struct.pack("<f", 1.0))
        payload.extend(b"\0" * (5120 * 4 - 4))
        signature_count = 17408 if site == "mlp_down" else 6144
        signature_offset = align_up(len(payload))
        payload.extend(b"\0" * (signature_offset - len(payload)))
        payload.extend(struct.pack("<f", index + 0.25))
        payload.extend(b"\0" * (signature_count * 4 - 4))
        records.append(
            SidecarRecord(
                key=key,
                layer=layer,
                site=site,
                coefficient=index + 0.5,
                direction_offset=direction_offset,
                direction_count=5120,
                signature_offset=signature_offset,
                signature_count=signature_count,
                weight_format=(
                    "fp8_row"
                    if site != "mlp_down" or layer >= 56
                    else "nvfp4_block"
                ),
                weight_payload_sha256=f"{index:064x}",
            )
        )
    return records, bytes(payload)


def write_raw(path: Path, values: list[dict[str, object]], payload: bytes, mode: str) -> None:
    manifest = json.dumps(
        values, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if mode == "literal_duplicate_key":
        manifest = manifest.replace(b'"key":', b'"key":"duplicate","key":', 1)
    payload_offset = align_up(HEADER.size + len(manifest), PAYLOAD_ALIGNMENT)
    manifest_sha = hashlib.sha256(manifest).digest()
    artifact_sha = bytes.fromhex(APPROVED_ARTIFACT_SHA256)
    direction_sha = bytes.fromhex(APPROVED_DIRECTION_SHA256)
    version = 1
    magic = b"NINFRP1\0"
    manifest_bytes = len(manifest)
    payload_bytes = len(payload)
    if mode == "wrong_artifact_hash":
        artifact_sha = b"\0" * 32
    elif mode == "wrong_direction_hash":
        direction_sha = b"\0" * 32
    elif mode == "bad_manifest_hash":
        manifest_sha = b"\0" * 32
    elif mode == "wrong_version":
        version = 2
    elif mode == "bad_magic":
        magic = b"BADMAGIC"
    elif mode == "oversized_manifest":
        manifest_bytes = 2 * 1024 * 1024
    elif mode == "oversized_payload":
        payload_bytes = 64 * 1024 * 1024
    header = HEADER.pack(
        magic,
        version,
        manifest_bytes,
        payload_bytes,
        manifest_sha,
        artifact_sha,
        direction_sha,
    )
    path.write_bytes(
        header
        + manifest
        + b"\0" * (payload_offset - HEADER.size - len(manifest))
        + payload
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("mode")
    args = parser.parse_args()
    records, payload = records_and_payload()
    values = [record.to_json() for record in records]
    mode = args.mode
    if mode == "missing":
        values.pop()
    elif mode == "lambda":
        values[0]["lambda"] = 1.0
    elif mode == "override":
        values[0]["coefficient_override"] = 1.0
    elif mode == "duplicate":
        values[1]["key"] = values[0]["key"]
        values[1]["layer"] = values[0]["layer"]
        values[1]["site"] = values[0]["site"]
    elif mode == "wrong_layer":
        values[0]["layer"] = 1
    elif mode == "wrong_site":
        values[0]["site"] = "attention_output"
    elif mode == "wrong_order":
        values[0], values[1] = values[1], values[0]
    elif mode == "overlap":
        values[1]["direction_offset"] = values[0]["direction_offset"]
    elif mode == "unaligned":
        values[0]["direction_offset"] = 4
    elif mode == "out_of_range":
        values[0]["signature_offset"] = len(payload)
    elif mode == "wrong_count":
        values[0]["direction_count"] = 5119
    elif mode == "wrong_format":
        values[0]["weight_format"] = "nvfp4_block"
    elif mode == "arbitrary_weight_hash":
        values[0]["weight_payload_sha256"] = "f" * 64
    elif mode in {
        "finite_direction_mutation",
        "finite_signature_mutation",
        "zero_direction",
        "nonunit_direction",
    }:
        payload = bytearray(payload)
        if mode == "finite_direction_mutation":
            struct.pack_into("<f", payload, records[0].direction_offset, -1.0)
        elif mode == "finite_signature_mutation":
            struct.pack_into("<f", payload, records[0].signature_offset, 1.25)
        elif mode == "zero_direction":
            struct.pack_into("<f", payload, records[0].direction_offset, 0.0)
        else:
            struct.pack_into("<f", payload, records[0].direction_offset, 0.5)
        payload = bytes(payload)
    elif mode not in {
        "valid",
        "literal_duplicate_key",
        "wrong_artifact_hash",
        "wrong_direction_hash",
        "bad_manifest_hash",
        "wrong_version",
        "bad_magic",
        "oversized_manifest",
        "oversized_payload",
    }:
        raise ValueError(f"unknown fixture mode: {mode}")
    write_raw(args.output, values, payload, mode)


if __name__ == "__main__":
    main()
