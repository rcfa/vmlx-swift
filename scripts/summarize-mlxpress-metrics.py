#!/usr/bin/env python3
"""Summarize MLXPress metrics JSONL artifacts.

The CLI intentionally writes append-only JSON Lines so long model runs do not
lose all telemetry if a process exits early. This helper turns one or more
`metrics.jsonl` files into stable comparison rows for MLXPress scheduler,
direct-kernel, and cache-stack experiments.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _optional_mb(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value) / (1024.0 * 1024.0)
    return None


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"{path}:{line_number}: expected JSON object")
            records.append(record)
    return records


def _summarize(path: Path, top: int) -> dict[str, Any]:
    records = _load_records(path)
    turns = [record for record in records if record.get("type") == "turn"]
    memory = {record.get("phase"): record for record in records if record.get("type") == "memory"}
    gates = {
        record.get("phase"): record
        for record in records
        if record.get("type") == "activity_gate"
    }
    pressures = {
        record.get("phase"): record
        for record in records
        if record.get("type") == "system_pressure"
    }
    profiles = [
        record for record in records if record.get("type") == "streaming_profile"
    ]
    file_read_pressures = [
        record for record in records if record.get("type") == "file_read_pressure"
    ]
    effective_read_pressures = [
        record for record in records if record.get("type") == "effective_read_pressure"
    ]
    active_expert_traces = [
        record for record in records if record.get("type") == "active_expert_trace"
    ]
    active_slice_residencies = [
        record for record in records if record.get("type") == "active_slice_residency"
    ]
    run_start = next(
        (record for record in records if record.get("type") == "run_start"),
        {},
    )

    prompt_tokens = 0
    prompt_seconds = 0.0
    generation_tokens = 0
    generation_seconds = 0.0
    coherent = True
    turn_summaries: list[dict[str, Any]] = []
    for turn in turns:
        telemetry = turn.get("telemetry") if isinstance(turn.get("telemetry"), dict) else {}
        coherency = turn.get("coherency") if isinstance(turn.get("coherency"), dict) else {}
        prompt_tokens += int(telemetry.get("prompt_tokens") or 0)
        prompt_seconds += _float(telemetry.get("prompt_time_seconds"))
        generation_tokens += int(telemetry.get("generation_tokens") or 0)
        generation_seconds += _float(telemetry.get("generation_time_seconds"))
        coherent = coherent and bool(coherency.get("passed"))
        turn_summaries.append(
            {
                "turn": turn.get("turn"),
                "tokens_per_second": _float(telemetry.get("tokens_per_second")),
                "generation_tokens": int(telemetry.get("generation_tokens") or 0),
                "prompt_tokens_per_second": _float(
                    telemetry.get("prompt_tokens_per_second")
                ),
                "coherent": bool(coherency.get("passed")),
                "stop_reason": telemetry.get("stop_reason"),
                "visible_preview": turn.get("visible_preview") or "",
            }
        )

    latest_profile = profiles[-1] if profiles else {}
    rows = latest_profile.get("rows") if isinstance(latest_profile.get("rows"), list) else []
    profile_rows = [
        row
        for row in rows
        if isinstance(row, dict)
    ]
    profile_rows.sort(key=lambda row: _float(row.get("milliseconds")), reverse=True)
    profile_read_rows = [
        row
        for row in profile_rows
        if str(row.get("name") or "") in {
            "tensor.read",
            "tensor.stacked_read",
            "tensor.stacked_bank_read",
            "tensor.mach_offset_read",
        }
    ]
    profile_read_mb = sum(_float(row.get("bytes_mb")) for row in profile_read_rows)
    profile_read_mb_per_generated_token = (
        profile_read_mb / generation_tokens if generation_tokens > 0 else 0.0
    )
    latest_file_pressure = (
        file_read_pressures[-1] if file_read_pressures else {}
    )
    latest_effective_pressure = (
        effective_read_pressures[-1] if effective_read_pressures else {}
    )
    if latest_file_pressure:
        profile_read_mb = max(
            profile_read_mb,
            _float(latest_file_pressure.get("read_mb"), profile_read_mb),
        )
        profile_read_mb_per_generated_token = max(
            profile_read_mb_per_generated_token,
            _float(
                latest_file_pressure.get("read_mb_per_generated_token"),
                profile_read_mb_per_generated_token,
            ),
        )
    effective_read_mb = profile_read_mb
    effective_read_mb_per_generated_token = profile_read_mb_per_generated_token
    if latest_effective_pressure:
        effective_read_mb = _float(
            latest_effective_pressure.get("effective_read_mb"),
            effective_read_mb,
        )
        effective_read_mb_per_generated_token = _float(
            latest_effective_pressure.get("effective_read_mb_per_generated_token"),
            effective_read_mb_per_generated_token,
        )
    latest_active_trace = active_expert_traces[-1] if active_expert_traces else {}
    latest_slice_residency = (
        active_slice_residencies[-1] if active_slice_residencies else {}
    )
    trace_layers = (
        latest_active_trace.get("layers")
        if isinstance(latest_active_trace.get("layers"), list)
        else []
    )
    top_trace_layers = [
        layer
        for layer in trace_layers[: min(3, len(trace_layers))]
        if isinstance(layer, dict)
    ]

    peak = memory.get("peak") if isinstance(memory.get("peak"), dict) else {}
    post_decode = (
        memory.get("post_decode") if isinstance(memory.get("post_decode"), dict) else {}
    )
    peak_gate = gates.get("peak") if isinstance(gates.get("peak"), dict) else {}
    post_decode_pressure = (
        pressures.get("post_decode")
        if isinstance(pressures.get("post_decode"), dict)
        else {}
    )
    pressure_available = bool(post_decode_pressure)

    prompt_tps = prompt_tokens / prompt_seconds if prompt_seconds > 0 else 0.0
    decode_tps = (
        generation_tokens / generation_seconds if generation_seconds > 0 else 0.0
    )
    top_rows = []
    for row in profile_rows[:top]:
        top_rows.append(
            {
                "name": row.get("name") or "",
                "milliseconds": _float(row.get("milliseconds")),
                "count": int(row.get("count") or 0),
                "bytes_mb": _float(row.get("bytes_mb")),
                "bandwidth_mb_per_second": _float(
                    row.get("bandwidth_mb_per_second")
                ),
            }
        )

    return {
        "path": str(path),
        "model_name": run_start.get("model_name") or "",
        "model_bytes": int(run_start.get("model_bytes") or 0),
        "turn_count": len(turns),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_tokens_per_second": prompt_tps,
        "decode_tokens_per_second": decode_tps,
        "coherent": coherent if turns else False,
        "peak_footprint_delta_percent_of_model": _float(
            peak.get("physical_footprint_delta_percent_of_model")
        ),
        "post_decode_footprint_delta_percent_of_model": _float(
            post_decode.get("physical_footprint_delta_percent_of_model")
        ),
        "peak_mlx_active_mb": _optional_mb(peak.get("mlx_active_memory_bytes")),
        "peak_mlx_cache_mb": _optional_mb(peak.get("mlx_cache_memory_bytes")),
        "peak_mlx_peak_mb": _optional_mb(peak.get("mlx_peak_memory_bytes")),
        "post_decode_mlx_active_mb": _optional_mb(
            post_decode.get("mlx_active_memory_bytes")
        ),
        "post_decode_mlx_cache_mb": _optional_mb(
            post_decode.get("mlx_cache_memory_bytes")
        ),
        "post_decode_mlx_peak_mb": _optional_mb(
            post_decode.get("mlx_peak_memory_bytes")
        ),
        "peak_activity_gate_passed": bool(peak_gate.get("passed")),
        "profile_read_mb": profile_read_mb,
        "profile_read_mb_per_generated_token": profile_read_mb_per_generated_token,
        "file_read_pressure_available": bool(latest_file_pressure),
        "file_read_pressure_passed": (
            latest_file_pressure.get("passed")
            if latest_file_pressure
            else None
        ),
        "effective_read_pressure_available": bool(latest_effective_pressure),
        "effective_read_pressure_passed": (
            latest_effective_pressure.get("passed")
            if latest_effective_pressure
            else None
        ),
        "effective_read_mb": effective_read_mb,
        "effective_read_mb_per_generated_token": effective_read_mb_per_generated_token,
        "active_expert_trace_available": bool(latest_active_trace),
        "active_expert_trace_calls": int(latest_active_trace.get("total_calls") or 0),
        "active_expert_trace_reuse_rate": _float(
            latest_active_trace.get("consecutive_reuse_rate")
        ),
        "active_expert_trace_top_layers": top_trace_layers,
        "active_slice_residency_available": bool(latest_slice_residency),
        "active_slice_residency_budget_mb": _optional_mb(
            latest_slice_residency.get("budget_bytes")
        ),
        "active_slice_residency_resident_mb": _optional_mb(
            latest_slice_residency.get("resident_bytes")
        ),
        "active_slice_residency_tensor_resident_mb": _optional_mb(
            latest_slice_residency.get("tensor_resident_bytes")
        ),
        "active_slice_residency_slice_resident_mb": _optional_mb(
            latest_slice_residency.get("slice_resident_bytes")
        ),
        "active_slice_residency_bank_resident_mb": _optional_mb(
            latest_slice_residency.get("bank_resident_bytes")
        ),
        "active_slice_residency_hit_rate": _float(
            latest_slice_residency.get("hit_rate")
        ),
        "active_slice_residency_byte_hit_rate": _float(
            latest_slice_residency.get("byte_hit_rate")
        ),
        "active_slice_residency_hits": int(latest_slice_residency.get("hits") or 0),
        "active_slice_residency_tensor_hits": int(
            latest_slice_residency.get("tensor_hits") or 0
        ),
        "active_slice_residency_slice_hits": int(
            latest_slice_residency.get("slice_hits") or 0
        ),
        "active_slice_residency_bank_hits": int(
            latest_slice_residency.get("bank_hits") or 0
        ),
        "active_slice_residency_misses": int(latest_slice_residency.get("misses") or 0),
        "active_slice_residency_tensor_misses": int(
            latest_slice_residency.get("tensor_misses") or 0
        ),
        "active_slice_residency_slice_misses": int(
            latest_slice_residency.get("slice_misses") or 0
        ),
        "active_slice_residency_bank_misses": int(
            latest_slice_residency.get("bank_misses") or 0
        ),
        "active_slice_residency_evictions": int(
            latest_slice_residency.get("evictions") or 0
        ),
        "active_slice_residency_tensor_evictions": int(
            latest_slice_residency.get("tensor_evictions") or 0
        ),
        "active_slice_residency_slice_evictions": int(
            latest_slice_residency.get("slice_evictions") or 0
        ),
        "active_slice_residency_bank_evictions": int(
            latest_slice_residency.get("bank_evictions") or 0
        ),
        "system_pressure_available": pressure_available,
        "system_pagein_mb": _optional_mb(post_decode_pressure.get("pageins_delta_bytes")),
        "system_pageout_mb": _optional_mb(post_decode_pressure.get("pageouts_delta_bytes")),
        "system_swapin_mb": _optional_mb(post_decode_pressure.get("swapins_delta_bytes")),
        "system_swapout_mb": _optional_mb(post_decode_pressure.get("swapouts_delta_bytes")),
        "turns": turn_summaries,
        "top_profile_rows": top_rows,
    }


def _print_text(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    baseline = summaries[0]
    for index, summary in enumerate(summaries):
        decode_tps = summary["decode_tokens_per_second"]
        prompt_tps = summary["prompt_tokens_per_second"]
        peak_pct = summary["peak_footprint_delta_percent_of_model"]
        if summary["system_pressure_available"]:
            pressure = (
                f"pagein_mb={summary['system_pagein_mb']:.2f} "
                f"swapin_mb={summary['system_swapin_mb']:.2f} "
                f"swapout_mb={summary['system_swapout_mb']:.2f}"
            )
        else:
            pressure = "pagein_mb=n/a swapin_mb=n/a swapout_mb=n/a"
        if summary["file_read_pressure_passed"] is None:
            file_read_gate = "file_read_gate=n/a"
        else:
            file_read_gate = (
                f"file_read_gate={str(summary['file_read_pressure_passed']).lower()}"
            )
        if summary["effective_read_pressure_passed"] is None:
            effective_read_gate = "effective_read_gate=n/a"
        else:
            effective_read_gate = (
                f"effective_read_gate={str(summary['effective_read_pressure_passed']).lower()} "
                f"effective_read_mb_per_gen_token={summary['effective_read_mb_per_generated_token']:.2f}"
            )
        if summary["active_expert_trace_available"]:
            trace = (
                f"trace_calls={summary['active_expert_trace_calls']} "
                f"trace_reuse={summary['active_expert_trace_reuse_rate']:.3f}"
            )
        else:
            trace = "trace_calls=n/a trace_reuse=n/a"
        if summary["active_slice_residency_available"]:
            tensor_resident = summary["active_slice_residency_tensor_resident_mb"]
            if tensor_resident is None:
                tensor_resident = 0.0
            slice_resident = summary["active_slice_residency_slice_resident_mb"]
            if slice_resident is None:
                slice_resident = summary["active_slice_residency_resident_mb"] or 0.0
            bank_resident = summary["active_slice_residency_bank_resident_mb"]
            if bank_resident is None:
                bank_resident = 0.0
            slice_residency = (
                f"cache_hit_rate={summary['active_slice_residency_hit_rate']:.3f} "
                f"cache_byte_hit_rate={summary['active_slice_residency_byte_hit_rate']:.3f} "
                f"cache_resident_mb={summary['active_slice_residency_resident_mb']:.2f} "
                f"tensor_resident_mb={tensor_resident:.2f} "
                f"slice_resident_mb={slice_resident:.2f} "
                f"bank_resident_mb={bank_resident:.2f} "
                f"tensor_evictions={summary['active_slice_residency_tensor_evictions']} "
                f"slice_evictions={summary['active_slice_residency_slice_evictions']} "
                f"bank_evictions={summary['active_slice_residency_bank_evictions']}"
            )
        else:
            slice_residency = (
                "cache_hit_rate=n/a cache_byte_hit_rate=n/a "
                "cache_resident_mb=n/a tensor_resident_mb=n/a "
                "slice_resident_mb=n/a bank_resident_mb=n/a "
                "tensor_evictions=n/a slice_evictions=n/a bank_evictions=n/a"
            )
        if summary["peak_mlx_active_mb"] is None:
            mlx_memory = "mlx_active_mb=n/a mlx_cache_mb=n/a mlx_peak_mb=n/a"
        else:
            mlx_memory = (
                f"mlx_active_mb={summary['peak_mlx_active_mb']:.2f} "
                f"mlx_cache_mb={summary['peak_mlx_cache_mb']:.2f} "
                f"mlx_peak_mb={summary['peak_mlx_peak_mb']:.2f}"
            )
        delta = ""
        if index > 0:
            base_tps = baseline["decode_tokens_per_second"]
            base_peak = baseline["peak_footprint_delta_percent_of_model"]
            tps_delta = decode_tps - base_tps
            peak_delta = peak_pct - base_peak
            delta = f" decode_delta={tps_delta:+.3f} peak_delta={peak_delta:+.3f}pp"

        print(summary["path"])
        print(
            "  "
            f"model={summary['model_name']} turns={summary['turn_count']} "
            f"coherent={str(summary['coherent']).lower()} "
            f"prompt_tps={prompt_tps:.3f} decode_tps={decode_tps:.3f} "
            f"gen_tokens={summary['generation_tokens']} "
            f"profile_read_mb={summary['profile_read_mb']:.2f} "
            f"profile_read_mb_per_gen_token={summary['profile_read_mb_per_generated_token']:.2f} "
            f"{file_read_gate} "
            f"{effective_read_gate} "
            f"{trace} "
            f"{slice_residency} "
            f"{mlx_memory} "
            f"{pressure} "
            f"peak_pct={peak_pct:.3f} gate={str(summary['peak_activity_gate_passed']).lower()}"
            f"{delta}"
        )
        for turn in summary["turns"]:
            print(
                "  "
                f"turn={turn['turn']} decode_tps={turn['tokens_per_second']:.3f} "
                f"tokens={turn['generation_tokens']} coherent={str(turn['coherent']).lower()} "
                f"stop={turn['stop_reason']} preview={turn['visible_preview']!r}"
            )
        if summary["top_profile_rows"]:
            print("  top_profile:")
            for row in summary["top_profile_rows"]:
                print(
                    "    "
                    f"{row['name']} ms={row['milliseconds']:.1f} "
                    f"count={row['count']} bytes_mb={row['bytes_mb']:.2f} "
                    f"bw_mb_s={row['bandwidth_mb_per_second']:.2f}"
                )
        if summary["active_expert_trace_top_layers"]:
            print("  active_expert_trace:")
            for layer in summary["active_expert_trace_top_layers"]:
                top_experts = layer.get("top_experts")
                if isinstance(top_experts, list):
                    experts = ",".join(
                        f"{expert.get('expert')}:{expert.get('count')}"
                        for expert in top_experts[:5]
                        if isinstance(expert, dict)
                    )
                else:
                    experts = ""
                print(
                    "    "
                    f"L{layer.get('layer')} calls={layer.get('calls')} "
                    f"slots={layer.get('routed_slots')} "
                    f"unique={layer.get('unique_expert_touches')} "
                    f"reuse={layer.get('consecutive_reuse_touches')} "
                    f"top={experts}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path, help="metrics.jsonl path(s)")
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="Number of streaming profile rows to include. Default: 8.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    args = parser.parse_args()

    summaries = [_summarize(path, max(1, args.top)) for path in args.metrics]
    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        _print_text(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
