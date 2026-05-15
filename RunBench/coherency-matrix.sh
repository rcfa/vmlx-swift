#!/usr/bin/env bash
# coherency-matrix.sh — systematic per-family coherency check.
#
# Each row exercises a model family × original feature × {JP=off, JP=70}
# and records PASS/FAIL into a CSV under /tmp/coherency-results-<DATE>.csv.
#
# Run subsets via: ./coherency-matrix.sh <row-name>
#   row names: omni-jp, qwen36-jp, jang2l-disk, jang2l-prefix, minimax-fp32,
#              gemma4-swa, mistral35-vl, holo3-mxfp4, all
#
# Pre-flight kills are in `preflight()` — call once before the first row.

set -uo pipefail
cd "$(dirname "$0")/.."

DATE="$(date +%Y%m%d-%H%M%S)"
CSV="/tmp/coherency-results-${DATE}.csv"
echo "row,model,feature,jp_setting,exit_code,duration_sec,note" > "$CSV"

DRIVE="/Volumes/EricsLLMDrive"
JANGQ="$DRIVE/jangq-ai"
DEALIGN="$DRIVE/dealignai"

preflight() {
  echo "[preflight] killing zombie inference processes…" >&2
  pkill -f mlx_lm 2>/dev/null || true
  pkill -f ollama 2>/dev/null || true
  pkill -f lms 2>/dev/null || true
  pkill -f RunBench 2>/dev/null || true
  pkill -f xctest 2>/dev/null || true
  sleep 2
}

run_row() {
  local row="$1" model="$2" feature="$3" jp="$4"
  shift 4
  local start=$(date +%s)
  echo "──────── [$row] $model · $feature · JP=$jp ────────" >&2
  "$@" > "/tmp/coh-${row}-${jp}.log" 2>&1
  local rc=$?
  local dur=$(( $(date +%s) - start ))
  local note="see /tmp/coh-${row}-${jp}.log"
  echo "${row},${model},${feature},${jp},${rc},${dur},${note}" >> "$CSV"
  if [ $rc -eq 0 ]; then
    echo "[$row JP=$jp] PASS in ${dur}s" >&2
  else
    echo "[$row JP=$jp] FAIL rc=$rc in ${dur}s — tail:" >&2
    tail -20 "/tmp/coh-${row}-${jp}.log" >&2 || true
  fi
}

# ─────────────── ROW DEFINITIONS ───────────────

row_omni_jp() {
  local M="$JANGQ/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4"
  for jp in 0 70; do
    local jpflag=0 ; [ "$jp" = "70" ] && jpflag=1
    run_row "omni-jangtq4" "$(basename "$M")" "BENCH_OMNI hybrid-SSM 13/13" "$jp" \
      env BENCH_OMNI=1 BENCH_MODEL="$M" \
          BENCH_JANGPRESS="$jpflag" BENCH_JANGPRESS_PCT=70 \
          swift run -c release RunBench
  done
}

row_qwen36_jp() {
  local M="$JANGQ/Qwen3.6-35B-A3B-JANGTQ4"
  for jp in 0 70; do
    local jpflag=0 ; [ "$jp" = "70" ] && jpflag=1
    run_row "qwen36-jangtq4" "$(basename "$M")" "BENCH_STABILITY hybrid-SSM × SWA" "$jp" \
      env BENCH_STABILITY=1 BENCH_MODEL="$M" \
          BENCH_JANGPRESS="$jpflag" BENCH_JANGPRESS_PCT=70 \
          swift run -c release RunBench
  done
}

row_jang2l_disk() {
  local M="$JANGQ/DeepSeek-V4-Flash-JANG_2L"
  for jp in 0 70; do
    local jpflag=0 ; [ "$jp" = "70" ] && jpflag=1
    run_row "jang2l-disk-restore" "$(basename "$M")" "BENCH_BATCH_DISK_RESTORE L2" "$jp" \
      env BENCH_BATCH_DISK_RESTORE=1 BENCH_MODEL="$M" \
          BENCH_JANGPRESS="$jpflag" BENCH_JANGPRESS_PCT=70 \
          swift run -c release RunBench
  done
}

row_jang2l_prefix() {
  local M="$JANGQ/DeepSeek-V4-Flash-JANG_2L"
  for jp in 0 70; do
    local jpflag=0 ; [ "$jp" = "70" ] && jpflag=1
    run_row "jang2l-cache-hit" "$(basename "$M")" "BENCH_BATCH_CACHE_HIT prefix" "$jp" \
      env BENCH_BATCH_CACHE_HIT=1 BENCH_MODEL="$M" \
          BENCH_JANGPRESS="$jpflag" BENCH_JANGPRESS_PCT=70 \
          swift run -c release RunBench
  done
}

row_minimax_fp32() {
  local M="$JANGQ/MiniMax-M2.7-JANGTQ4"
  for jp in 0 70; do
    local jpflag=0 ; [ "$jp" = "70" ] && jpflag=1
    run_row "minimax-jangtq4" "$(basename "$M")" "BENCH_BATCH_CHAT multi-turn coherency" "$jp" \
      env BENCH_BATCH_CHAT=1 BENCH_MODEL="$M" BENCH_MAX_TOKENS=128 \
          BENCH_JANGPRESS="$jpflag" BENCH_JANGPRESS_PCT=70 \
          swift run -c release RunBench
  done
}

row_holo3_mxfp4() {
  local M="$JANGQ/Holo3-35B-A3B-mxfp4"
  for jp in 0 70; do
    local jpflag=0 ; [ "$jp" = "70" ] && jpflag=1
    run_row "holo3-mxfp4" "$(basename "$M")" "BENCH_BATCH_CHAT multi-turn" "$jp" \
      env BENCH_BATCH_CHAT=1 BENCH_MODEL="$M" \
          BENCH_JANGPRESS="$jpflag" BENCH_JANGPRESS_PCT=70 \
          swift run -c release RunBench
  done
}

# ─────────────── DRIVER ───────────────

case "${1:-}" in
  omni-jp)         preflight; row_omni_jp ;;
  qwen36-jp)       preflight; row_qwen36_jp ;;
  jang2l-disk)     preflight; row_jang2l_disk ;;
  jang2l-prefix)   preflight; row_jang2l_prefix ;;
  minimax-fp32)    preflight; row_minimax_fp32 ;;
  holo3-mxfp4)     preflight; row_holo3_mxfp4 ;;
  all)
    preflight
    row_omni_jp
    row_qwen36_jp
    row_jang2l_disk
    row_jang2l_prefix
    row_minimax_fp32
    row_holo3_mxfp4
    ;;
  *)
    echo "usage: $0 <omni-jp|qwen36-jp|jang2l-disk|jang2l-prefix|minimax-fp32|holo3-mxfp4|all>" >&2
    echo "results CSV: /tmp/coherency-results-*.csv" >&2
    exit 1 ;;
esac

echo
echo "===== SUMMARY ====="
column -t -s, "$CSV"
echo
echo "CSV: $CSV"
