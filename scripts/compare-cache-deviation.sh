#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Usage:
  scripts/compare-cache-deviation.sh [options] <model-dir>

Options:
  --turn TEXT                 Add one validation turn. Repeat for multi-turn.
  --expect TEXT               Expected visible text for a turn. Repeat to match turns.
  --thinking on|off           Pass enable_thinking to the chat template. Default: off.
  --reasoning-effort VALUE    Pass the family reasoning_effort knob.
  --max-tokens N              Max generated tokens per turn. Default: 64.
  --prefill-step-size N       Prompt prefill chunk size. Default: 64.
  --min-visible-chars N       Require at least N visible chars per turn.
  --min-generation-tokens N   Require at least N generated tokens per turn.
  --fail-on-length-stop       Fail coherency if a turn stops at max tokens.
  --activity-gate PCT         Peak footprint gate for MLXPress-on runs. Default: 30.
  --activity-gate-report-only Record gate failures without killing the run.
  --kv-cache none|turboquant  MLXPress-on KV mode. Default: turboquant.
  --run-dir PATH              Output directory. Default: docs/local/deviation/<timestamp>.
  --skip-off                  Skip cache-off baseline when loading the full model is unsafe.
  --no-build                  Reuse existing .build/debug/mlxpress.

Runs:
  1. optional cache-off baseline
  2. MLXPress cache-stack cold run with an isolated disk cache directory
  3. MLXPress cache-stack warm run reusing that same isolated disk cache

Pass criteria:
  - each executed run exits 0
  - when cache-off baseline is present, normalized visible stdout matches cold
    and warm MLXPress stdout
  - when --skip-off is used, cold and warm stdout match
  - every run records prompt-tokens/s and decode tokens/s in stderr
  - every turn is visible, printable, expected-matching, and loop=pass
EOF
}

MAX_TOKENS=64
PREFILL_STEP_SIZE=64
THINKING=off
REASONING_EFFORT=""
ACTIVITY_GATE=30
ACTIVITY_GATE_REPORT_ONLY=0
KV_CACHE=turboquant
RUN_DIR=""
SKIP_OFF=0
BUILD=1
MIN_VISIBLE_CHARS=1
MIN_GENERATION_TOKENS=0
FAIL_ON_LENGTH_STOP=0
TURNS=()
EXPECTS=()
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --turn)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      TURNS+=("$2")
      shift 2
      ;;
    --expect)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      EXPECTS+=("$2")
      shift 2
      ;;
    --thinking)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      THINKING="$2"
      shift 2
      ;;
    --reasoning-effort)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      REASONING_EFFORT="$2"
      shift 2
      ;;
    --max-tokens)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      MAX_TOKENS="$2"
      shift 2
      ;;
    --prefill-step-size|--prefill-step)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      PREFILL_STEP_SIZE="$2"
      shift 2
      ;;
    --min-visible-chars)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      MIN_VISIBLE_CHARS="$2"
      shift 2
      ;;
    --min-generation-tokens)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      MIN_GENERATION_TOKENS="$2"
      shift 2
      ;;
    --fail-on-length-stop)
      FAIL_ON_LENGTH_STOP=1
      shift
      ;;
    --activity-gate)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      ACTIVITY_GATE="$2"
      shift 2
      ;;
    --activity-gate-report-only|--no-activity-gate-stop)
      ACTIVITY_GATE_REPORT_ONLY=1
      shift
      ;;
    --kv-cache)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      KV_CACHE="$2"
      shift 2
      ;;
    --run-dir)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      RUN_DIR="$2"
      shift 2
      ;;
    --skip-off)
      SKIP_OFF=1
      shift
      ;;
    --no-build)
      BUILD=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ne 1 ]]; then
  usage >&2
  exit 2
fi

MODEL_DIR="${POSITIONAL[0]}"
if [[ ${#TURNS[@]} -eq 0 ]]; then
  TURNS=("Reply with exactly one word: ready." "Reply with exactly one word: done.")
  EXPECTS=("ready" "done")
fi
if [[ ${#EXPECTS[@]} -gt 0 && ${#EXPECTS[@]} -ne ${#TURNS[@]} ]]; then
  echo "--expect count must match --turn count" >&2
  exit 2
fi

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="docs/local/deviation/$(date -u +"%Y%m%dT%H%M%SZ")"
fi
mkdir -p "$RUN_DIR"
CACHE_DIR="${RUN_DIR}/disk-cache"
mkdir -p "$CACHE_DIR"

if [[ "$BUILD" -eq 1 ]]; then
  if ! swift build --jobs "${MLXPRESS_SWIFT_BUILD_JOBS:-2}" --product mlxpress \
    >"${RUN_DIR}/build.out" 2>"${RUN_DIR}/build.err"
  then
    echo "build failed; see ${RUN_DIR}/build.err" >&2
    exit 1
  fi
fi

BIN=".build/debug/mlxpress"

COMMON_ARGS=(
  "$MODEL_DIR"
  --max-tokens "$MAX_TOKENS"
  --prefill-step-size "$PREFILL_STEP_SIZE"
  --thinking "$THINKING"
  --min-visible-chars "$MIN_VISIBLE_CHARS"
  --print-memory
)
if [[ "$MIN_GENERATION_TOKENS" -gt 0 ]]; then
  COMMON_ARGS+=(--min-generation-tokens "$MIN_GENERATION_TOKENS")
fi
if [[ "$FAIL_ON_LENGTH_STOP" -eq 1 ]]; then
  COMMON_ARGS+=(--fail-on-length-stop)
fi
if [[ -n "$REASONING_EFFORT" ]]; then
  COMMON_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
fi
for turn in "${TURNS[@]}"; do
  COMMON_ARGS+=(--turn "$turn")
done
for expect in "${EXPECTS[@]}"; do
  COMMON_ARGS+=(--expect "$expect")
done

run_case() {
  local name="$1"
  shift
  local out="${RUN_DIR}/${name}.out"
  local err="${RUN_DIR}/${name}.err"
  local -a args=("${COMMON_ARGS[@]}")
  args+=("$@")
  if "$BIN" "${args[@]}" >"$out" 2>"$err"; then
    printf "%s\tpass\n" "$name" >>"${RUN_DIR}/status.tsv"
    return 0
  fi
  local code=$?
  printf "%s\tfail:%s\n" "$name" "$code" >>"${RUN_DIR}/status.tsv"
  return "$code"
}

normalize_stdout() {
  sed -e 's/[[:space:]]*$//' -e '/^[[:space:]]*$/d' "$1"
}

avg_decode_tps() {
  awk '
    /Generation telemetry/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^tokens\/s=/) {
          value = $i
          sub(/^tokens\/s=/, "", value)
          if (value ~ /^[0-9]+(\.[0-9]+)?$/) { sum += value; n += 1 }
        }
      }
    }
    END { if (n > 0) printf "%.2f", sum / n; else printf "n/a" }
  ' "$1"
}

avg_prompt_tps() {
  awk '
    /Generation telemetry/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^prompt-tokens\/s=/) {
          value = $i
          sub(/^prompt-tokens\/s=/, "", value)
          if (value ~ /^[0-9]+(\.[0-9]+)?$/) { sum += value; n += 1 }
        }
      }
    }
    END { if (n > 0) printf "%.2f", sum / n; else printf "n/a" }
  ' "$1"
}

turn_coherency_ok() {
  local file="$1"
  awk '
    /Turn coherency/ {
      n += 1
      visible = "missing"
      printable = "missing"
      expected_status = "missing"
      loop = "missing"
      min_visible = "n/a"
      min_generated = "n/a"
      length_stop = "n/a"
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^visible=/) { visible = $i; sub(/^visible=/, "", visible) }
        if ($i ~ /^printable=/) { printable = $i; sub(/^printable=/, "", printable) }
        if ($i ~ /^expected=/) { expected_status = $i; sub(/^expected=/, "", expected_status) }
        if ($i ~ /^loop=/) { loop = $i; sub(/^loop=/, "", loop) }
        if ($i ~ /^min-visible=/) { min_visible = $i; sub(/^min-visible=/, "", min_visible) }
        if ($i ~ /^min-generated=/) { min_generated = $i; sub(/^min-generated=/, "", min_generated) }
        if ($i ~ /^length-stop=/) { length_stop = $i; sub(/^length-stop=/, "", length_stop) }
      }
      if (visible != "pass" ||
          printable != "pass" ||
          (expected_status != "pass" && expected_status != "n/a") ||
          loop != "pass" ||
          (min_visible != "pass" && min_visible != "n/a") ||
          (min_generated != "pass" && min_generated != "n/a") ||
          (length_stop != "pass" && length_stop != "n/a")) {
        failed = 1
      }
    }
    END {
      if (n == 0 || failed) print "fail"; else print "pass"
    }
  ' "$file"
}

: >"${RUN_DIR}/status.tsv"
RUN_FAILURE=0
REPORT_ONLY_ARGS=()
if [[ "$ACTIVITY_GATE_REPORT_ONLY" -eq 1 ]]; then
  REPORT_ONLY_ARGS+=(--activity-gate-report-only)
fi

if [[ "$SKIP_OFF" -eq 0 ]]; then
  run_case "cache_off" --mlxpress off --cache-stack off --disk-cache off || RUN_FAILURE=1
fi
run_case "mlxpress_cold" \
  --mlxpress auto \
  --router-advice \
  --cache-stack on \
  --disk-cache on \
  --disk-cache-dir "$CACHE_DIR" \
  --kv-cache "$KV_CACHE" \
  --activity-gate "$ACTIVITY_GATE" \
  "${REPORT_ONLY_ARGS[@]}" || RUN_FAILURE=1
run_case "mlxpress_warm" \
  --mlxpress auto \
  --router-advice \
  --cache-stack on \
  --disk-cache on \
  --disk-cache-dir "$CACHE_DIR" \
  --kv-cache "$KV_CACHE" \
  --activity-gate "$ACTIVITY_GATE" \
  "${REPORT_ONLY_ARGS[@]}" || RUN_FAILURE=1

normalize_stdout "${RUN_DIR}/mlxpress_cold.out" >"${RUN_DIR}/mlxpress_cold.normalized"
normalize_stdout "${RUN_DIR}/mlxpress_warm.out" >"${RUN_DIR}/mlxpress_warm.normalized"

BASELINE_LABEL="mlxpress_cold"
if [[ "$SKIP_OFF" -eq 0 ]]; then
  normalize_stdout "${RUN_DIR}/cache_off.out" >"${RUN_DIR}/cache_off.normalized"
  BASELINE_LABEL="cache_off"
fi

DEVIATION_STATUS=pass
if [[ "$RUN_FAILURE" -ne 0 ]]; then
  DEVIATION_STATUS=fail
fi
if ! cmp -s "${RUN_DIR}/${BASELINE_LABEL}.normalized" "${RUN_DIR}/mlxpress_cold.normalized"; then
  DEVIATION_STATUS=fail
fi
if ! cmp -s "${RUN_DIR}/${BASELINE_LABEL}.normalized" "${RUN_DIR}/mlxpress_warm.normalized"; then
  DEVIATION_STATUS=fail
fi
for name in cache_off mlxpress_cold mlxpress_warm; do
  [[ -f "${RUN_DIR}/${name}.err" ]] || continue
  if [[ "$(turn_coherency_ok "${RUN_DIR}/${name}.err")" != "pass" ]]; then
    DEVIATION_STATUS=fail
  fi
done

{
  printf "# MLXPress Cache Deviation Check\n\n"
  printf -- "- model: %s\n" "$MODEL_DIR"
  printf -- "- thinking: %s\n" "$THINKING"
  if [[ -n "$REASONING_EFFORT" ]]; then
    printf -- "- reasoning effort: %s\n" "$REASONING_EFFORT"
  fi
  printf -- "- kv cache: %s\n" "$KV_CACHE"
  printf -- "- max tokens: %s\n" "$MAX_TOKENS"
  printf -- "- min visible chars: %s\n" "$MIN_VISIBLE_CHARS"
  printf -- "- min generation tokens: %s\n" "$MIN_GENERATION_TOKENS"
  printf -- "- fail on length stop: %s\n" "$FAIL_ON_LENGTH_STOP"
  printf -- "- disk cache dir: %s\n" "$CACHE_DIR"
  printf -- "- deviation status: %s\n\n" "$DEVIATION_STATUS"
  printf "| Run | Prompt tok/s avg | Decode tok/s avg | Output |\n"
  printf "|---|---:|---:|---|\n"
  for name in cache_off mlxpress_cold mlxpress_warm; do
    [[ -f "${RUN_DIR}/${name}.err" ]] || continue
    printf "| %s | %s | %s | %s |\n" \
      "$name" \
      "$(avg_prompt_tps "${RUN_DIR}/${name}.err")" \
      "$(avg_decode_tps "${RUN_DIR}/${name}.err")" \
      "$(tr '\n' '/' <"${RUN_DIR}/${name}.normalized" | sed 's#/$##')"
  done
} >"${RUN_DIR}/RESULT.md"

echo "result: ${RUN_DIR}/RESULT.md" >&2
[[ "$DEVIATION_STATUS" == "pass" ]]
