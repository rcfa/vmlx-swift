#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Usage:
  scripts/validate-models.sh [--load] [--no-build] [--mlxpress off|auto|N] [--activity-gate PCT] [--no-activity-gate] [--gate-dense] [--max-tokens N] [--prefill-step-size N] [--thinking on|off] [--reasoning-effort VALUE] [--disk-cache-dir PATH] [--max-load-percent PCT] [--allow-over-ram-loads] [--allow-concurrent-model-runs] [--prompt TEXT] [--second-prompt TEXT] [--expect TEXT] [--second-expect TEXT] [--turn TEXT] [--single-turn] [--min-visible-chars N] [--min-generation-tokens N] [--fail-on-length-stop] <model-root-or-dir> [...]

Examples:
  scripts/validate-models.sh /path/to/models
  scripts/validate-models.sh --load --mlxpress auto --activity-gate 30 /path/to/models/JANGQ
  scripts/validate-models.sh --load --turn "Answer with one word: ready." --turn "Now answer with one word: done." /path/to/model

The script writes local logs to docs/local/model-validation/<timestamp>/, which
is intentionally ignored by git.

By default, --load runs a two-turn chat in one loaded session. This records
post-load RAM, post-decode RAM, peak RAM, peak/original ratio, per-turn
generation telemetry, and average tokens/sec. The Activity Monitor gate is
applied only to routed-MoE bundles unless --gate-dense is supplied.
Validation also defaults to --thinking off so reasoning-trained templates emit
final visible answers instead of reasoning-only streams.
Validation uses chunked prompt prefill by default to bound activation peaks;
raise --prefill-step-size only when intentionally testing the speed path.
Validation records per-turn no-loop status. Use --min-visible-chars,
--min-generation-tokens, and --fail-on-length-stop for longer no-loop rows
where a one-word answer is not enough proof.
For host safety, --load skips bundles larger than 90% of physical RAM unless
--allow-over-ram-loads or a higher --max-load-percent is supplied.
For measurement integrity, --load refuses to start a model row while another
mlxpress or RunBench model process is active unless
--allow-concurrent-model-runs is supplied.
EOF
}

LOAD=0
ACTIVITY_GATE=30
MAX_TOKENS=64
PREFILL_STEP_SIZE=64
PROMPT="Reply with exactly one word: ready."
SECOND_PROMPT="Reply with exactly one word: done."
EXPECT="ready"
SECOND_EXPECT="done"
MAX_DEPTH=5
MLXPRESS_POLICY="auto"
GATE_DENSE=0
ALLOW_RUNTIME_NOT_READY=0
MULTI_TURN=1
THINKING=off
REASONING_EFFORT=""
DISK_CACHE_DIR=""
MAX_LOAD_PERCENT=90
ALLOW_OVER_RAM_LOADS=0
ALLOW_CONCURRENT_MODEL_RUNS=0
MAX_LOAD_BYTES_OVERRIDE=""
MIN_VISIBLE_CHARS=1
MIN_GENERATION_TOKENS=0
FAIL_ON_LENGTH_STOP=0
NO_BUILD=0
CUSTOM_TURNS=()
CUSTOM_EXPECTS=()
ROOTS=()
SWIFT_BUILD_JOBS=${MLXPRESS_SWIFT_BUILD_JOBS:-2}
MIN_FREE_MEMORY_PERCENT=${MLXPRESS_MIN_FREE_MEMORY_PERCENT:-20}

lowercase() {
  printf "%s" "$1" | tr '[:upper:]' '[:lower:]'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --load)
      LOAD=1
      shift
      ;;
    --no-build)
      NO_BUILD=1
      shift
      ;;
    --activity-gate)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      ACTIVITY_GATE="$2"
      shift 2
      ;;
    --no-activity-gate)
      ACTIVITY_GATE=""
      shift
      ;;
    --gate-dense)
      GATE_DENSE=1
      shift
      ;;
    --allow-runtime-not-ready)
      ALLOW_RUNTIME_NOT_READY=1
      shift
      ;;
    --max-tokens)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      MAX_TOKENS="$2"
      shift 2
      ;;
    --prefill-step-size|--prefill-step)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      PREFILL_STEP_SIZE="$2"
      shift 2
      ;;
    --prompt)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      PROMPT="$2"
      shift 2
      ;;
    --second-prompt)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      SECOND_PROMPT="$2"
      shift 2
      ;;
    --expect)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      EXPECT="$2"
      CUSTOM_EXPECTS+=("$2")
      shift 2
      ;;
    --second-expect)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      SECOND_EXPECT="$2"
      shift 2
      ;;
    --turn)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      CUSTOM_TURNS+=("$2")
      shift 2
      ;;
    --single-turn)
      MULTI_TURN=0
      shift
      ;;
    --multi-turn)
      MULTI_TURN=1
      shift
      ;;
    --thinking)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      case "$(lowercase "$2")" in
        on|true|yes|1)
          THINKING=on
          ;;
        off|false|no|0)
          THINKING=off
          ;;
        *)
          echo "bad --thinking value: $2" >&2
          usage
          exit 2
          ;;
      esac
      shift 2
      ;;
    --reasoning-effort)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      REASONING_EFFORT="$2"
      shift 2
      ;;
    --min-visible-chars)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      MIN_VISIBLE_CHARS="$2"
      shift 2
      ;;
    --min-generation-tokens)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      MIN_GENERATION_TOKENS="$2"
      shift 2
      ;;
    --fail-on-length-stop)
      FAIL_ON_LENGTH_STOP=1
      shift
      ;;
    --disk-cache-dir)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      DISK_CACHE_DIR="$2"
      shift 2
      ;;
    --max-load-percent)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      MAX_LOAD_PERCENT="$2"
      shift 2
      ;;
    --max-load-gb)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      MAX_LOAD_BYTES_OVERRIDE=$(awk -v gb="$2" 'BEGIN { printf "%.0f", gb * 1024 * 1024 * 1024 }')
      shift 2
      ;;
    --allow-over-ram-loads|--no-load-safety-cap)
      ALLOW_OVER_RAM_LOADS=1
      shift
      ;;
    --allow-concurrent-model-runs)
      ALLOW_CONCURRENT_MODEL_RUNS=1
      shift
      ;;
    --mlxpress)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      MLXPRESS_POLICY="$2"
      shift 2
      ;;
    --max-depth)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      MAX_DEPTH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        ROOTS+=("$1")
        shift
      done
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      ROOTS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ROOTS[@]} -eq 0 ]]; then
  usage
  exit 2
fi

if [[ ${#CUSTOM_TURNS[@]} -gt 0 ]]; then
  RUN_TURNS=("${CUSTOM_TURNS[@]}")
  RUN_EXPECTS=()
  if [[ ${#CUSTOM_EXPECTS[@]} -gt 0 ]]; then
    RUN_EXPECTS=("${CUSTOM_EXPECTS[@]}")
  fi
elif [[ "$MULTI_TURN" -eq 1 ]]; then
  RUN_TURNS=("$PROMPT" "$SECOND_PROMPT")
  RUN_EXPECTS=("$EXPECT" "$SECOND_EXPECT")
else
  RUN_TURNS=("$PROMPT")
  RUN_EXPECTS=("$EXPECT")
fi

RUN_ID=$(date -u +"%Y%m%dT%H%M%SZ")
RUN_DIR=${MLXPRESS_VALIDATION_DIR:-"docs/local/model-validation/${RUN_ID}"}
SUMMARY_TSV="${RUN_DIR}/summary.tsv"
SUMMARY_MD="${RUN_DIR}/SUMMARY.md"
mkdir -p "$RUN_DIR"

BIN=".build/debug/mlxpress"
if [[ "$NO_BUILD" -eq 0 ]]; then
  if ! swift build --jobs "$SWIFT_BUILD_JOBS" >"${RUN_DIR}/build.out" 2>"${RUN_DIR}/build.err"; then
    echo "swift build failed; see ${RUN_DIR}/build.err" >&2
    exit 1
  fi
else
  printf "skipped by --no-build\n" >"${RUN_DIR}/build.out"
  : >"${RUN_DIR}/build.err"
  if [[ ! -x "$BIN" ]]; then
    echo "--no-build requested but ${BIN} is missing or not executable" >&2
    exit 1
  fi
fi
if ! scripts/prepare-mlx-metal.sh >"${RUN_DIR}/metal.out" 2>"${RUN_DIR}/metal.err"; then
  echo "metallib preparation failed; see ${RUN_DIR}/metal.err" >&2
  exit 1
fi
"$BIN" --runtime-check --json >"${RUN_DIR}/runtime.json" 2>"${RUN_DIR}/runtime.err" || true

json_get() {
  local file="$1"
  local key="$2"
  if command -v plutil >/dev/null 2>&1; then
    plutil -extract "$key" raw -o - "$file" 2>/dev/null && return 0
  fi
  printf "unknown"
}

safe_name() {
  printf "%s" "$1" | tr -c 'A-Za-z0-9._-' '_'
}

md_cell() {
  printf "%s" "$1" | sed 's/|/\\|/g'
}

format_bytes() {
  local bytes="$1"
  awk -v bytes="$bytes" '
    BEGIN {
      split("B KB MB GB TB PB", unit, " ")
      value = bytes + 0
      idx = 1
      while (value >= 1024 && idx < 6) {
        value /= 1024
        idx += 1
      }
      if (idx == 1) {
        printf "%d B", value
      } else {
        printf "%.2f %s", value, unit[idx]
      }
    }
  '
}

is_positive_integer() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

load_safety_cap_bytes() {
  local physical_bytes="$1"
  if [[ -n "$MAX_LOAD_BYTES_OVERRIDE" ]]; then
    printf "%s" "$MAX_LOAD_BYTES_OVERRIDE"
    return
  fi
  awk -v physical="$physical_bytes" -v percent="$MAX_LOAD_PERCENT" \
    'BEGIN { printf "%.0f", physical * percent / 100 }'
}

concurrent_model_runs() {
  ps -axo pid=,command= | awk -v self="$$" '
    $1 == self { next }
    index($0, "/RunBench") ||
    index($0, ".build/debug/mlxpress") ||
    index($0, ".build/release/mlxpress") {
      print
    }
  '
}

available_memory_percent() {
  if ! command -v memory_pressure >/dev/null 2>&1; then
    return 0
  fi
  memory_pressure 2>/dev/null | awk '
    /System-wide memory free percentage:/ {
      value = $NF
      gsub(/%/, "", value)
      print value
      exit
    }
  '
}

guard_available_memory() {
  local free_pct
  free_pct=$(available_memory_percent)
  if [[ -z "$free_pct" ]]; then
    return 0
  fi
  if awk -v free="$free_pct" -v min="$MIN_FREE_MEMORY_PERCENT" \
    'BEGIN { exit !(free < min) }'; then
    echo "system free memory is ${free_pct}%, below ${MIN_FREE_MEMORY_PERCENT}% guard; refusing model load" >&2
    return 4
  fi
}

guard_concurrent_model_runs() {
  if [[ "$ALLOW_CONCURRENT_MODEL_RUNS" -eq 1 ]]; then
    return 0
  fi
  local active
  active=$(concurrent_model_runs)
  if [[ -n "$active" ]]; then
    echo "concurrent model process detected; refusing to mix validation measurements" >&2
    echo "$active" >&2
    echo "wait for it to finish, or pass --allow-concurrent-model-runs for intentional diagnostics" >&2
    return 3
  fi
}

extract_labeled_field() {
  local label="$1"
  local field="$2"
  local file="$3"
  awk -v label="$label" -v field="$field" '
    index($0, label) {
      prefix = field "="
      for (i = 1; i <= NF; i++) {
        if (index($i, prefix) == 1) {
          value = substr($i, length(prefix) + 1)
          if (i < NF && $(i + 1) ~ /^(B|KB|MB|GB|TB|PB)$/) {
            value = value " " $(i + 1)
          }
          out = value
        }
      }
    }
    END {
      if (out != "") {
        print out
      } else {
        print "n/a"
      }
    }
  ' "$file"
}

average_tokens_per_second() {
  local file="$1"
  awk '
    /Generation telemetry/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^tokens\/s=/) {
          value = $i
          sub(/^tokens\/s=/, "", value)
          if (value ~ /^[0-9]+(\.[0-9]+)?$/) {
            sum += value
            n += 1
          }
        }
      }
    }
    END {
      if (n > 0) {
        printf "%.2f", sum / n
      } else {
        printf "n/a"
      }
    }
  ' "$file"
}

generation_turn_count() {
  local file="$1"
  awk '/Generation telemetry/ { n += 1 } END { printf "%d", n }' "$file"
}

turn_coherency_status() {
  local file="$1"
  local expected="$2"
  awk -v expected="$expected" '
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
        if ($i ~ /^visible=/) {
          visible = $i
          sub(/^visible=/, "", visible)
        }
        if ($i ~ /^printable=/) {
          printable = $i
          sub(/^printable=/, "", printable)
        }
        if ($i ~ /^expected=/) {
          expected_status = $i
          sub(/^expected=/, "", expected_status)
        }
        if ($i ~ /^loop=/) {
          loop = $i
          sub(/^loop=/, "", loop)
        }
        if ($i ~ /^min-visible=/) {
          min_visible = $i
          sub(/^min-visible=/, "", min_visible)
        }
        if ($i ~ /^min-generated=/) {
          min_generated = $i
          sub(/^min-generated=/, "", min_generated)
        }
        if ($i ~ /^length-stop=/) {
          length_stop = $i
          sub(/^length-stop=/, "", length_stop)
        }
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
      if (n == 0) {
        print "n/a"
      } else if (n != expected || failed) {
        print "fail"
      } else {
        print "pass"
      }
    }
  ' "$file"
}

turn_output_preview() {
  local file="$1"
  awk '
    /Turn coherency/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^preview=/) {
          value = $i
          sub(/^preview=/, "", value)
          if (value != "") {
            if (out != "") {
              out = out " / " value
            } else {
              out = value
            }
          }
        }
      }
    }
    END {
      if (out != "") {
        print out
      } else {
        print "n/a"
      }
    }
  ' "$file"
}

runtime_ready=$(json_get "${RUN_DIR}/runtime.json" "activityCompressionReady")
if [[ "$LOAD" -eq 1 && -n "$ACTIVITY_GATE" && "$runtime_ready" != "true" && "$ALLOW_RUNTIME_NOT_READY" -eq 0 ]]; then
  echo "runtime is not Activity compression ready; see ${RUN_DIR}/runtime.json" >&2
  echo "use --allow-runtime-not-ready only for loader debugging without release-gate meaning" >&2
  exit 1
fi

header="name	format	model_type	routed	safetensors_bytes	model_size	inspect	load	gate	multi_turn	coherency	output_preview	post_load_ram	post_decode_ram	peak_ram	peak_vs_model	tokens_s_avg	turns	logs"
printf "%s\n" "$header" | tee "$SUMMARY_TSV"
{
  printf "# MLXPress Validation %s\n\n" "$RUN_ID"
  printf "%s\n" "- mlxpress policy: ${MLXPRESS_POLICY}"
  printf "%s\n" "- activity gate: ${ACTIVITY_GATE:-disabled}"
  printf "%s\n" "- max tokens per turn: ${MAX_TOKENS}"
  printf "%s\n" "- prefill step size: ${PREFILL_STEP_SIZE}"
  printf "%s\n" "- thinking: ${THINKING}"
  if [[ -n "$REASONING_EFFORT" ]]; then
    printf "%s\n" "- reasoning effort: ${REASONING_EFFORT}"
  fi
  if [[ -n "$DISK_CACHE_DIR" ]]; then
    printf "%s\n" "- disk cache dir: ${DISK_CACHE_DIR}"
  fi
  if [[ "$ALLOW_OVER_RAM_LOADS" -eq 1 ]]; then
    printf "%s\n" "- load safety cap: disabled"
  elif [[ -n "$MAX_LOAD_BYTES_OVERRIDE" ]]; then
    printf "%s\n" "- load safety cap: $(format_bytes "$MAX_LOAD_BYTES_OVERRIDE")"
  else
    printf "%s\n" "- load safety cap: ${MAX_LOAD_PERCENT}% of physical RAM"
  fi
  if [[ "$ALLOW_CONCURRENT_MODEL_RUNS" -eq 1 ]]; then
    printf "%s\n" "- concurrent model guard: disabled"
  else
    printf "%s\n" "- concurrent model guard: enabled"
  fi
  printf "%s\n" "- minimum free memory before each load: ${MIN_FREE_MEMORY_PERCENT}%"
  printf "%s\n\n" "- configured turns: ${#RUN_TURNS[@]}"
  printf "| Model | Format | Type | Routed | Original | Inspect | Load | Gate | Multi-turn | Coherency | Output preview | Post-load RAM | Post-decode RAM | Peak RAM | Peak/original | Tokens/s avg | Turns |\n"
  printf "|---|---|---|---:|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|\n"
} >"$SUMMARY_MD"

for root in "${ROOTS[@]}"; do
  if [[ ! -e "$root" ]]; then
    echo "missing root: $root" >&2
    continue
  fi

  while IFS= read -r -d '' config; do
    model_dir=$(dirname "$config")
    name=$(basename "$model_dir")
    safe=$(safe_name "$name")
    inspect_json="${RUN_DIR}/${safe}.inspect.json"
    inspect_err="${RUN_DIR}/${safe}.inspect.err"
    load_out="${RUN_DIR}/${safe}.load.out"
    load_err="${RUN_DIR}/${safe}.load.err"

    inspect_status="pass"
    if ! "$BIN" "$model_dir" --inspect --json --print-memory >"$inspect_json" 2>"$inspect_err"; then
      inspect_status="fail"
    fi

    format=$(json_get "$inspect_json" "bundle.format")
    model_type=$(json_get "$inspect_json" "bundle.modelType")
    routed=$(json_get "$inspect_json" "bundle.isRouted")
    safetensors_bytes=$(json_get "$inspect_json" "bundle.totalSafetensorsBytes")
    physical_bytes=$(json_get "$inspect_json" "bundle.physicalMemoryBytes")

    model_size="n/a"
    if is_positive_integer "$safetensors_bytes"; then
      model_size=$(format_bytes "$safetensors_bytes")
    fi
    load_status="skip"
    gate_status="skip"
    should_load=0
    post_load_ram="n/a"
    post_decode_ram="n/a"
    peak_ram="n/a"
    peak_vs_model="n/a"
    tokens_s_avg="n/a"
    turns_done=0
    multi_turn_status="skip"
    coherency_status="skip"
    output_preview="n/a"

    if [[ "$LOAD" -eq 1 ]]; then
      if [[ "$inspect_status" != "pass" ]]; then
        load_status="skip:inspect-failed"
        gate_status="skip:inspect-failed"
        multi_turn_status="skip"
        coherency_status="skip"
        output_preview="skipped-inspect-failed"
        printf "skip load: %s inspect failed; refusing to load without trusted bundle facts\n" \
          "$model_dir" >"$load_err"
        : >"$load_out"
      elif [[ "$ALLOW_OVER_RAM_LOADS" -eq 0 ]] \
        && is_positive_integer "$safetensors_bytes" \
        && is_positive_integer "$physical_bytes"; then
        safety_cap_bytes=$(load_safety_cap_bytes "$physical_bytes")
        if is_positive_integer "$safety_cap_bytes" && [[ "$safetensors_bytes" -gt "$safety_cap_bytes" ]]; then
          load_status="skip:load-cap"
          gate_status="skip:load-cap"
          multi_turn_status="skip"
          coherency_status="skip"
          output_preview="skipped-over-load-cap"
          printf "skip load: %s safetensors=%s cap=%s physical=%s\n" \
            "$model_dir" \
            "$(format_bytes "$safetensors_bytes")" \
            "$(format_bytes "$safety_cap_bytes")" \
            "$(format_bytes "$physical_bytes")" >"$load_err"
          : >"$load_out"
        else
          should_load=1
        fi
      elif ! is_positive_integer "$safetensors_bytes"; then
        load_status="skip:no-local-safetensors"
        gate_status="skip:no-local-safetensors"
        multi_turn_status="skip"
        coherency_status="skip"
        output_preview="skipped-no-local-safetensors"
        printf "skip load: %s has no top-level safetensors bytes; activity gate cannot protect this row\n" \
          "$model_dir" >"$load_err"
        : >"$load_out"
      else
        should_load=1
      fi
    fi

    if [[ "${should_load:-0}" -eq 1 ]]; then
      guard_available_memory >"${RUN_DIR}/${safe}.memory.out" 2>"${RUN_DIR}/${safe}.memory.err"
      code=$?
      if [[ "$code" -ne 0 ]]; then
        load_status="blocked:memory:${code}"
        gate_status="blocked:memory"
        multi_turn_status="blocked"
        coherency_status="blocked"
        output_preview="blocked-low-free-memory"
        should_load=0
      fi
    fi

    if [[ "${should_load:-0}" -eq 1 ]]; then
      guard_concurrent_model_runs >"${RUN_DIR}/${safe}.concurrency.out" 2>"${RUN_DIR}/${safe}.concurrency.err"
      code=$?
      if [[ "$code" -ne 0 ]]; then
        load_status="blocked:concurrent:${code}"
        gate_status="blocked:concurrent"
        multi_turn_status="blocked"
        coherency_status="blocked"
        output_preview="blocked-concurrent-model-run"
        should_load=0
      fi
    fi

    if [[ "${should_load:-0}" -eq 1 ]]; then
      load_args=(
        "$model_dir"
        --max-tokens "$MAX_TOKENS"
        --prefill-step-size "$PREFILL_STEP_SIZE"
        --thinking "$THINKING"
        --mlxpress "$MLXPRESS_POLICY"
        --min-visible-chars "$MIN_VISIBLE_CHARS"
        --print-memory
      )
      if [[ "$MIN_GENERATION_TOKENS" -gt 0 ]]; then
        load_args+=(--min-generation-tokens "$MIN_GENERATION_TOKENS")
      fi
      if [[ "$FAIL_ON_LENGTH_STOP" -eq 1 ]]; then
        load_args+=(--fail-on-length-stop)
      fi
      if [[ -n "$REASONING_EFFORT" ]]; then
        load_args+=(--reasoning-effort "$REASONING_EFFORT")
      fi
      if [[ -n "$DISK_CACHE_DIR" ]]; then
        load_args+=(--disk-cache-dir "$DISK_CACHE_DIR")
      fi
      for turn in "${RUN_TURNS[@]}"; do
        load_args+=(--turn "$turn")
      done
      if [[ ${#RUN_EXPECTS[@]} -gt 0 ]]; then
        for expected in "${RUN_EXPECTS[@]}"; do
          load_args+=(--expect "$expected")
        done
      fi
      if [[ -n "$ACTIVITY_GATE" && ( "$routed" == "true" || "$GATE_DENSE" -eq 1 ) ]]; then
        load_args+=(--activity-gate "$ACTIVITY_GATE")
      else
        gate_status="n/a"
      fi

      if "$BIN" "${load_args[@]}" >"$load_out" 2>"$load_err"; then
        load_status="pass"
        if [[ "$gate_status" != "n/a" ]]; then
          gate_status="pass"
        fi
      else
        code=$?
        if [[ "$code" -eq 2 ]]; then
          load_status="pass"
          gate_status="fail"
        else
          load_status="fail:${code}"
          gate_status="unknown"
        fi
      fi

      post_load_ram=$(extract_labeled_field "Post-load memory delta" "footprint-increase" "$load_err")
      post_decode_ram=$(extract_labeled_field "Post-decode memory delta" "footprint-increase" "$load_err")
      peak_ram=$(extract_labeled_field "Peak memory delta" "footprint-increase" "$load_err")
      if [[ "$peak_ram" == "n/a" ]]; then
        peak_ram=$(extract_labeled_field "Peak Activity Monitor compression gate" "footprint-increase" "$load_err")
      fi
      peak_vs_model=$(extract_labeled_field "Peak memory delta" "ratio" "$load_err")
      if [[ "$peak_vs_model" == "n/a" ]]; then
        peak_vs_model=$(extract_labeled_field "Peak Activity Monitor compression gate" "ratio" "$load_err")
      fi
      measured_model_size=$(extract_labeled_field "Peak memory delta" "model-bytes" "$load_err")
      if [[ "$measured_model_size" == "n/a" ]]; then
        measured_model_size=$(extract_labeled_field "Peak Activity Monitor compression gate" "model-bytes" "$load_err")
      fi
      if [[ "$measured_model_size" == "n/a" ]]; then
        measured_model_size=$(extract_labeled_field "Post-load memory delta" "model-bytes" "$load_err")
      fi
      if [[ "$measured_model_size" != "n/a" ]]; then
        model_size="$measured_model_size"
      fi
      tokens_s_avg=$(average_tokens_per_second "$load_err")
      turns_done=$(generation_turn_count "$load_err")
      if [[ "$turns_done" -eq "${#RUN_TURNS[@]}" ]]; then
        multi_turn_status="pass"
      else
        multi_turn_status="fail"
      fi
      coherency_status=$(turn_coherency_status "$load_err" "${#RUN_TURNS[@]}")
      output_preview=$(turn_output_preview "$load_err")
    fi
    unset should_load

    row=$(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" \
      "$name" "$format" "$model_type" "$routed" "$safetensors_bytes" "$model_size" \
      "$inspect_status" "$load_status" "$gate_status" "$multi_turn_status" \
      "$coherency_status" "$output_preview" "$post_load_ram" "$post_decode_ram" "$peak_ram" \
      "$peak_vs_model" "$tokens_s_avg" "$turns_done" "$RUN_DIR")
    printf "%s\n" "$row" | tee -a "$SUMMARY_TSV"

    printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
      "$(md_cell "$name")" "$(md_cell "$format")" "$(md_cell "$model_type")" \
      "$routed" "$(md_cell "$model_size")" "$inspect_status" "$load_status" \
      "$gate_status" "$multi_turn_status" "$coherency_status" \
      "$(md_cell "$output_preview")" "$(md_cell "$post_load_ram")" "$(md_cell "$post_decode_ram")" \
      "$(md_cell "$peak_ram")" "$(md_cell "$peak_vs_model")" \
      "$(md_cell "$tokens_s_avg")" "$turns_done" >>"$SUMMARY_MD"
  done < <(find "$root" -maxdepth "$MAX_DEPTH" -name config.json -print0 2>/dev/null)
done

echo "summary: ${SUMMARY_MD}" >&2
