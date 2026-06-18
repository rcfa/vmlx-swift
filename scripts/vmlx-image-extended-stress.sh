#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

STAMP="${VMLX_IMAGE_EXTENDED_STAMP:-$(date -u +%Y-%m-%dT%H%M%SZ)-extended}"
DOC_PREFIX=""
if [[ -f "$ROOT/docs/OSAURUS_IMAGE_UI_MANIFEST.json" ]]; then
  DOC_PREFIX="docs/"
fi

ART_ROOT="${VMLX_IMAGE_EXTENDED_ARTIFACT_ROOT:-${DOC_PREFIX}local/vmlx-flux-probes/${STAMP}}"
OUT_ROOT="${VMLX_IMAGE_EXTENDED_OUTPUT_ROOT:-${DOC_PREFIX}local/vmlx-flux-outputs/${STAMP}}"
MODEL_ROOT="${VMLX_IMAGE_MODEL_ROOT:-}"
SEED="${VMLX_IMAGE_EXTENDED_SEED:-17}"
IDEOGRAM_SEED="${VMLX_IMAGE_EXTENDED_IDEOGRAM_SEED:-103437}"
Z_STEPS="${VMLX_IMAGE_EXTENDED_Z_STEPS:-8}"
FLUX_STEPS="${VMLX_IMAGE_EXTENDED_FLUX_STEPS:-4}"
QWEN_IMAGE_STEPS="${VMLX_IMAGE_EXTENDED_QWEN_IMAGE_STEPS:-20}"
QWEN_EDIT_STEPS="${VMLX_IMAGE_EXTENDED_QWEN_EDIT_STEPS:-20}"
QWEN_DIAG_STEPS="${VMLX_IMAGE_EXTENDED_QWEN_DIAG_STEPS:-4}"
IDEOGRAM_STEPS="${VMLX_IMAGE_EXTENDED_IDEOGRAM_STEPS:-20}"
SKIP_BUILD="${VMLX_IMAGE_EXTENDED_SKIP_BUILD:-0}"
MIN_OPEN_FILES="${VMLX_IMAGE_EXTENDED_MIN_OPEN_FILES:-4096}"

mkdir -p "$ART_ROOT" "$OUT_ROOT"

CURRENT_OPEN_FILES="$(ulimit -n)"
if [[ "$CURRENT_OPEN_FILES" != "unlimited" && "$CURRENT_OPEN_FILES" -lt "$MIN_OPEN_FILES" ]]; then
  ulimit -n "$MIN_OPEN_FILES" 2>/dev/null || true
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  swift build --product vmlxflux-probe
fi

resolve_probe() {
  if [[ -n "${VMLX_FLUX_PROBE:-}" ]]; then
    printf '%s\n' "$VMLX_FLUX_PROBE"
    return
  fi

  local candidate
  for candidate in \
    ".build/arm64-apple-macosx/debug/vmlxflux-probe" \
    ".build/debug/vmlxflux-probe"
  do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  printf 'vmlx-image-extended-stress: vmlxflux-probe executable not found after build\n' >&2
  return 1
}

PROBE="$(resolve_probe)"
TIME_BIN="/usr/bin/time"
if [[ ! -x "$TIME_BIN" ]]; then
  TIME_BIN=""
fi

run_probe() {
  local label="$1"
  shift
  local artifacts="${ART_ROOT}/${label}"
  local outputs="${OUT_ROOT}/${label}"
  mkdir -p "$artifacts" "$outputs"
  printf '\n== %s ==\n' "$label"

  local args=()
  if [[ -n "$MODEL_ROOT" ]]; then
    args+=(--root "$MODEL_ROOT")
  fi
  args+=("$@" --artifacts "$artifacts" --output-dir "$outputs")

  set +e
  if [[ -n "$TIME_BIN" ]]; then
    "$TIME_BIN" -l "$PROBE" "${args[@]}" >"$artifacts/stdout.log" 2>"$artifacts/stderr-time.log"
  else
    "$PROBE" "${args[@]}" >"$artifacts/stdout.log" 2>"$artifacts/stderr-time.log"
  fi
  local status=$?
  set -e

  printf '%s\n' "$status" >"$artifacts/exit-code.txt"
  if [[ "$status" != "0" ]]; then
    printf 'vmlx-image-extended-stress: %s exited %s; continuing to summary\n' "$label" "$status" >&2
  fi
}

time_metrics_json() {
  local label="$1"
  local time_log="${ART_ROOT}/${label}/stderr-time.log"
  local max_rss=""
  local elapsed=""
  if [[ -f "$time_log" ]]; then
    max_rss="$(awk '/maximum resident set size/ {print $1}' "$time_log" | tail -n 1)"
    elapsed="$(awk '/real/ {print $1}' "$time_log" | tail -n 1)"
  fi
  jq -n \
    --arg log "$time_log" \
    --arg max_rss "$max_rss" \
    --arg elapsed "$elapsed" \
    '{
      log: $log,
      max_rss_bytes: (if $max_rss == "" then null else ($max_rss | tonumber) end),
      elapsed_seconds: (if $elapsed == "" then null else ($elapsed | tonumber) end)
    }'
}

append_row() {
  local row="$1"
  ROWS_JSON="$(jq -c --argjson row "$row" '. + [$row]' <<< "$ROWS_JSON")"
}

summarize_matrix() {
  local label="$1"
  local matrix_path="${ART_ROOT}/${label}/compatibility-matrix.json"
  local exit_code
  exit_code="$(cat "${ART_ROOT}/${label}/exit-code.txt" 2>/dev/null || printf 'missing')"
  if [[ ! -f "$matrix_path" ]]; then
    append_row "$(
      jq -n \
        --arg label "$label" \
        --arg artifact "$matrix_path" \
        --arg exit_code "$exit_code" \
        --argjson metrics "$(time_metrics_json "$label")" \
        '{
          label: $label,
          kind: "matrix_load",
          artifact: $artifact,
          exit_code: $exit_code,
          model_count: 0,
          loaded_count: 0,
          unloaded: [],
          time: $metrics,
          error: "missing matrix artifact",
          status: "failed"
        }'
    )"
    return
  fi
  local row
  row="$(
    jq -n \
      --arg label "$label" \
      --arg artifact "$matrix_path" \
      --arg exit_code "$exit_code" \
      --argjson metrics "$(time_metrics_json "$label")" \
      --slurpfile matrix "$matrix_path" \
      '($matrix[0] // {}) as $m
      | ([($m.rows // [])[] | select(.load_status == "loaded")] | length) as $loaded
      | ($m.model_count // 0) as $count
      | {
          label: $label,
          kind: "matrix_load",
          artifact: $artifact,
          exit_code: $exit_code,
          model_count: $count,
          loaded_count: $loaded,
          unloaded: [($m.rows // [])[] | select(.load_status != "loaded") | .directory_name],
          time: $metrics,
          status: (if $exit_code == "0" and $count > 0 and $loaded == $count then "passed" else "failed" end)
        }'
  )"
  append_row "$row"
}

summarize_turns() {
  local label="$1"
  local file="$2"
  local key="$3"
  local minimum="$4"
  local repeat_index="$5"
  local sensitive_index="$6"
  local path="${ART_ROOT}/${label}/${file}"
  local exit_code
  exit_code="$(cat "${ART_ROOT}/${label}/exit-code.txt" 2>/dev/null || printf 'missing')"
  if [[ ! -f "$path" ]]; then
    append_row "$(
      jq -n \
        --arg label "$label" \
        --arg artifact "$path" \
        --arg key "$key" \
        --arg exit_code "$exit_code" \
        --argjson metrics "$(time_metrics_json "$label")" \
        '{
          label: $label,
          kind: $key,
          artifact: $artifact,
          exit_code: $exit_code,
          load_status: "missing",
          statuses: [],
          shas: [],
          outputs: [],
          deterministic_repeat: false,
          prompt_sensitive: false,
          time: $metrics,
          error: "missing load artifact",
          status: "failed"
        }'
    )"
    return
  fi
  local row
  row="$(
    jq -n \
      --arg label "$label" \
      --arg artifact "$path" \
      --arg key "$key" \
      --arg exit_code "$exit_code" \
      --argjson minimum "$minimum" \
      --argjson repeat_index "$repeat_index" \
      --argjson sensitive_index "$sensitive_index" \
      --argjson metrics "$(time_metrics_json "$label")" \
      --slurpfile payload "$path" \
      '($payload[0] // {}) as $p
      | ($p[$key] // []) as $turns
      | ($turns | map(.status)) as $statuses
      | ($turns | map(.image_diagnostics.sha256 // null)) as $shas
      | ($turns | map(.image_diagnostics.path // .output // null)) as $outputs
      | ($statuses | length >= $minimum and all(. == "completed")) as $completed
      | (if $repeat_index < 0 then true else ($shas[0] != null and $shas[$repeat_index] != null and $shas[0] == $shas[$repeat_index]) end) as $repeat_ok
      | (if $sensitive_index < 0 then true else ($shas[0] != null and $shas[$sensitive_index] != null and $shas[0] != $shas[$sensitive_index]) end) as $sensitive_ok
      | {
          label: $label,
          kind: $key,
          artifact: $artifact,
          exit_code: $exit_code,
          load_status: ($p.load_status // "unknown"),
          statuses: $statuses,
          shas: $shas,
          outputs: $outputs,
          deterministic_repeat: $repeat_ok,
          prompt_sensitive: $sensitive_ok,
          time: $metrics,
          status: (if $exit_code == "0" and ($p.load_status // "") == "loaded" and $completed and $repeat_ok and $sensitive_ok then "passed" else "failed" end)
        }'
  )"
  append_row "$row"
}

summarize_qwen_diagnostics() {
  local label="$1"
  local file="$2"
  local path="${ART_ROOT}/${label}/${file}"
  local exit_code
  exit_code="$(cat "${ART_ROOT}/${label}/exit-code.txt" 2>/dev/null || printf 'missing')"
  if [[ ! -f "$path" ]]; then
    append_row "$(
      jq -n \
        --arg label "$label" \
        --arg artifact "$path" \
        --arg exit_code "$exit_code" \
        --argjson metrics "$(time_metrics_json "$label")" \
        '{
          label: $label,
          kind: "qwen_edit_diagnostics",
          artifact: $artifact,
          exit_code: $exit_code,
          load_status: "missing",
          prompt_ok: false,
          conditioning_ok: false,
          vision_ok: false,
          denoise_ok: false,
          prompt_records: 0,
          vision_records: 0,
          denoise_records: 0,
          time: $metrics,
          error: "missing load artifact",
          status: "failed"
        }'
    )"
    return
  fi
  local row
  row="$(
    jq -n \
      --arg label "$label" \
      --arg artifact "$path" \
      --arg exit_code "$exit_code" \
      --argjson metrics "$(time_metrics_json "$label")" \
      --slurpfile payload "$path" \
      '($payload[0] // {}) as $p
      | ($p.qwen_edit_prompt_tokens // []) as $prompt
      | ($p.qwen_edit_conditioning // {}) as $conditioning
      | ($p.qwen_edit_vision_language // []) as $vision
      | ($p.qwen_edit_denoise // []) as $denoise
      | ($prompt | length > 0 and all(.status == "tokenized")) as $prompt_ok
      | (($conditioning.status // "") == "encoded" and ($conditioning.image_count // 0) >= 1) as $conditioning_ok
      | ($vision | length > 0 and all(.status == "encoded" and .matches_features == true)) as $vision_ok
      | ($denoise | length > 0 and all(.status == "predicted")) as $denoise_ok
      | {
          label: $label,
          kind: "qwen_edit_diagnostics",
          artifact: $artifact,
          exit_code: $exit_code,
          load_status: ($p.load_status // "unknown"),
          prompt_ok: $prompt_ok,
          conditioning_ok: $conditioning_ok,
          vision_ok: $vision_ok,
          denoise_ok: $denoise_ok,
          prompt_records: ($prompt | length),
          vision_records: ($vision | length),
          denoise_records: ($denoise | length),
          time: $metrics,
          status: (if $exit_code == "0" and ($p.load_status // "") == "loaded" and $prompt_ok and $conditioning_ok and $vision_ok and $denoise_ok then "passed" else "failed" end)
        }'
  )"
  append_row "$row"
}

summarize_mask_rejection() {
  local label="$1"
  local file="$2"
  local path="${ART_ROOT}/${label}/${file}"
  local exit_code
  exit_code="$(cat "${ART_ROOT}/${label}/exit-code.txt" 2>/dev/null || printf 'missing')"
  if [[ ! -f "$path" ]]; then
    append_row "$(
      jq -n \
        --arg label "$label" \
        --arg artifact "$path" \
        --arg exit_code "$exit_code" \
        --argjson metrics "$(time_metrics_json "$label")" \
        '{
          label: $label,
          kind: "qwen_edit_mask_rejection",
          artifact: $artifact,
          exit_code: $exit_code,
          load_status: "missing",
          statuses: [],
          messages: [],
          rejected: false,
          mentions_mask: false,
          time: $metrics,
          error: "missing load artifact",
          status: "failed"
        }'
    )"
    return
  fi
  local row
  row="$(
    jq -n \
      --arg label "$label" \
      --arg artifact "$path" \
      --arg exit_code "$exit_code" \
      --argjson metrics "$(time_metrics_json "$label")" \
      --slurpfile payload "$path" \
      '($payload[0] // {}) as $p
      | ($p.edit_turns // []) as $turns
      | ($turns | map(.status)) as $statuses
      | ($turns | map(.message // .error // "")) as $messages
      | ($turns | length > 0 and all(.status != "completed")) as $rejected
      | (($messages | join(" ") | ascii_downcase) | contains("mask")) as $mentions_mask
      | {
          label: $label,
          kind: "qwen_edit_mask_rejection",
          artifact: $artifact,
          exit_code: $exit_code,
          load_status: ($p.load_status // "unknown"),
          statuses: $statuses,
          messages: $messages,
          rejected: $rejected,
          mentions_mask: $mentions_mask,
          time: $metrics,
          status: (if $exit_code == "0" and ($p.load_status // "") == "loaded" and $rejected and $mentions_mask then "passed" else "failed" end)
        }'
  )"
  append_row "$row"
}

APPLE_PROMPT="a red apple on a plain white background, centered, clean product photo"
MOUNTAIN_PROMPT="a blue mountain landscape under a golden sun, watercolor"
WIDE_PROMPT="a wide cinematic product photo of a red apple beside a blue ceramic cup on a white table"
TALL_PROMPT="a tall portrait poster of a blue mountain under a yellow sun with clean white margins"
EDIT_BLUE_PROMPT="turn the apple blue while keeping the object centered and the background white"
EDIT_PEAR_PROMPT="turn the apple into a green pear on a plain white background"
EDIT_MULTI_PROMPT="combine the fruit from the first image with the color palette from the second image on a white studio background"
IDEOGRAM_WIDE_PROMPT='{"high_level_description":"A clean wide product photograph of a green glass apple beside a blue ceramic cup on a white table.","style_description":{"aesthetics":"clean, crisp, minimal","lighting":"soft diffuse studio lighting","photo":"wide eye-level product photography","medium":"photograph","color_palette":["#31A354","#FFFFFF","#2F6FAE","#D9D9D9"]},"compositional_deconstruction":{"background":"A neutral pale studio backdrop and white tabletop.","elements":[{"type":"obj","bbox":[210,250,460,700],"desc":"A translucent green glass apple with bright highlights."},{"type":"obj","bbox":[520,260,800,720],"desc":"A simple blue ceramic cup beside the apple."}]}}'
IDEOGRAM_TALL_PROMPT='{"high_level_description":"A clean tall poster of blue mountains under a warm yellow sun on a white background.","style_description":{"aesthetics":"clean, balanced, graphic","lighting":"bright even poster lighting","medium":"graphic_design","art_style":"crisp poster illustration with soft print texture","color_palette":["#2F6FAE","#F5C542","#FFFFFF","#D6E7F5"]},"compositional_deconstruction":{"background":"A pure white poster background with no text or lettering.","elements":[{"type":"obj","bbox":[220,430,810,840],"desc":"Layered blue mountain peaks centered low in the frame."},{"type":"obj","bbox":[360,150,620,390],"desc":"A warm yellow sun above the mountains."}]}}'

run_probe "status-load-matrix-cycle-1" \
  --matrix --no-generate

run_probe "status-load-matrix-cycle-2" \
  --matrix --no-generate

run_probe "zimage-4bit-wide" \
  --model Z-Image-Turbo-mflux-4bit --generate \
  --seed "$SEED" --width 768 --height 512 --steps "$Z_STEPS" \
  --turn "$WIDE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$WIDE_PROMPT"

run_probe "zimage-8bit-large-square" \
  --model Z-Image-Turbo-mflux-8bit --generate \
  --seed "$SEED" --width 768 --height 768 --steps "$Z_STEPS" \
  --turn "$APPLE_PROMPT" --turn "$TALL_PROMPT" --turn "$APPLE_PROMPT"

run_probe "flux-schnell-4bit-tall" \
  --model FLUX.1-schnell-mflux-4bit --generate \
  --seed "$SEED" --width 512 --height 768 --steps "$FLUX_STEPS" \
  --turn "$TALL_PROMPT" --turn "$APPLE_PROMPT" --turn "$TALL_PROMPT"

run_probe "flux-schnell-8bit-wide" \
  --model FLUX.1-schnell-mflux-8bit --generate \
  --seed "$SEED" --width 768 --height 512 --steps "$FLUX_STEPS" \
  --turn "$WIDE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$WIDE_PROMPT"

run_probe "qwen-image-8bit-source-square" \
  --model qwen-image-mflux-8bit --generate \
  --seed "$SEED" --width 512 --height 512 --steps "$QWEN_IMAGE_STEPS" \
  --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

run_probe "qwen-image-8bit-source-wide" \
  --model qwen-image-mflux-8bit --generate \
  --seed "$SEED" --width 768 --height 512 --steps "$QWEN_IMAGE_STEPS" \
  --turn "$WIDE_PROMPT" --turn "$TALL_PROMPT" --turn "$WIDE_PROMPT"

run_probe "qwen-image-6bit-tall" \
  --model Qwen-Image-mflux-6bit --generate \
  --seed "$SEED" --width 512 --height 768 --steps "$QWEN_IMAGE_STEPS" \
  --turn "$TALL_PROMPT" --turn "$APPLE_PROMPT" --turn "$TALL_PROMPT"

QWEN_SOURCE_SQUARE="$(
  jq -r '.generation_turns[] | select(.turn == 1 and .status == "completed") | (.image_diagnostics.path // .output)' \
    "$ART_ROOT/qwen-image-8bit-source-square/qwen-image-mflux-8bit-load.json" 2>/dev/null || true
)"
QWEN_SOURCE_WIDE="$(
  jq -r '.generation_turns[] | select(.turn == 1 and .status == "completed") | (.image_diagnostics.path // .output)' \
    "$ART_ROOT/qwen-image-8bit-source-wide/qwen-image-mflux-8bit-load.json" 2>/dev/null || true
)"

if [[ ! -f "$QWEN_SOURCE_SQUARE" || ! -f "$QWEN_SOURCE_WIDE" ]]; then
  printf 'vmlx-image-extended-stress: qwen source images missing: %s %s\n' "$QWEN_SOURCE_SQUARE" "$QWEN_SOURCE_WIDE" >&2
else
  run_probe "qwen-edit-q8-single-image" \
    --model Qwen-Image-Edit-mflux-q8 --edit --source-image "$QWEN_SOURCE_SQUARE" \
    --seed "$SEED" --width 512 --height 512 --steps "$QWEN_EDIT_STEPS" \
    --turn "$EDIT_BLUE_PROMPT" --turn "$EDIT_PEAR_PROMPT" --turn "$EDIT_BLUE_PROMPT"

  run_probe "qwen-edit-q8-multi-image" \
    --model Qwen-Image-Edit-mflux-q8 --edit \
    --source-image "$QWEN_SOURCE_SQUARE" --source-image "$QWEN_SOURCE_WIDE" \
    --seed "$SEED" --width 768 --height 512 --steps "$QWEN_EDIT_STEPS" \
    --turn "$EDIT_MULTI_PROMPT" --turn "$EDIT_PEAR_PROMPT" --turn "$EDIT_MULTI_PROMPT"

  run_probe "qwen-edit-q5-multi-image" \
    --model Qwen-Image-Edit-mflux-q5 --edit \
    --source-image "$QWEN_SOURCE_SQUARE" --source-image "$QWEN_SOURCE_WIDE" \
    --seed "$SEED" --width 768 --height 512 --steps "$QWEN_EDIT_STEPS" \
    --turn "$EDIT_MULTI_PROMPT" --turn "$EDIT_BLUE_PROMPT" --turn "$EDIT_MULTI_PROMPT"

  run_probe "qwen-edit-q8-mask-reject" \
    --model Qwen-Image-Edit-mflux-q8 --edit \
    --source-image "$QWEN_SOURCE_SQUARE" --mask-image "$QWEN_SOURCE_SQUARE" \
    --seed "$SEED" --width 512 --height 512 --steps "$QWEN_DIAG_STEPS" \
    --turn "$EDIT_BLUE_PROMPT"

  run_probe "qwen-edit-q8-diagnostics" \
    --model Qwen-Image-Edit-mflux-q8 \
    --qwen-edit-prompt --qwen-edit-conditioning --qwen-edit-vision --qwen-edit-denoise \
    --source-image "$QWEN_SOURCE_SQUARE" --source-image "$QWEN_SOURCE_WIDE" \
    --seed "$SEED" --width 768 --height 512 --steps "$QWEN_DIAG_STEPS" \
    --turn "$EDIT_MULTI_PROMPT"
fi

run_probe "ideogram-fp8-wide-json" \
  --model ideogram-4-fp8 --generate \
  --seed "$IDEOGRAM_SEED" --width 768 --height 512 --steps "$IDEOGRAM_STEPS" \
  --turn "$IDEOGRAM_WIDE_PROMPT" --turn "$IDEOGRAM_TALL_PROMPT" --turn "$IDEOGRAM_WIDE_PROMPT"

run_probe "ideogram-nf4-tall-json" \
  --model ideogram-4-nf4 --generate \
  --seed "$IDEOGRAM_SEED" --width 512 --height 768 --steps "$IDEOGRAM_STEPS" \
  --turn "$IDEOGRAM_TALL_PROMPT" --turn "$IDEOGRAM_WIDE_PROMPT" --turn "$IDEOGRAM_TALL_PROMPT"

SUMMARY="$ART_ROOT/extended-stress-summary.json"
ROWS_JSON='[]'

summarize_matrix "status-load-matrix-cycle-1"
summarize_matrix "status-load-matrix-cycle-2"
summarize_turns "zimage-4bit-wide" "Z-Image-Turbo-mflux-4bit-load.json" "generation_turns" 3 2 1
summarize_turns "zimage-8bit-large-square" "Z-Image-Turbo-mflux-8bit-load.json" "generation_turns" 3 2 1
summarize_turns "flux-schnell-4bit-tall" "FLUX.1-schnell-mflux-4bit-load.json" "generation_turns" 3 2 1
summarize_turns "flux-schnell-8bit-wide" "FLUX.1-schnell-mflux-8bit-load.json" "generation_turns" 3 2 1
summarize_turns "qwen-image-8bit-source-square" "qwen-image-mflux-8bit-load.json" "generation_turns" 3 2 1
summarize_turns "qwen-image-8bit-source-wide" "qwen-image-mflux-8bit-load.json" "generation_turns" 3 2 1
summarize_turns "qwen-image-6bit-tall" "Qwen-Image-mflux-6bit-load.json" "generation_turns" 3 2 1
summarize_turns "qwen-edit-q8-single-image" "Qwen-Image-Edit-mflux-q8-load.json" "edit_turns" 3 2 1
summarize_turns "qwen-edit-q8-multi-image" "Qwen-Image-Edit-mflux-q8-load.json" "edit_turns" 3 2 1
summarize_turns "qwen-edit-q5-multi-image" "Qwen-Image-Edit-mflux-q5-load.json" "edit_turns" 3 2 1
summarize_mask_rejection "qwen-edit-q8-mask-reject" "Qwen-Image-Edit-mflux-q8-load.json"
summarize_qwen_diagnostics "qwen-edit-q8-diagnostics" "Qwen-Image-Edit-mflux-q8-load.json"
summarize_turns "ideogram-fp8-wide-json" "ideogram-4-fp8-load.json" "generation_turns" 3 2 1
summarize_turns "ideogram-nf4-tall-json" "ideogram-4-nf4-load.json" "generation_turns" 3 2 1

FAILED="$(jq -r '[.[] | select(.status != "passed")] | length' <<< "$ROWS_JSON")"
if [[ "$FAILED" == "0" ]]; then
  SUMMARY_STATUS="passed"
else
  SUMMARY_STATUS="failed"
fi

jq -n \
  --arg status "$SUMMARY_STATUS" \
  --arg generated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg artifact_root "$ART_ROOT" \
  --arg output_root "$OUT_ROOT" \
  --arg model_root "${MODEL_ROOT:-default}" \
  --arg probe "$PROBE" \
  --arg source_square "$QWEN_SOURCE_SQUARE" \
  --arg source_wide "$QWEN_SOURCE_WIDE" \
  --argjson failed "$FAILED" \
  --argjson rows "$ROWS_JSON" \
  '{
    status: $status,
    generated_at: $generated_at,
    artifact_root: $artifact_root,
    output_root: $output_root,
    model_root: $model_root,
    probe: $probe,
    qwen_sources: {
      square: $source_square,
      wide: $source_wide
    },
    failed_rows: $failed,
    rows: $rows,
    stress_scope: [
      "two complete load-only matrix cycles",
      "wide/tall/large-square generation sizes",
      "qwen-image source regeneration for edit chaining",
      "qwen single-image and ordered multi-image edits",
      "qwen unsupported-mask rejection",
      "qwen prompt, conditioning, vision-language, and denoise diagnostics",
      "per-row /usr/bin/time -l max RSS/elapsed evidence"
    ],
    visual_gate: "View generated PNGs before claiming visual quality or app release readiness.",
    osaurus_gate: "Osaurus HTTP/UI bridge proof is outside this CLI runner and remains required before app-side readiness claims."
  }' > "$SUMMARY"

printf 'extended stress summary: %s\n' "$SUMMARY"
printf 'status=%s failed_rows=%s\n' "$SUMMARY_STATUS" "$FAILED"
jq -r '.rows[] | "\(.label): \(.status) kind=\(.kind) max_rss=\(.time.max_rss_bytes // "n/a") artifact=\(.artifact)"' "$SUMMARY"
printf '\nStress artifacts: %s\n' "$ART_ROOT"
printf 'Stress outputs:   %s\n' "$OUT_ROOT"
printf 'Summary:          %s\n' "$SUMMARY"
printf 'PARTIAL until generated PNGs are visually inspected and Osaurus HTTP/UI bridge proof exists.\n'

if [[ "$FAILED" != "0" ]]; then
  exit 1
fi
