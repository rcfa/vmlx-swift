#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD_CONFIGURATION=${MLXPRESS_BUILD_CONFIGURATION:-debug}
ARCH_TRIPLE=${MLXPRESS_BUILD_TRIPLE:-arm64-apple-macosx}
GENERATED_METAL_DIR="${ROOT_DIR}/Source/Cmlx/mlx-generated/metal"

declare -a OUT_PATHS=()

add_output_pair() {
  local directory="$1"
  OUT_PATHS+=("${directory}/mlx.metallib")
  OUT_PATHS+=("${directory}/default.metallib")
}

if [[ $# -gt 0 ]]; then
  requested="$1"
  if [[ "$requested" = /* ]]; then
    out_path="$requested"
  else
    out_path="${ROOT_DIR}/${requested}"
  fi
  OUT_PATHS+=("$out_path")
  case "$(basename "$out_path")" in
    mlx.metallib)
      OUT_PATHS+=("$(dirname "$out_path")/default.metallib")
      ;;
    default.metallib)
      OUT_PATHS+=("$(dirname "$out_path")/mlx.metallib")
      ;;
  esac
else
  add_output_pair "${ROOT_DIR}/.build/${BUILD_CONFIGURATION}"
  add_output_pair "${ROOT_DIR}/.build/${ARCH_TRIPLE}/${BUILD_CONFIGURATION}"
fi

outputs_ready() {
  local output
  for output in "${OUT_PATHS[@]}"; do
    [[ -s "$output" ]] || return 1
  done
  return 0
}

install_metallib() {
  local source="$1"
  local output
  for output in "${OUT_PATHS[@]}"; do
    mkdir -p "$(dirname "$output")"
    cp "$source" "$output"
  done
}

find_xcrun() {
  local dev_dir
  if xcrun -find metal >/dev/null 2>&1 && xcrun -find metallib >/dev/null 2>&1; then
    printf "xcrun"
    return 0
  fi
  dev_dir="/Applications/Xcode.app/Contents/Developer"
  if [[ -d "$dev_dir" ]] &&
    DEVELOPER_DIR="$dev_dir" xcrun -find metal >/dev/null 2>&1 &&
    DEVELOPER_DIR="$dev_dir" xcrun -find metallib >/dev/null 2>&1; then
    printf "DEVELOPER_DIR=%q xcrun" "$dev_dir"
    return 0
  fi
  return 1
}

compile_metallib() {
  [[ -d "$GENERATED_METAL_DIR" ]] || return 1

  local xcrun_prefix
  xcrun_prefix=$(find_xcrun) || return 1

  local work_dir="${ROOT_DIR}/.build/mlx-metal-prep"
  local air_dir="${work_dir}/air"
  local out="${work_dir}/mlx.metallib"
  rm -rf "$work_dir"
  mkdir -p "$air_dir"

  local src
  while IFS= read -r src; do
    local stem
    stem=$(basename "$src" .metal)
    eval "$xcrun_prefix" -sdk macosx metal \
      -x metal \
      -Wall \
      -Wextra \
      -fno-fast-math \
      -Wno-c++17-extensions \
      -Wno-c++20-extensions \
      -mmacosx-version-min=14.0 \
      -c "$src" \
      "-I${GENERATED_METAL_DIR}" \
      -o "${air_dir}/${stem}.air"
  done < <(find "$GENERATED_METAL_DIR" -name '*.metal' | sort)

  eval "$xcrun_prefix" -sdk macosx metallib "${air_dir}"/*.air -o "$out"
  [[ -s "$out" ]] || return 1
  install_metallib "$out"
  return 0
}

copy_candidate() {
  local source="$1"
  if [[ -s "$source" ]]; then
    install_metallib "$source"
    return 0
  fi
  return 1
}

if outputs_ready; then
  exit 0
fi

if [[ -n "${MLXPRESS_MLX_METALLIB:-}" ]]; then
  if ! copy_candidate "$MLXPRESS_MLX_METALLIB"; then
    echo "MLXPRESS_MLX_METALLIB did not point to a readable metallib: $MLXPRESS_MLX_METALLIB" >&2
    exit 1
  fi
  exit 0
fi

if compile_metallib; then
  exit 0
fi

for candidate in \
  "${ROOT_DIR}/.build/${ARCH_TRIPLE}/debug/mlx.metallib" \
  "${ROOT_DIR}/.build/${ARCH_TRIPLE}/debug/default.metallib" \
  "${ROOT_DIR}/.build/${ARCH_TRIPLE}/release/mlx.metallib" \
  "${ROOT_DIR}/.build/${ARCH_TRIPLE}/release/default.metallib" \
  "${ROOT_DIR}/../vmlx-swift-lm/.build/${ARCH_TRIPLE}/debug/mlx.metallib" \
  "${ROOT_DIR}/../vmlx-swift-lm/.build/${ARCH_TRIPLE}/debug/default.metallib" \
  "${ROOT_DIR}/../vmlx-swift-lm/.build/${ARCH_TRIPLE}/release/mlx.metallib" \
  "${ROOT_DIR}/../vmlx-swift-lm/.build/${ARCH_TRIPLE}/release/default.metallib" \
  "${ROOT_DIR}/../vmlx/swift/.build/${ARCH_TRIPLE}/debug/mlx.metallib" \
  "${ROOT_DIR}/../vmlx/swift/.build/${ARCH_TRIPLE}/debug/default.metallib" \
  "${ROOT_DIR}/../vmlx/swift/.build/${ARCH_TRIPLE}/release/mlx.metallib" \
  "${ROOT_DIR}/../vmlx/swift/.build/${ARCH_TRIPLE}/release/default.metallib" \
  "${ROOT_DIR}/../vmlx/swift/Sources/Cmlx/default.metallib"; do
  if copy_candidate "$candidate"; then
    exit 0
  fi
done

cat >&2 <<'EOF'
Unable to prepare MLX metallibs.

The command-line SwiftPM build does not emit MLX's Metal kernel library. This
script first tries to compile Source/Cmlx/mlx-generated/metal/*.metal with the
installed Xcode Metal toolchain, then falls back to MLXPRESS_MLX_METALLIB or
known local sibling build artifacts. Install Xcode's Metal Toolchain component
or set MLXPRESS_MLX_METALLIB=/path/to/mlx.metallib.
EOF
exit 1
