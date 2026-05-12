#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Package pins =="
swift package resolve

echo "== Build facade target =="
swift build --target VMLXSwift

echo "== Build CLI =="
swift build --product vmlx-swift

echo "== Run CLI smoke =="
swift run vmlx-swift version

echo "== Check for local package paths =="
if rg -n 'package\(path:|/Users/' Package.swift Package.resolved; then
  echo "Found local package path in public package files" >&2
  exit 1
fi

echo "== Dependency graph =="
swift package show-dependencies
