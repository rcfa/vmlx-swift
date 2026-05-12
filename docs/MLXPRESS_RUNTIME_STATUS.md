# MLXPress Runtime Status

Last updated: 2026-05-12

This repo is being prepared as the `mlx-swift` foundation for MLXPress/JangPress
runtime work in the Osaurus stack. The current scope is library readiness:
runtime-status ABI hooks, mmap safetensors support, and the Osaurus MLX
retained-buffer backport needed by cache-heavy concurrent decode paths.

## Current Library Baseline

- Top-level branch: `vmlx-0.31.3`.
- `Source/Cmlx/mlx`: `885e5d82f3e23bd4d7b7e55d11e5536c4b5b6378`.
  - Based on Osaurus MLX `7086ba37b1250ba2622a66b181de33e135af6484`.
  - Preserves the local vMLX custom-kernel output-shape patch.
  - Adapts the retained-buffer backport to this branch's `Device` stream API.
  - Adds safetensors mmap expert-advice hooks.
- `Source/Cmlx/mlx-c`: `ddef9122f990014d2b1d78ecccb07cd44c848301`.
  - Adds C ABI wrappers for the mmap advice hooks.
- `.gitmodules` now points `Source/Cmlx/mlx` at `https://github.com/osaurus-ai/mlx`.

## Implemented Runtime Surface

- `MLX_SAFETENSORS_MMAP=1` or `VMLINUX_MMAP_SAFETENSORS=1` enables mmap-backed
  safetensors loading for file-path loads.
- Loaded tensor arrays share the mmap-backed shard buffer instead of copying the
  tensor payload into separate allocations.
- Routed expert regions are registered by `(layer, expert)` when tensor names
  match known MoE/JANGQ/JANGTQ patterns.
- Public C++ API:
  - `safetensors_mmap_advise_routed(int32_t advice, int32_t cold_pct)`
  - `safetensors_mmap_advise_experts(int32_t advice, const int32_t* layers, const int32_t* experts, int64_t count)`
- Public C ABI:
  - `mlx_safetensors_mmap_advise_routed`
  - `mlx_safetensors_mmap_advise_experts`

## Verification

Passing:

- `swift build --target Cmlx`
- `swift build --target MLX`
- `swift build -c release --target Cmlx`
- `swift build -c release --product Example1`
- `nm -gU .build/arm64-apple-macosx/release/Example1 | rg 'mlx_safetensors_mmap_advise'`
- `git diff --check` in the top repo and both nested Cmlx submodules

Blocked:

- `swift test --filter CmlxTests/testSafetensorsMmapAdviceSymbolsAreExported`
  still fails before assertion with `no such module 'XCTest'`.

## Boundaries

- This is not yet a coherence claim for any model family.
- MLXPress has not yet been wired to call the advice ABI from Swift runtime code.
- Cache-stack enabled generation has not yet been validated for JANGQ/JANGTQ,
  reasoning models, hybrid SSM, sliding-window attention, or multimodal variants.
- The pre-existing local deletions of `xcode/default.profraw` and
  `Source/Cmlx/mlx-c/examples/arrays.safetensors` were left untouched.

## Next Work

1. Wire MLXPress runtime status and mmap advice calls through `vmlx-swift-lm`.
2. Keep speculative/MTP decode disabled for base MLXPress unless explicitly
   enabled by a later topology-specific implementation.
3. Validate cache-stack-enabled coherent output per model family and attention
   architecture, starting with the locally downloaded JANGQ/JANGTQ families.
4. Document each family as implemented, gated, or blocked with the exact command
   and model path used for proof.
