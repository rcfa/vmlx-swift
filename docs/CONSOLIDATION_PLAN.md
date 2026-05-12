# Consolidation plan

This repo is the consolidation target for Osaurus Swift ML runtime libraries.
The goal is to let Osaurus eventually depend on one repo without losing model
family fixes, parser behavior, cache contracts, or Swift package product
names.

## Source repos

| Repo | Current role | Status |
|---|---|---|
| `osaurus-ai/vmlx-swift-lm` | language/VLM runtime, JANG/JANGTQ, cache, parsers, distributed primitives | clean at `b166896` when this scaffold was created |
| `osaurus-ai/mlx-swift` | MLX core fork | use the pinned remote SHA; do not repoint a protected local fork blindly |
| `osaurus-ai/swift-transformers` | tokenizer, Hub, generation helpers | clean at `087a66b` when this scaffold was created |
| `osaurus-ai/Jinja` | Jinja renderer used by transformers and vmlx-swift-lm | pinned to Osaurus fork |
| integrated vMLX app/server/runtime repo | older app/server/runtime work | dirty local reference material; reconcile intentionally before import |
| `osaurus-ai/osaurus` | consumer app/runtime integration | validate package shape and model policy from Osaurus tests |

## Non-negotiable migration constraints

- Preserve public product names that Osaurus imports today, including `MLX`,
  `MLXNN`, `MLXLMCommon`, `MLXLLM`, `MLXVLM`, `MLXEmbedders`, `Hub`,
  `Tokenizers`, `Transformers`, and `Jinja`.
- Preserve vmlx `Chat.Message`, `UserInput`, `GenerateParameters`, and
  `Generation` event contracts.
- Preserve JANG capability stamp resolution for reasoning and tool parsers:
  caller override, then JANG stamp, then model_type heuristic.
- Preserve bundle `generation_config.json` behavior. Do not add sampling
  defaults to hide a template/parser bug.
- Preserve cache topology boundaries. Paged KV, rotating KV, TurboQuant KV,
  Mamba/SSM, ArraysCache, ZayaCCACache, CacheList, and media-salted VLM inputs
  must remain distinguishable.
- Preserve Osaurus package hygiene: no local paths, no stale upstream fork URLs,
  no uncommitted dependency fixes behind a pinned SHA.

## Phases

### Phase 0: facade package

The current repo exports a `VMLXSwift` facade product and pins the known-good
dependency SHAs. This gives Osaurus and plugins a stable place to test package
identity without moving source trees yet.

Done when:

- `swift build --target VMLXSwift` passes on a clean checkout.
- `swift run vmlx-swift version` passes on a clean checkout.
- `Package.resolved` resolves all Osaurus forks to expected SHAs.
- Osaurus can add this package without changing runtime behavior.

Distributed/JACCL products are gated out of phase 0 because the current pins
try to build `CmlxDistributedShim` against MLX C distributed headers that are
not present in the pinned MLX package. They belong in a separate source-import
slice with explicit MLX C header support and distributed runtime tests.

### Phase 1: source import

Bring sources into one tree while preserving module names.

Recommended layout:

```text
Sources/
  Cmlx/
  MLX/
  MLXNN/
  MLXRandom/
  MLXOptimizers/
  MLXFast/
  MLXFFT/
  MLXLinalg/
  MLXLMCommon/
  MLXLLM/
  MLXVLM/
  MLXEmbedders/
  MLXHuggingFace/
  Hub/
  Tokenizers/
  Generation/
  Models/
  Jinja/
```

Do not do this as a bulk copy. Import one source family at a time and run the
contract checks after each import.

### Phase 2: Osaurus consumer switch

Switch Osaurus from `vmlx-swift-lm`, `mlx-swift`, `swift-transformers`, and
`Jinja` to this single repo. The Osaurus diff should be mostly Package.swift
and Package.resolved changes plus import cleanup if needed.

Done when:

- Existing Osaurus tests pass.
- Runtime matrix has clean rows for the active model families.
- App UI manual checks confirm reasoning, media, tool, cache, and stop/cancel
  behavior.

## Known local hazards

- A protected local MLX fork had a disabled push URL and dirty
  submodule/profile state when this scaffold was created. It is not safe to
  repoint that checkout to `osaurus-ai/vmlx-swift`.
- The older integrated vMLX app/server/runtime checkout had useful code but a
  dirty working tree. Treat it as reference material, not as source of truth.
- Osaurus runtime fixes can span four repos at once. A green build against a
  local path is not proof that a remote pin is ready.

## Required checks before source import

- Compare package products and target names across source repos.
- Build each source repo at the exact SHA to be imported.
- Run the runtime coverage matrix in this repo after each import group.
- Run Osaurus no-load policy tests after each public API change.
- Run at least one real-model smoke for every architecture bucket before
  replacing Osaurus pins.
