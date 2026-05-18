# vMLX Swift / Osaurus PR Engine Coverage Audit - 2026-05-18

This is the current crosswalk from recent user-authored Osaurus PRs and pinned
runtime libraries to `vmlx-swift` engine proof. It is intentionally stricter
than a package build: a row is not switch-ready unless the exact runtime surface
has model-aware cache proof, multi-turn coherency, visible stop behavior, and no
hidden sampler/parser guard.

Fresh inspection commands used in this pass:

```sh
gh pr list -R osaurus-ai/osaurus --author @me --state all --search "created:>=2026-04-24" --json number,title,state,url,headRefOid,updatedAt,mergedAt,closedAt,isDraft,mergeStateStatus --limit 100
gh pr view -R osaurus-ai/osaurus <pr> --json number,title,state,headRefOid,mergeCommit,commits,files
git -C /Users/eric/osaurus-staging show HEAD:osaurus.xcworkspace/xcshareddata/swiftpm/Package.resolved
gh api 'repos/osaurus-ai/{repo}/commits?since=2026-04-24T00:00:00Z&until=2026-05-18T23:59:59Z&per_page=100'
```

Current `vmlx-swift` branch head at audit time:

```text
6560879 fix(cache): preserve prompt tail in TurboQuant KV
```

## Current Switch Verdict

Not ready to say "single `vmlx-swift` dependency is production-clear for all
Osaurus models." Large parts are proven, but there are still explicit open
promotion blockers:

- DSV4: coherent post-fix chat exists, but long-context/vector drift, API
  route matrix, speed matrix, and low-footprint production gates remain open.
- ZAYA1-VL JANGTQ_K: still fails the production math row and cold structured VL
  cache budget. This cannot be hidden with top-k, repetition penalty, or looser
  validators.
- MiniMax large CRACK: cache-chat and strict TQ B=2 now pass after the real TQ
  cache-codec tail fix, but low-footprint active-routed proof is still open.
- Hy3 JANGTQ_K: old active-streaming evidence exists, but it needs a current
  non-Kimi all-model rerun before Osaurus promotion.
- GPT-OSS / GLM5 / Mistral4 / Pixtral: parser/unit coverage exists, but there
  are no local live decode rows in this pass.
- Omni live voice: core text/image/audio/video rows pass, but repeated
  cache-on live audio remains a focused quality/root-cause gate.
- Qwen high-resolution video: bounded media resize rows pass; raw 1080p video
  is not production-clear because the pre-fix row peaked at 164.2 GiB physical
  footprint.

## Osaurus PR Crosswalk

| PR | State | Runtime payload | Current `vmlx-swift` coverage | Remaining requirement |
| --- | --- | --- | --- | --- |
| #931 `fix(ci): bump vmlx-swift-lm pin to 5b84387` | merged | Early resolver pin movement. | Captured in `docs/VMLX_OSAURUS_PR_PIN_LINEAGE_2026_05_17.md`; later pins supersede this. | No standalone engine blocker; use as lineage only. |
| #932 `feat: honor per-model generation_config.json sampling defaults` | merged | Bundle `generation_config.json` defaults must flow into local generation. | Current ledger rows report bundle defaults per family: e.g. MiniMax `temp=1.000 topP=0.950 topK=40 rep=nil`, Qwen `topK=20`, ZAYA `temp=0.600 topP=1.000 topK=0`, Laguna `temp=0.700 topP=0.900`. | Keep every new live row printing resolved defaults. Do not add hidden fallback penalties or top-k clamps when a model loops. |
| #944 `feat: jang_config.json chat metadata + vmlx bump` | merged | `jang_config.json` chat metadata, DSV4/Kimi/LFM routing, reasoning capability metadata. | DSV4 template/metadata rows and the non-Kimi config/template sweep are current. Kimi is deliberately excluded by user direction. | LFM is not live-cleared by this matrix; if Osaurus exposes it, add a live multi-turn/cache row. |
| #946 `feat(model-picker): Performance filter` | merged | UI filtering based on model performance/fit. | Engine docs now record speed/RSS caveats by family, especially DSV4, MiniMax, ZAYA, Qwen, Gemma4, and Omni. | Osaurus UI must consume these as explicit capability/performance metadata, not infer from name or size alone. |
| #953 `fix(preflight): detect mislabeled JANGTQ bundles + vmlx fa77575 auto-correct` | merged | Mislabeled JANGTQ detection, sidecar/family preflight, streaming/event mapping. | Current ledger keeps tensor/sidecar evidence separate from model names. ZAYA/MiniMax/Qwen CRACK rows explicitly state non-MTP unless tensor evidence exists. | Add switch-PR resolver tests that reject name-only MTP/JANGTQ claims. |
| #967 `feat: Nemotron-3 Hybrid + storage fix + multimodal API` | merged | Nemotron hybrid, multimodal content parts, storage, early resolver skew. | Fresh Omni rows cover JANGTQ/JANGTQ4/MXFP4 core text/image/audio/video, media salt, hybrid SSM, and BatchEngine rows. | Repeated cache-on audio quality remains partial; resolver skew means the Osaurus switch PR must pin one path only. |
| #990 `feat(api): OpenAI input_audio + video_url content parts` | closed/unmerged | API-surface context for audio/video content parts. | Do not treat as shipped resolver state. Its content is effectively covered later by #967/#1073 live voice/multimodal rows. | No direct switch blocker, but API route probes remain package-wide open. |
| #993 `fix(preflight): reject JANGTQ Mistral 3 / Laguna before vmlx loads` | merged | Preflight/fail-fast behavior and converged Jinja identity. | Laguna is live-proven; Mistral3/Laguna parser and JANGTQ Hadamard fixes are represented in parser/cache refresh and ledger notes. | If Osaurus re-enables Mistral3 JANGTQ, require a current live row; do not rely on preflight-only proof. |
| #998 `fix(quality): revert default KV mode .turboQuant(4,4) -> .none` | merged | Important no-fake-default precedent: global TQ caused degenerate repetition, real fix was to stop forcing it. | `vmlx-swift` now uses explicit TQ rows only. MiniMax strict TQ B=2 was fixed by preserving exact prompt tail in the TQ codec, not by forcing sampler policy. | Keep TQ off unless explicitly selected or model-compatible; never use global TQ as quality default. |
| #1037 `Ling/ZAYA hardening + BatchEngine lifecycle` | merged | Ling/Bailing, ZAYA, BatchEngine lifecycle, topology-aware cache. | Ling JANGTQ2/MXFP4 pass; ZAYA text JANGTQ4/JANGTQ_K pass; ZAYA1-VL JANGTQ4 passes. Cache proof is topology-specific: disk/SSM/CCA, not generic prefix hit. | ZAYA1-VL JANGTQ_K remains partial; Hy3 K needs current rerun. |
| #1057 `MiniMax speed fix` | merged | MiniMax speed/lifecycle, typed load config, VLM detection, tokenizer/Jinja compatibility. | Large MiniMax JANGTQ_K/JANG cache-off infer rows pass; production-shaped chat-cache row passes; strict TQ B=2 now passes after `6560879`. | Low-footprint active-routed MiniMax proof is still open. Shape-inferred 6-bit metadata repair in JANG_K should be corrected in bundle or explicitly accepted. |
| #1066 `pin DSV4 vmlx update` | merged | DSV4 tokenizer/cache/runtime pin, local tokenizer fallback. | DSV4 separator fix and template kwargs rows pass; DSV4 live cache OFF/ON chat is coherent. | DSV4 remains partial until long-context/vector/API/speed/low-footprint gates pass. |
| #1073 `Nemotron Omni live voice input path` | merged | Live voice, Parakeet/RADIO, media-cache token-aware restore, DSV4 pool/compressor fixes. | Omni JANGTQ/JANGTQ4/MXFP4 core matrices pass; current docs track Parakeet chunk concat caveat. DSV4 pool/compressor lineage is recorded. | Repeated cache-on audio and package-wide HTTP route proof remain open. |
| #1110 `Harden DSV4 reasoning gates and runtime proof` | open, dirty | Native DSV4 chat encoder/tokenizer bridge, live DSV4 proof, runtime pin check. Current Osaurus head pins `vmlx-swift-lm 2cc64dd`. | `vmlx-swift` has DSV4 prompt-boundary fix and partial live proof, but it does not yet close the full #1110 bar. | Do not treat #1110 as merged release state; switch PR must resolve dirty state and rerun DSV4 release gates. |

## Pinned Dependency Window

Current open Osaurus PR head (#1110) resolves:

| Package | Revision | Commit fact | `vmlx-swift` requirement |
| --- | --- | --- | --- |
| `osaurus-ai/mlx-swift` | `0a56f904` | `2026-05-01 deps(mlx): advance submodule to 96aa27a5 (mx::malloc tracer for Bug 2)` | Compare behavior through local Cmlx/MLX checkout; package identity alone is not enough. |
| `osaurus-ai/Jinja` | `58d21aa` | `2026-05-01 fix(parser): for-loop iterable accepts binary expressions` | Vendored Jinja fallback tests must keep binary iterable and `tojson(separators:)` behavior. |
| `osaurus-ai/swift-transformers` | `087a66b` | `2026-05-11 fix(tokenizer): skip unused placeholders in delimiter regex` | Tokenizers must skip `<unusedN>` placeholders and preserve wrapper-token paths for MiniMax, DSV4, Qwen, and Omni. |
| `osaurus-ai/vmlx-swift-lm` | `2cc64dd` | `2026-05-15 Wire native DSV4 chat encoder` | DSV4 native chat encoder/tokenizer bridge is part of the switch-readiness target, not yet fully released in `vmlx-swift` by a complete DSV4 gate. |

GitHub commit scan for `swift-transformers` also shows default-branch tokenizer
speed work (`MetaspaceDecoder`, byte-level pre-tokenizer, Bert regex,
SentencePiece O(N) improvements). Those are not the current Osaurus resolver
pin unless the switch PR moves the pin or vendors equivalent code. Treat them
as performance-watch items, not proven-current Osaurus runtime behavior.

GitHub commit scan for `mlx-swift` returned no additional commits in the
2026-04-24 to 2026-05-18 window beyond the pinned `0a56f904` fact checked
directly above.

## Dependency Fixes Mapped To Engine Surfaces

| Upstream fix family | Required engine surface | Current proof | Status |
| --- | --- | --- | --- |
| Jinja parser and compact tool JSON | DSV4/Kimi/Gemma4/ZAYA/Laguna tool templates render without broken syntax or bloated separators. | `docs/local/production-readiness/20260517T2200_jinja_pin_parity/` and parser/cache refresh rows. | Covered for non-Kimi; Kimi excluded by instruction. |
| Swift-transformers unused placeholder skip | Added-token delimiter regex must not include thousands of unused placeholders and must preserve special wrapper tokens. | Vendored tokenizer static check plus MiniMax/DSV4/Qwen/Omni live template rows. | Covered for pinned behavior; later speed commits not yet part of Osaurus pin. |
| Generation defaults | Bundle defaults are source of truth before explicit request override. | Ledger rows print temp/topP/topK/minP/rep per family. | Covered for tested rows; require same telemetry for new rows. |
| Hybrid SSM / CCA / SWA cache | Cache proof must be topology-specific, not generic prefix-hit. | Qwen/Ling/ZAYA/Gemma4 rows record disk L2, SSM companion, CCA, SWA incompatibility, and media salt where applicable. | Covered for listed PASS rows; DSV4 long-context and ZAYA1-VL K remain open. |
| TurboQuant KV | Explicit TQ mode must preserve coherency and prove actual compression. | `20260518T_minimax_m27_jangtqk_tq_tail_fix_exact/` proves actual TQ transitions and exact outputs after tail preservation. | Fixed for MiniMax strict row; keep family-by-family gates. |
| VL/media salt | Image/video/audio state must be isolated across turns and cache hits. | Qwen, ZAYA1-VL, Gemma4, and Omni rows prove same/different media behavior where implemented. | Raw Qwen high-res video and repeated Omni cache-on audio remain open. |
| Reasoning on/off | No fake close; reasoning off must affect template/runtime where supported, and visible output must remain coherent. | Gemma4 reasoning matrix, MiniMax rows, DSV4 reasoning kwargs, Ling/Bailing aliases. | Covered for tested families; package-wide model matrix still open for absent local bundles. |
| MTP autodetect | Only real tensor evidence may enable MTP; model names and stale metadata are insufficient. | Non-Kimi MTP census and Qwen MTP settings docs; CRACK rows explicitly stay MTP off. | Correct policy documented; full MTP speed target remains separate/open. |

## Production-Quality Checklist Still Required

Before the Osaurus switch PR can honestly say `vmlx-swift` replaces the split
libraries, every exposed model family needs:

1. Config/template row with resolved `generation_config.json` defaults.
2. Multi-turn live row with visible coherent output, normal stop, no loops, no
   gibberish, no hidden reasoning-only fake pass.
3. Reasoning on/off/effort row where the family supports reasoning.
4. Tool parser row where the family supports tools.
5. Cache-on/cache-off row with topology-specific stats:
   prefix/paged/L2 disk/TurboQuant/SSM/CCA/SWA/media salt as applicable.
6. Continuous batching row for single-slot and B=2 behavior.
7. VL/video/audio row with real media payloads where the family supports media.
8. Speed and RAM row, including tok/s and physical-footprint caveat.
9. Failure rows left visible with artifact paths; no sampler guard, fake EOS,
   fake `</think>`, name-based MTP, or forced repetition penalty.

The active ledger files for those rows remain:

```text
docs/VMLX_SWIFT_MODEL_CAPABILITY_LEDGER.md
docs/VMLX_ACTIVE_MODEL_PRODUCTION_SCOPE_2026_05_17.md
docs/VMLX_OSAURUS_PR_PIN_LINEAGE_2026_05_17.md
```
