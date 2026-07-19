# Gemma 4 QAT cache correctness checkpoint — 2026-07-19

Status: **PARTIAL — the exact current vMLX `bbc0b20d` pin is live-proven in an
isolated Release Osaurus build for the Gemma 4 12B JANG_4M and MXFP8 native,
TurboQuant, SSD-only restart/partial-restore, explicit SSD-Off, and settings
default rows. The 31B RAM-override control works, but its Activity Monitor
Memory column exceeded the bundle's on-disk size, so the low-footprint row is
failed rather than release-cleared.**

This checkpoint is intentionally limited to locally installed Gemma 4 MXFP8
and JANG_4M bundles through the real Osaurus/vMLX Swift runtime. MXFP4 is not a
substitute test artifact and is not part of this checkpoint.

## Source contract

- Ordinary Osaurus single-batch loads keep paged RAM KV off by default.
- Engine-selected/native cache mode keeps TurboQuant KV off. TurboQuant is an
  explicit user opt-in with explicit key/value bit widths.
- Block-disk L2 is on by default and is independent of paged RAM cache. An
  explicit block-disk Off must survive memory-safety policy resolution.
- The tested Gemma 4 12B bundles declare 48 attention layers: 8 full-attention
  `KVCacheSimple` layers and 40 `RotatingKVCache` sliding-attention layers.
- TurboQuant converts only eligible `KVCacheSimple` layers. It must preserve
  all rotating SWA layers as rotating caches.
- The shared conversion hook is type-selective, not architecture-name driven:
  it converts `KVCacheSimple` only and leaves rotating, DeepseekV4, Mamba,
  Arrays/CCA, and composite cache entries native. For hybrid families, an
  explicit TQ request therefore applies only to actual full-attention KV
  layers; companion state must be restored in its matching native type or the
  prefix hit must be rejected and synchronously rederived.
- `TurboQuantCacheTransitionSnapshot` records the real before/after cache
  classes at the conversion point. A configured TurboQuant mode or a non-zero
  compression-event counter is not accepted as layer-level proof.
- Prompt-boundary L2 entries may intentionally remain typed raw KV plus typed
  rotating state even when live decode uses TurboQuant. Telemetry must report
  the live codec and stored codec separately; neither implies the other.
- Gemma mixed SWA/full attention has no recurrent SSM/GLA companion state.
  Hybrid SSM/GDN/CCA async-rederive gates therefore remain separate family
  rows and must not be inferred from Gemma evidence.

## Current evidence

| Row | Current evidence | Status |
|---|---|---|
| Stale `<end_of_turn>` stop | vMLX `604d24e4`; focused Gemma 3/4 regression 3/3; live MXFP8 and JANG_4M both emitted a literal marker and continued through `FINAL-OMEGA` | VERIFIED-LIVE |
| Default paged policy | Exact Release binary at `bbc0b20d`: Settings visibly showed paged off and `/admin/cache-stats` reported requested/effective paged false. Explicit paged On on Gemma truthfully remained effective false with `is_paged_incompatible=true` | VERIFIED-LIVE |
| MXFP8, TurboQuant off | Current build visibly emitted `MXFP8-NATIVE-COLD-7412` at 31.9 tok/s and, after restoring defaults, `MXFP8-DEFAULT-RESTORED-9021` at 32.0 tok/s. Endpoint: native fp16, 8 KV + 40 rotating, transition null, paged off, SSD on | VERIFIED-LIVE |
| MXFP8, TurboQuant 4/4 | Current build visibly emitted eight exact `MXFP8-TQ-*` lines at 38.0 tok/s. Transition: 8 KV to 8 TQ; all 40 rotating layers preserved | VERIFIED-LIVE |
| MXFP8, TQ restart/partial SSD restore | After quitting/relaunching only the isolated app, changed `MXFP8-RESTORE-*` lines completed at 36.7 tok/s. Endpoint: disk hits 2, misses 13, stores 3; paged hits/misses 0; transition re-established 8-to-8/40 | VERIFIED-LIVE |
| JANG_4M, TurboQuant off | Current build visibly emitted `NATIVE-COLD-6724` at 36.8 tok/s and partial-reuse `NATIVE-PARTIAL-9381` at 41.7 tok/s; endpoint native 8 KV + 40 rotating, transition null, paged off, SSD active | VERIFIED-LIVE |
| JANG_4M, TurboQuant 4/4 | Current build visibly emitted eight exact CACHE lines at 55.2 tok/s; process-restart RESTORE lines at 52.8 tok/s; exact 8 KV to 8 TQ conversion with all 40 rotating layers preserved | VERIFIED-LIVE |
| TurboQuant transition telemetry | `TurboQuantCacheTransitionTelemetryTests` 3/3 on current source plus current-build endpoint transitions for both 12B formats | VERIFIED-SOURCE + VERIFIED-LIVE |
| Explicit block-disk Off | Settings visibly saved SSD Off; `DISK-OFF-CURRENT-2846` completed at 41.4 tok/s; endpoint reported block disk false and all disk/paged counters zero | VERIFIED-SOURCE + VERIFIED-LIVE |
| Engine-selected TurboQuant default | Settings visibly restored Engine Selected; current MXFP8 endpoint reported `effective_kv_mode=fp16`, transition null, paged off, SSD on | VERIFIED-SOURCE + VERIFIED-LIVE |
| RAM safety refusal/override | Strict custom 10% visibly refused the 31B JANG_4M at a 12.8 GiB budget before load. No Automatic Limits then loaded the same model and visibly emitted `RAM-OVERRIDE-3179` at 12.2 tok/s. Endpoint reported `automatic_memory_limits_disabled=true` and allowed the estimated 30.9 GiB request. Safe Auto was restored and saved | VERIFIED-LIVE for control behavior |
| Activity Monitor footprint | Exact proof executable inspected. 31B JANG_4M: main Memory 28.37 GB; inspector Real Memory 19.30 GB, Private 996 MB. Bundle is 25G on disk (25,926,564 KiB). Main Memory exceeded full bundle size, so this does not satisfy the low-footprint gate | FAILED-LIVE |

## Required live closure matrix

Every row must use a fresh Release development build with an isolated bundle
identifier and preferences root. Visible UI behavior and matching runtime
telemetry are both required.

| Gate | Required evidence | Status |
|---|---|---|
| Current-build MXFP8 TQ transition | 48 total / 8 KV / 40 rotating before and 48 / 8 TQ / 40 rotating after, plus coherent visible native/TQ/restart output | VERIFIED-LIVE |
| Current-build JANG_4M TQ transition | Same current-build exact transition and coherent visible native/TQ/restart output | VERIFIED-LIVE |
| Paged default/effective policy | Default visibly off; explicit request on paged-incompatible Gemma truthfully remained effective off with zero paged hits/misses | VERIFIED-LIVE |
| SSD L2 with paged off | Both 12B formats restored partial prefixes from disk after process restart while paged counters remained zero | VERIFIED-LIVE |
| Explicit SSD L2 off | Visible SSD-Off save plus endpoint false/zero-counter proof | VERIFIED-LIVE |
| Fresh-process L2 restore | Both 12B formats showed post-restart disk hits and coherent changed-prefix continuations | VERIFIED-LIVE |
| Raw-prefill fallback | Use a cache-salted or genuinely new prefix absent from both tiers; paged/disk miss counters increase, real prefill progress is visible, output remains coherent | OPEN |
| TurboQuant after L2 restore | Post-restart MXFP8 and JANG_4M rows re-established 8 TQ layers while preserving 40 rotating layers | VERIFIED-LIVE |
| TurboQuant off control | Current restored-default MXFP8 endpoint reported fp16, transition null, and zero TQ topology | VERIFIED-LIVE |
| Long rotating-window sentinel | Cross the 1,024-token SWA window, reuse prefix/L2, and reproduce exact early/middle/tail facts without a loop or truncated tail | OPEN |
| Ten-turn coherence | Ten visible turns with cache reuse, measured TTFT/tok/s on every generated turn, no marker leakage, no hidden-only answer, no looping | OPEN |
| Tool/parser continuation | Required tool, auto tool, no tool, tool-result continuation, tool error recovery, and post-tool text turn with TQ off/on | OPEN |
| Reasoning/template state | UI Thinking off/on/auto maps to the emitted reasoning/content channels and model-owned generation config; no prompt or sampler masking | OPEN |
| API parity | Chat Completions stream/non-stream, Responses stream/non-stream, Anthropic, and Ollama reconstruct the same visible content and finish reason | OPEN |
| Stop/cancel cleanup | Cancel during prefill and decode; next warm turn neither restores a poisoned partial boundary nor leaks stale output | OPEN |
| Memory settings next-load semantics | Strict custom 10% refusal and No Automatic Limits success are live-proven on the same 31B bundle; Safe Auto restored. Performance/Balanced modes remain open | PARTIAL |
| Low-RAM physical footprint | Activity Monitor main Memory must remain below full bundle size during load/generation | FAILED-LIVE on 31B: 28.37 GB versus 25G bundle |
| Delegation/admission | Local text subagent, Computer Use/AppleScript, image generation/edit, and concurrent delegation receive the same memory admission decision before model eviction | OPEN; Osaurus-level row |

## Non-Gemma rows retained for the wider campaign

- Hybrid SSM/GDN/GLA families (Qwen 3.5/3.5 VL, Ornith, Bonsai, Nemotron and
  applicable MiniMax variants) require typed companion-state hit/miss/rederive
  counters, selective TQ conversion of full-attention KV only, native
  SSM/GDN/CCA companion-state preservation, partial-hit rollback/rederive
  proof, media-salt proof for VL, and coherent post-hit continuation with
  TurboQuant off/on. Detached async rederive is not accepted as a production
  substitute for prompt-boundary synchronization.
- LFM and MiniMax M2.7 also remain wider-campaign rows for explicit
  TurboQuant off/on, partial SSD-only restore, paged eviction where their
  topology supports it, multi-turn coherence, TTFT/tok/s, and physical
  footprint. Gemma evidence does not close those rows.
- DSV4 Flash and OpenPangu must remain hard-excluded from TurboQuant KV even
  under explicit user opt-in. DSV4/ZAYA and MiniMax-M3 native composite caches
  remain explicit exceptions until their typed native codecs are separately
  source- and live-proven. Their evidence must not be generalized from Gemma.
- JANGTQ/MXTQ weight-format correctness is a separate issue from TurboQuant KV
  cache encoding and must remain a separate matrix.
- AppleScript 8B JANG_6M import/discovery, Computer Use app switching, success
  finalization, unexpected tool fallback, and spawned-agent plugin inheritance
  remain Osaurus UI/runtime rows. They are not closed by this cache checkpoint.

No row becomes release-ready from source inspection, a configured setting, or
an aggregate counter alone.

## Current-source test note

With the Xcode 26 toolchain selected, the current patch's focused run passed
8/8 assertions: 3 transition telemetry rows, the MiMo selective-conversion
source contract, both paged-incompatible coordinator rows, Engine Selected
keeping TQ off, and explicit SSD-Off preservation through memory-safety
resolution. The
older `automaticRuntimeCachePolicyCoversDownloadedArchitectureFamilies`
matrix still contains a separate stale Hunyuan reasoning expectation
(`think_xml` while the current source returns `hy_v3`). That unrelated row is
not changed or counted as Gemma cache proof in this checkpoint.
