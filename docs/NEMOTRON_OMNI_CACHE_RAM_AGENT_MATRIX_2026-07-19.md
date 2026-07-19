# Nemotron Omni cache, RAM, and agent matrix — 2026-07-19

Scope: the exact local bundle
`/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ4-CRACK`, vMLX Swift,
and the real Osaurus Release app. JANGTQ model weights and TurboQuant KV cache
encoding are separate features throughout this matrix.

This document is the live checklist. A row is `VERIFIED-LIVE` only when the
current-source isolated Release app has visible output plus telemetry or a
persisted artifact that proves the claimed tier. Source inspection and unit
tests alone are not live proof. The `nemotron4634proof` rows below are baseline
evidence for merged main before this cache patch, not current-patch closure.

Current clean-branch source gates:

- `CacheCoordinatorTopologyFocusedTests`: 39/39 passed; log
  `/tmp/vmlx_nemotron_clean_cache_topology_stateonly_20260719.log`.
- `CacheCoordinatorPagedIncompatibleHybridTests`: 12/12 passed; log
  `/tmp/vmlx_nemotron_clean_incompatible_failclosed_20260719.log`.
- `BatchEngineGrowingChatCacheSourceTests`: 16/16 passed; log
  `/tmp/vmlx_nemotron_clean_source_final_20260719.log`.
- `git diff --check`: clean after the focused source fixes. These results do
  not upgrade any `LIVE-OPEN` row.

## Required cache topology

- Default: prefix cache ON, paged RAM cache OFF, disk L2 ON, TurboQuant KV OFF.
- TurboQuant opt-in: encode only the six attention-KV layers. Preserve all 23
  Mamba companion layers in their native state and restore them at the exact
  matched boundary.
- Paged opt-in: try paged RAM blocks first. A usable hybrid hit requires both
  attention KV blocks and a complete native SSM companion snapshot at the same
  boundary.
- Paged miss/eviction: fall through to the longest valid disk-L2 boundary.
  Missing or incomplete companion state must reject the warm path and safely
  prefill; it must never restore attention KV alone.
- Paged OFF: disk L2 must still find the longest stored prompt/history boundary,
  restore it after unload/restart, and prefill only the unmatched suffix.
- Media: cache keys must include media bytes and effective reasoning/KV policy;
  same media may reuse, different media must miss.

## Current matrix

| Row | Required evidence | Status | Current evidence / blocker |
|---|---|---|---|
| Default cache settings | Real UI shows prefix ON, paged OFF, disk L2 ON, TQ OFF; fresh restart preserves values | BASELINE-LIVE / CURRENT-PARTIAL | Baseline isolated Release app `com.dinoki.osaurus.nemotron4634proof` showed FP16, paged 0, disk/SSM hits 4. Current cache patch still needs the same UI row. |
| Selective TQ topology | Real load + generation shows 6 TQ-KV and 23 native Mamba layers; TQ counters advance only on attention slots | BASELINE-LIVE / CURRENT-PARTIAL | Baseline UI TQ 4/4 run reported transition 6/23, converted 6, compressions 3, disk hits 4, SSM companion hits 4. Current patch focused test independently constructs the exact 29-layer 6/23 topology. |
| Default disk-L2 restart | Quit/restart with paged OFF; no media reattachment; coherent recall and disk+SSM hits | BASELINE-LIVE / CURRENT-PARTIAL | Baseline no-attachment image follow-up after restart: TTFT 1.17 s, 102.6 tok/s, disk L2 hits 4, SSM hits 4. Current patch disk-only partial-prefix unit row passes; UI rerun pending. |
| Paged ON is honored | UI toggle ON; effective stats show paged enabled and a real paged hit | SOURCE-TESTED / LIVE-OPEN | Broad disk-backed classifier was split from paged incompatibility. Focused exact-topology row proves a paged hit; current Release UI proof is missing. |
| Paged partial-prefix hit | Growing/diverging prompt hits full blocks, restores six attention layers at N, restores complete SSM state at N, and prefills suffix | SOURCE-TESTED / LIVE-OPEN | Focused row restores 8 tokens into 6 compressed TQ attention slots and 23 Mamba slots at offset 8, leaving a 3-token suffix. Exact-model visible coherence/telemetry is pending. |
| Paged eviction to disk fallback | Force paged eviction; paged miss/eviction counter rises; same prefix then hits disk L2 with coherent output | SOURCE-TESTED / LIVE-OPEN | Tiny 3-block focused row records at least two evictions and restores the earlier 8-token prefix from disk with companion state. UI telemetry/coherence pending. |
| Paged miss with missing SSM | KV block exists but companion is absent/incomplete; paged block is released; disk is tried; otherwise clean prefill | SOURCE-TESTED / LIVE-OPEN | Focused missing and partial companion rows reject the paged hit; current exact-model UI row pending. |
| Unknown/state-only topology fail-closed | An unrecognized or recurrent-only cache never publishes a token-only paged hit; typed disk/prefill remains reachable | SOURCE-TESTED / LIVE-OPEN | Explicit unknown-cache allowlist test fails closed; state-only Mamba row allocates zero paged blocks and restores the valid disk+companion boundary. Current UI proof is pending. |
| Disk-only partial-prefix reuse | Paged OFF; growing or reasoning-divergent turn hits the longest stored L2 boundary and prefills only suffix | SOURCE-TESTED / LIVE-OPEN | Fresh-coordinator focused row restores the longest 8-token disk+companion prefix and leaves a 3-token suffix. UI restart proof pending. |
| TQ disk restart | TQ ON; restart; disk payload restores encoded attention KV and native Mamba state; output coherent | BASELINE-LIVE / CURRENT-PARTIAL | Baseline 6 TQ + 23 Mamba run had disk/SSM hits 4 and a coherent mixed-media follow-up. Current patch UI rerun pending. |
| TQ OFF restoration | Engine-selected/FP16 restart and coherent media/text follow-up | BASELINE-LIVE / CURRENT-PARTIAL | Baseline video A/B and restart rows completed with effective FP16 and TQ layers 0. Current patch UI rerun pending. |
| Same/different media isolation | Same media reuses; different image/audio/video bytes miss; post-media text remains coherent | PARTIAL | Baseline same-media restart/follow-up and mixed media passed. Explicit different-media miss counter and current patch rerun remain open. |
| Reasoning OFF/ON/transition | Real toolbar toggles; clean visible content; no hidden-only turn, marker leak, loop, or forced prompt/sampler | PARTIAL | Baseline fresh OFF and fresh ON chats passed. One OFF-to-ON transition repeated a correct line three times; current patch still needs adversarial rerun. |
| Tools | No-tool acknowledgement, auto tool, required tool, valid args, tool-result continuation, failed-tool honesty, loop stop | OPEN | One baseline audio attempt made an unsolicited valid `share_artifact` call; full matrix not yet run on this bundle. |
| Content/delta streaming | Visible reasoning/content deltas, terminal finish, no truncation or raw protocol markers across text/media/tools | OPEN | Basic baseline live UI output passed; structured stream/tool matrix is not complete. |
| RAM footprint | Activity Monitor / `phys_footprint` during load, prefill, decode, cache store, restart | BASELINE-LIVE / CURRENT-PARTIAL | Baseline current 17.6–17.9 GB, peak 20.3 GB. Current paged/TQ cache growth, eviction, and concurrent-delegation peaks remain open. |
| RAM warning override | Known-size oversized model triggers warning; user changes the real setting and load proceeds/blocks according to that value | OPEN | External bundle has unknown catalog size, so the warning never appeared in the baseline run. |
| Spawn/delegation RAM preflight | Real settings enable delegation; selected Nemotron target resolves; insufficient-RAM notification is truthful; accepted run completes | OPEN | No current live proof. |
| Model swap/unload | Unload drops volatile paged/SSM state but preserves disk L2; reload restores without stale cross-model state | PARTIAL | Baseline restart persistence passed; repeated model-swap and isolation rows remain open. |

## Immediate implementation and proof order

1. Keep “needs typed disk serialization” separate from “cannot use paged
   attention KV plus companion state.” DSV4, ZAYA CCA, rotating/SWA, affine,
   and other unproven cache types remain paged-incompatible; only Mamba/Arrays
   hybrid plus plain/TurboQuant attention slots use the new route.
2. Run focused tests for hybrid paged partial hits, layer ordering, incomplete
   companion rejection, paged eviction followed by disk hit, TQ restoration,
   media salt isolation, and disk-only longest-boundary reuse.
3. Rebuild an isolated Release Osaurus app pinned to the exact engine commit.
4. Drive default, paged ON, tiny-pool eviction, paged OFF, TQ ON/OFF, restart,
   reasoning, tool, RAM-warning, model-swap, and delegation rows through the UI.
5. Record `phys_footprint`, TTFT, token/s, cache tier/counters, effective cache
   topology, visible answer/reasoning, and exact artifact path for every row.

## Non-goals and safety boundaries

- Do not enable TurboQuant or paged cache by default.
- Do not apply generic TQ to DSV4 Flash, OpenPangu, ZAYA CCA, DSV4 composite
  pools, Mamba/SSM/GDN companion state, or rotating/SWA state.
- Do not add forced thinking tags, prompt coercion, sampler changes, token
  biases, synthetic closers, or length caps to make a row look coherent.
- A coherent response without tier telemetry is not cache proof; a cache hit
  without visible coherent multi-turn output is not correctness proof.
