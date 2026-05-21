# MLXPress Attention Architecture Ledger

Last updated: 2026-05-13

This is the running implementation ledger for making the combined
`vmlx-swift` repo a low-Activity-Monitor-footprint MLX/JANG/JANGTQ inference
runtime. It is intentionally concrete: every family needs its own cache,
decode, parser, and memory proof before it is called production-ready.

## Global Production Gate

Every model-family row must eventually prove all of the following:

- Load with MLXPress enabled and Activity Monitor physical footprint below the
  target gate, currently 30% of model safetensors bytes unless a stricter
  family gate is documented.
- Cache stack enabled: paged KV, disk L2, TurboQuant KV when supported, and
  any family companion state required for correctness.
- Cold and warm cache rows with token/s recorded for prompt and decode.
- Decode token/s must improve over the comparable non-MLXPress or previous
  baseline path for the same bundle; a low-RAM row that is slower is only a
  diagnostic failure mode.
- Multi-turn generation in one loaded session, not separate single-turn loads.
- Coherent visible output per turn, no visible loop, no reasoning loop, no
  max-token stop for no-loop rows, and enough generated tokens to make the
  test meaningful.
- Reasoning parser and tool parser selected by family capability, not by
  caller-side string hacks.
- Base MLXPress decode stays plain autoregressive. MTP/speculative decode must
  remain off unless a family-specific implementation proves parity.
- Any cache hit that restores path-dependent state must include the matching
  non-KV companion state or be rejected.
- Routed JANGTQ bundles default to compression-first canonical mmap residency
  under MLXPress. The normal path keeps routed weights logically loaded,
  advises inactive expert pages cold, and relies on macOS reclaim/compression
  rather than rereading active slices from SSD every token. Permanent
  prestacked safetensors overlays remain explicit opt-in only with
  `MLXPRESS_PRESTACK=1` or `JANGPRESS_PRESTACK=1`.
- Active-streaming slice reads are fallback diagnostics. On Darwin the fallback
  reader uses a size-aware cached-`pread` policy: normal OS cache for routed
  byte sets that fit the RAM warming budget, `F_NOCACHE` for Kimi-scale
  oversized byte sets. High file/page pressure means the row has not proven the
  intended MLXPress compression method.
- Layer-level mmap cold sweep exists as a diagnostic ABI
  (`mlx_safetensors_mmap_advise_layer`, `MLXPressMmapColdSweep`), but it is
  not default-enabled. The Kimi short-answer layer-sweep row did not reduce
  the 43.24 GB peak, so the next fix must look beyond simple per-layer
  `madvise(DONTNEED)`.
- Readiness rows explicitly track attention architecture, Hadamard/TurboQuant
  matmul, RoPE/MRoPE, cache block storage/encode, hybrid companion-state split,
  and VL/vector media cache identity. Do not hide those under a generic
  "cache works" note.
- Inspect rows now expose separate machine-readable arrays for
  `matmulKinds`, `positionVectorKinds`, `cacheEncodingKinds`, and
  `hybridSplitKinds`. Use those to distinguish JANGTQ Hadamard/gather matmul,
  scalar RoPE offsets, Qwen VL 2D/3D grid vectors, media-salted cache keys, and
  path-dependent non-KV companion state before running a heavy proof row.
- RoPE values can live directly on the text config or inside
  `rope_parameters` / `rope_scaling`. Inspect must account for both locations,
  otherwise Qwen3.6 and DSV4 can look like they have generic RoPE when they
  actually carry MRoPE sections, interleaved MRoPE, or dual/topology-specific
  RoPE parameters.
- A non-empty `vision_config` is a multimodal cache contract even if the nested
  model type is a generic string such as `qwen3_5`, unless JANG metadata
  explicitly says `has_vision=false`. Real multimodal bundles must surface VL
  attention, Qwen 2D/3D MRoPE position vectors, media salt, and media
  partial-hit rejection before any cache-stack row can be trusted.

## Shared Surfaces To Check

- Loading and memory policy:
  `MLXPressLoadConfiguration`, `LoadConfiguration`, `JangPressPrestacker`,
  `JangPressMmapTier`, `JangPressCanonicalExpertAdvisor`, `Load.swift`,
  `JangLoader`, `LLMModelFactory`, `VLMModelFactory`.
- JANGTQ decode:
  `JANGTQRuntimeCache`, `JANGTQKernels`, `TurboQuantSwitchLinear`,
  `TurboQuantSwitchGLU`, `JANGTQStreamingExperts`,
  `StreamingTurboQuantSwitchGLU`.
- Cache stack:
  `CacheCoordinatorConfig`, `CacheCoordinator`, `PagedCacheManager`,
  `DiskCache`, `TQDiskSerializer`, `CompilableTurboQuantKVCache`,
  `KVQuantizationMode`, `SSMStateCache`, `SSMCompanionDiskStore`,
  `SSMReDerive`.
- Position/media/cache identity:
  `RoPEApplication`, Qwen VL `RotaryEmbedding`, `CacheBlock`,
  `mediaSalt`, `cacheScopeSalt`, Qwen VL `getRopeIndex`, image/video `THW`
  grids, and MRoPE delta vectors.
- Generation and channels:
  `GenerateParameters`, `TokenIterator`, `BatchEngine`, `Evaluate.generate`,
  `ReasoningParser`, `ToolCallFormat`, `ToolCallProcessor`,
  `Generation.reasoning`, `Generation.chunk`, `Generation.toolCall`.
- Streaming performance:
  `MLXPressStreamingProfile`, `StreamingTurboQuantSwitchGLU.reduced`,
  `JANGTQStreamingExpertStore.load`, stacked safetensor byte offsets,
  `MLXPRESS_STREAMING_REDUCE_TOKEN_CHUNK_SIZE`,
  `MLXPRESS_STREAMING_FAST_REDUCE_MAX_TOKENS`, and
  `MLXPRESS_STREAMING_EXPERT_CACHE_MB`,
  `MLXPRESS_STREAMING_BANK_LOAD`, `MLXPRESS_STREAMING_PROFILE_TOP`,
  `MLXPRESS_STREAMING_EVAL_EACH_LAYER`,
  `MLXPRESS_STREAMING_EVAL_DECODER_LAYER`,
  `MLXPressActiveExpertTrace`, `MLXPressActiveSliceResidency`,
  `active_slice_residency`, `mlxpress --metrics-jsonl`,
  `scripts/summarize-mlxpress-metrics.py`.
- Validation:
  `Sources/MLXPressCLI/main.swift`, `scripts/validate-models.sh`,
  `scripts/compare-cache-deviation.sh`, `docs/local/model-validation/*`,
  `docs/local/deviation/*`. New `--metrics-jsonl` rows include
  `system_pressure` records, so every family can distinguish model-file reads
  from OS swap/page-in pressure. `--file-read-gate-mb-per-token` adds a hard
  gate for active tensor file reads per generated token.
- Active expert scheduler plan:
  `docs/MLXPRESS_ACTIVE_EXPERT_SCHEDULER_PLAN.md` tracks the concrete helper
  surfaces still needed for speed: active expert trace, budgeted active-slice
  residency, routing-aware prefetch, direct stacked-offset JANGTQ kernels,
  family policy, and runtime pressure watcher.
- Readiness checklist:
  `MLXPressModelReadinessChecklist`, `MLXPressArchitectureFacts`, and
  `docs/MLXPRESS_MODEL_READINESS_CHECKLIST.md`. Inspect every bundle with
  `.build/debug/mlxpress <model-dir> --inspect --json` before heavy generation
  so the family, attention/cache architecture, config-derived attention kinds,
  Hadamard/matmul path, RoPE/MRoPE signals, position-vector requirements,
  cache storage/encode requirements, companion-state split, media-cache
  identity, load method, required proofs, and blockers are explicit.

## Architecture Axes That Must Not Collapse Together

- Hadamard/TurboQuant matmul: JANGTQ decode is not one generic matmul. The
  active path uses randomized Hadamard rotation, power-of-two block
  decomposition for non-power-of-two dimensions, runtime sidecar signs and
  `codebook.{N}.{bits}`, `tq_packed` / `tq_norms` gathers, active expert index
  remapping, and fused gate/up SwiGLU where the family uses GLU. For blocks
  <=1024, Swift now has the SIMD-shuffle Hadamard path used by Python; MiniMax's
  1536-dim intermediate path is the canonical `1024 + 512` case to remeasure
  after every kernel change with real decode token/s.
- Kimi routed reuse is its own architecture axis. The 32 GB active-window cache
  row on `Kimi-K2.6-JANGTQ_K` produced coherent text and stayed below 1% peak
  Activity Monitor footprint, but touched most routed experts in only 30
  generated tokens: 25,440 routed slots, 21,196 unique expert touches, 0.263
  reuse rate, and 60,895 span-cache evictions. For Kimi, file-backed mmap LRU
  caching is not equivalent to the intended MLXPress method because clean
  file-backed pages can be discarded and refaulted from disk. The production
  path must be anonymous/purgeable resident storage or persistent
  offset-addressed kernels, plus MLA absorb-decode for the attention side.
  `MLXPRESS_STREAMING_MACH_OFFSET_SPANS=1` is the current diagnostic bridge:
  it only registers exact one-expert active offset windows into Mach purgeable
  anonymous storage and records `tensor.mach_offset_*` rows, so it can test
  native compression without counting broader file-backed LRU behavior as a
  Kimi win.
- RoPE/MRoPE vectors: text rows need scalar position IDs and cache offsets;
  DeepSeek/Kimi MLA rows additionally need the partial-RoPE head vector; Qwen
  MRoPE rows need the 3-axis head split; Qwen VL rows need
  `[3,batch,seq]` position IDs, image `THW`, video `THW`, and MRoPE delta
  vectors.
- VL cache identity: raw media bytes, shape, dtype, processor grid/scope, and
  reasoning/template state are hashed into `mediaSalt` / `cacheScopeSalt`.
  Partial hits that split a media-token region must roll back to full prefill.
- Cache block storage/encode: paged blocks use the parent-hash chain over
  model key, media salt, parent hash, and raw token IDs; disk L2 uses a
  separate token-hash domain; `TQDiskSerializer` v2 tags each layer as KV, TQ,
  QKV, Mamba, rotating, DSV4, CacheList, Zaya CCA, or skip.
- Hybrid split: path-dependent non-KV state is not optional. Mamba/Arrays,
  ZAYA CCA `conv_state` / `prev_hs`, and DSV4 compressor/indexer pools and
  incomplete-window buffers must serialize/restore with the hit or force a
  miss/re-prefill.

## Per-Family Build Matrix

This matrix is the running scratchpad for what must be built or verified for
each attention/cache architecture before a family can be called MLXPress-ready.

| Family | Attention / position contract | Decode / matmul contract | Cache-stack contract | Parser / tool contract | Async / warm / deviation contract |
| --- | --- | --- | --- | --- | --- |
| Kimi K2.x | MLA-style no-PE + partial-RoPE split, `qk_rope_head_dim=64`, full KV heads, YaRN scaling; current path lacks MLA absorb decode. | Default target is compression-first canonical mmap residency. Active-streaming stacked `switch_mlp` slice-by-offset rows are low-RAM fallback diagnostics, but decode speed still needs resident-page warm/cold proof and/or direct stacked-offset dispatch that avoids per-token bank rebuilds. | Paged + disk L2 + TurboQuant KV; prompt boundary raw KV; no SSM unless future config adds it. | Kimi thinking aliases (`thinking` and `enable_thinking`) plus Kimi K2 tool parser; thinking-on currently leaks reasoning-style visible text. | No-thinking stacked multi-turn rows pass below Activity Monitor gate with token/s only on the streaming fallback; profile rows show 130.55 GB active slice reads for six visible tokens, so compression-first rows, cold/warm rows, reasoning-on closure, tool calls, and faster decode remain open. |
| MiniMax M2.7 | Dense attention RoPE with routed MoE; preserve KV position continuity across disk hits. | Mixed JANGTQ_K gate/up 2-bit and down 4-bit; router precision must stay stable. | Paged + disk L2 + TurboQuant KV already has thinking-off rows; thinking-on/tool still gated. | MiniMax XML think/tool handling; final visible answer closure required. | Compiled SwitchGLU remains denied until parity; cache deviation rows must include thinking-on final answer, not reasoning-only pass. |
| Hy3 / Hunyuan | Dense RoPE from `rope_parameters`; MTP weights exist but base decode must stay non-speculative. | Uniform JANGTQ row has proof; K mixed-bit row must stream active experts without full expert stack. | KV heads 8; JANGTQ_K decode still crosses RAM gate; prestack overlay is unacceptable. | Hunyuan reasoning/tool parser rows required. | Recheck router indices, score dtype, mixed-bit codebooks, and chunk graph lifetime before more materialization tweaks. |
| Qwen3.6 MoE | Qwen hybrid/MRoPE fields live under `rope_parameters`, including MRoPE section/interleaving and linear attention layers. | Uniform JANGTQ2-style routed experts; default target is compression-first canonical mmap residency, with active streaming only as an explicit fallback diagnostic. | Hybrid/linear recurrence state must be serialized, rederived, or hit must be rejected. | Qwen reasoning/tool parser must keep `.reasoning`, `.chunk`, and tool calls distinct. | SSM/linear companion audit and cold/warm deviation proof are mandatory before full disk hits. |
| Qwen VL | Text RoPE plus Qwen `[3,batch,seq]` MRoPE, image `THW`, video `THW`, and delta vectors. | VLM encoder plus text decoder; text-only and media rows must not share unsafe cache identity. | Media bytes/shape/dtype and processor scope salt; partial hits that split media token region must full-prefill. | Qwen reasoning/tool parser plus media template behavior. | Need real image/video cold/warm rows proving same-media hits and different-media misses with token/s. |
| ZAYA text/VL | Text RoPE plus ZAYA CCA offset/state; VL rows use Qwen2.5-style image geometry. | ZAYA split routed JANGTQ names must use compression-first canonical mmap residency by default; active expert streaming is fallback only. | `ZayaCCACache` must persist/reject KV + `conv_state` + `prev_hs`; VL salt required. | ZAYA reasoning-on and tool rows not fully proven; VL media proof missing. | CCA companion cache hit/reject proof plus real media cold/warm rows. |
| Ling / Bailing | Bailing hybrid attention/MoE; determine exact recurrent/linear state per bundle. | JANGTQ2 and MXFP4 variants both need low-RAM proof. | Mamba/Arrays/linear recurrence state must go through SSMStateCache/rederive or reject. | Ling/Bailing parser and tool rows need current combined-repo proof. | Current low-RAM multi-turn proof must be rerun; disk/full hit unsafe until companion audit passes. |
| DSV4 Flash | MLA partial RoPE + sliding window + compressor/indexer topology. | JANGTQ routed experts plus DSV4 topology-specific attention. | `HybridPoolCache` rotating KV, compressor/indexer pools, and incomplete-window buffers via TQDiskSerializer. | DSV4 parser rows are separate from Kimi/MiniMax/Hy3; do not generalize. | Deviation proof must cover compressor/indexer cache keys and pool restore; DSV4 evidence cannot certify other families. |

Cross-family validation rows must always include Activity Monitor RAM, prompt
and decode token/s, raw stdout, no-loop verdict, cache hit/miss tier, active
model-file read pressure when streaming JANGTQ is enabled, and the exact
parser/cache settings used for that turn.

## MiniMax M2.7

Local bundles:

- `~/models/JANGQ/MiniMax-M2.7-Small-JANGTQ`
- `~/models/dealign.ai/MiniMax-M2.7-JANGTQ_K-CRACK`

Live config for CRACK K:

- `model_type=minimax_m2`
- hidden 3072, layers 62, heads 48, KV heads 8
- local experts 256, top-k 8
- `weight_format=mxtq`
- mixed JANGTQ_K routed bits: gate/up 2-bit, down 4-bit; attention/shared
  expert/embed/lm head 8-bit; norms/router/biases 16-bit

Implemented or currently proven:

- Low-RAM MLXPress path works for thinking-off long checklist rows.
- Cold/warm disk L2 and TurboQuant KV rows record token/s and pass strict
  visible no-loop checks for thinking-off validation.
- The chat-template bridge forces the corrected `MiniMaxM2Minimal` fallback for
  `enable_thinking=false`, because the native template always opens `<think>`.
- Current combined-repo split:
  - Active-streaming OS-cache fallback row
    `docs/local/model-validation/20260513T061054Z-minimax-small-streaming-oscache-hadamard/`
    is coherent and low-RAM (peak 2.53 GB / 6.8%) but slow (1.82 tok/s avg)
    and fails read pressure (5,913 MB/generated-token).
  - Resident routed-tensor row
    `docs/local/model-validation/20260513T061253Z-minimax-small-resident-speed-hadamard/`
    is coherent and has zero explicit streaming reads, but fails low RAM
    (35.37 GB / 95.5%) and only averages 3.72 tok/s.
  - Load-time Mach-backed full projection stacks are rejected:
    `docs/local/model-validation/20260513T164210Z-minimax-small-loadtime-mach-stacks-resident-one-turn/`
    is coherent and reaches 7.99 tok/s, but Activity Monitor still climbs to
    35.34 GB / 95.4% after GPU touch.
  - Ephemeral mmap-prestack is the current strongest resident-compute
    diagnostic:
    `docs/local/model-validation/20260513T170902Z-minimax-small-ephemeral-prestack-mmap-nocold-2turn/`
    passes coherent two-turn output with cache stack and TurboQuant KV on,
    removes its temporary 32.8 GB overlay at exit, keeps peak Activity Monitor
    to 2.62 GB / 7.1%, passes effective read pressure at 662.31
    MB/generated-token, and decodes at 13.89 / 14.49 tok/s. It is still
    blocked by explicit temporary overlay dependence and remains below the old
    39-47 tok/s JangPress target. The recipe is now exposed as
    `--ephemeral-prestack on` for repeatable MiniMax diagnostics; it is not the
    final no-overlay resident-kernel design.
  - Strict CLI flag proof
    `docs/local/model-validation/20260513T174200Z-minimax-small-cli-ephemeral-prestack-nocold-strict-2turn/`
    passed coherent two-turn output with `--fail-on-length-stop`, normal
    `stop=stop` endings, cache stack and TurboQuant KV on, peak Activity Monitor
    2.63 GB / 7.093%, 0 MB explicit reads, 780.76 MB/generated-token effective
    read pressure, and 13.162 aggregate tok/s. This proves the new option path
    but not the production target speed.
  - Longer sustained rows are better than the short two-turn row:
    `docs/local/model-validation/20260513T171708Z-minimax-small-ephemeral-prestack-mmap-nocold-tqkv-long/`
    produces a coherent 129-token paragraph with TurboQuant KV at 19.70 tok/s,
    peak Activity Monitor 2.59 GB / 7.0%, and effective read pressure 236.15
    MB/generated-token. The matching KV-none diagnostic
    `docs/local/model-validation/20260513T171915Z-minimax-small-ephemeral-prestack-mmap-nocold-kvnone-long/`
    reaches 21.61 tok/s, so TurboQuant KV is not the whole sustained-speed
    blocker.
  - Profiled sustained row
    `docs/local/model-validation/20260513T172505Z-minimax-small-ephemeral-prestack-mmap-nocold-tqkv-long-profile/`
    shows `decode.async_eval_submit` dominates at 47.396 ms average, while
    `decode.kv_quantize` averages 2.733 ms. The remaining MiniMax work is
    resident graph/eval or compiled/persistent-kernel behavior, not just KV
    encoding.
  - Pre-fix compiled MiniMax decode remains rejected for correctness. Row
    `docs/local/model-validation/20260513T175200Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-profile-strict-2turn/`
    forced `--compiled-decode on` with `--allow-minimax-compiled-decode`,
    `--compiled-max-cache-length 512`, `--kv-cache none`, and the no-cold
    `--ephemeral-prestack on` path. It reached 24.56 / 24.64 tok/s and kept
    peak Activity Monitor to 2.58 GB / 6.943%, but both turns visibly looped
    and length-stopped. This is a profiling clue, not a readiness row.
  - Single-sequence compiled decode now promotes `KVCacheSimple` to
    `CompilableKVCache`, matching the BatchEngine path. Follow-up row
    `docs/local/model-validation/20260513T180400Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-promotedcache-strict-2turn/`
    passes strict coherency with `--kv-cache none`, `--ephemeral-prestack on`,
    and the same compiled override: 26.29 / 26.26 tok/s, both turns
    `stop=stop`, peak Activity Monitor 2.64 GB / 7.109%, zero explicit reads,
    and effective read pressure 845.69 MB/generated-token. It is still
    diagnostic because TurboQuant KV/cache-stack parity and no-overlay resident
    storage remain open.
  - Longer promoted-cache guard row
    `docs/local/model-validation/20260513T180900Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-promotedcache-longer-2turn/`
    generated 107 coherent visible tokens at 24.15 / 25.62 tok/s with no
    loops, both turns `stop=stop`, peak Activity Monitor 2.79 GB / 7.510%,
    zero explicit reads, and 284.56 MB/generated-token effective read pressure.
    Use this as the current longer no-loop diagnostic baseline.
  - Single-sequence compiled decode now also supports TurboQuant KV by
    promoting compressed `TurboQuantKVCache` layers to
    `CompilableTurboQuantKVCache`. Strict row
    `docs/local/model-validation/20260513T181400Z-minimax-small-cli-ephemeral-prestack-compiled-tqkv-promotedcache-strict-2turn/`
    passed with `default-kv=turboQuant(k=3,v=3)`, 26.95 / 24.11 tok/s, peak
    Activity Monitor 2.53 GB / 6.818%, zero explicit reads, and 845.65
    MB/generated-token effective read pressure. Longer row
    `docs/local/model-validation/20260513T181800Z-minimax-small-cli-ephemeral-prestack-compiled-tqkv-promotedcache-longer-2turn/`
    generated 169 coherent visible tokens at 22.03 / 23.13 tok/s, peak
    Activity Monitor 2.63 GB / 7.089%, zero explicit reads, and 180.10
    MB/generated-token effective read pressure. This row is now superseded for
    speed diagnosis: it was slowed by the CLI coupling `--activity-gate` to a
    tight `MLX.Memory.memoryLimit`.
  - Activity Monitor gating is now measurement/enforcement only; it no longer
    rewrites the MLX memory budget. Strict cache-stack row
    `docs/local/model-validation/20260513T192000Z-minimax-small-mlxpress-cache-stack-compiled-tqkv-activitygate-unthrottled-2turn-stop/`
    passes with TurboQuant KV, disk L2, compiled decode, `--fail-on-length-stop`,
    and `--activity-gate 59`: 41 and 47 generated tokens, both `stop=stop`,
    coherent no-loop visible text, 49.09 / 48.66 tok/s, peak Activity Monitor
    2.65 GB / 7.159%, zero explicit reads, and 0.09 MB/generated-token
    effective read pressure.
  - Therefore MiniMax-Small now reaches the old resident JangPress speed/RAM
    target in the combined repo under the explicit `--ephemeral-prestack on`
    diagnostic. It is still not the final production method because the
    temporary overlay must be replaced by the no-permanent-file resident
    mmap/offset kernel path.
- MiniMax compiled SwitchGLU decode remains opt-in until compiled vs
  uncompiled parity is broadened to thinking-on, tool-call, and no-overlay
  resident rows.

Open or blocked:

- Thinking-on short/medium rows produce coherent non-looping reasoning but may
  not close into a visible final answer before the token cap. This is not a
  production pass until a long enough cap, template setting, or close-bias
  strategy produces visible final output without loop or forced corruption.
- Reasoning-close bias exists but automatic MiniMax handling does not force the
  close token. Any force mode must be an explicit diagnostic or have a real
  parity proof.
- Tool-call rows still need MLXPress-on multi-turn proof with MiniMax XML
  tool-call syntax.

Must check while building:

- `enable_thinking=false` must route all output through `.chunk`, not
  `.reasoning`.
- `enable_thinking=true` must preserve coherent `.reasoning`, close the
  parser, then emit final `.chunk`.
- Cache keys must include model hash, template hash, reasoning/tool mode,
  TurboQuant KV mode, and any family decode knobs.
- The JANGTQ_K sidecar must provide `codebook.3072.2` for gate/up and
  `codebook.1536.4` for down.

## Hy3 / Hunyuan

Local bundles:

- `~/models/JANGQ/Hy3-preview-JANGTQ`
- `~/models/JANGQ/Hy3-preview-JANGTQ_K`
- `~/models/Tencent/Hy3-preview`

Live config for JANGTQ_K:

- `model_type=hy_v3`
- hidden 4096, layers 80, heads 64, KV heads 8
- experts 192, top-k 8
- `weight_format=mxtq`
- mixed JANGTQ_K routed bits: gate/up 2-bit, down 4-bit
- attention/shared expert/dense FFN/MTP/embed/lm head 8-bit

Implemented or currently proven:

- Uniform `Hy3-preview-JANGTQ` has prior MLXPress cache-stack coherent rows
  with low peak footprint and token/s recorded.
- JANGTQ_K config decoding preserves mixed gate/up vs down bits.
- Active-expert streaming can skip 91,008 per-expert tensors at load time.
- Active-streaming post-load footprint is low: about 4.72 GB / 4.7% of the
  101.52 GB bundle in the current evidence.

Open or blocked:

- Hy3 K active-streaming decode still spikes above the 30% gate and one
  report-only row peaked around 74.49 GB. It also failed the `2+2?` coherence
  check by emitting `2 + 2`.
- Non-streaming prestack is not acceptable for Hy3 K. The first prestacked K
  overlay wrote 89.1 GB and post-load Activity Monitor footprint reached
  93.5 GB / 92.1% before decode.
- Hunyuan tool/reasoning parser rows still need MLXPress-on validation.

Must check while building:

- Use compression-first canonical mmap residency for K. Active-expert streaming
  is acceptable only as an explicit fallback diagnostic until the resident path
  is safe and proven.
- Avoid stacking or retaining full `[experts, out, packed]` routed tensors in
  Metal memory for K.
- Ensure `StreamingTurboQuantSwitchGLU.reduced` evaluates and clears only small
  per-token/per-slot chunks, and does not retain per-layer graph state.
- Confirm router output indices, score dtype, and mixed-bit codebooks against
  the Python/JANG reference before blaming parser or cache behavior.
- Keep MTP off for base decode even though the bundle carries MTP-weight tiers.

## Kimi K2.6

Local bundles:

- `~/models/JANGQ/Kimi-K2.6-Small-JANGTQ`
- `~/models/JANGQ/Kimi-K2.6-JANGTQ_K`
- Kimi Med is not currently present under `~/models`.
- User intends to download raw Kimi K2.6 1T later; small/med quantized bundles
  are the first target.

Live config for Small:

- top-level `model_type=kimi_k25`
- `architectures=["KimiK25ForConditionalGeneration"]`
- inner text `model_type=kimi_k2`
- hidden 7168, layers 61, heads 64, KV heads 64
- `num_experts_per_tok=8`, config `top_k=50`
- no `sliding_window` in the local config excerpt
- tokenizer exists locally: `TikTokenTokenizer`, `[BOS]`, `[EOS]`, native Kimi
  chat template with `<think>` and Kimi tool-call markers

Current prompt/decode loop:

1. `mlxpress` prepares the chat turn and `MLXPressSession.generate` forces
   plain autoregressive decode by normalizing `DraftStrategy.none`.
2. `TokenIterator.prepare` pre-fills the prompt in `prefillStepSize` windows;
   `TokenIterator.next` then performs one model forward per accepted token.
3. Each Kimi forward runs 61 decoder layers. Layer 0 is dense; the 60 routed
   layers execute attention, router, streamed JANGTQ MoE, and the dense shared
   expert MLP.
4. Streamed JANGTQ MoE reads only active stacked expert slices, but today it
   still constructs small `MLXArray` banks for every layer/top-k chunk before
   calling the Hadamard/TurboQuant kernels.

Current speed diagnosis:

- Kimi stacked JANGTQ_K is low-RAM and coherent in no-thinking mode, but still
  decode-speed blocked.
- `MLXPRESS_STREAMING_REDUCE_TOKEN_CHUNK_SIZE=4` plus
  `MLXPRESS_STREAMING_FAST_REDUCE_MAX_TOKENS=4` is the current prefill default;
  it reduced the 17-token short-prompt active chunk count from 1140 to 180
  while keeping peak Activity Monitor footprint below 2 GB.
- The six-token profile
  `docs/local/model-validation/20260513T033000Z-kimi-k-profile-6tok/`
  produced coherent text at 0.35 tok/s with peak 2.39 GB / 0.7%, but read
  130.55 GB of active stacked expert slices and materialized 55,866 active
  tensors. This is the concrete bottleneck to beat.
- The active bank-read follow-up
  `docs/local/model-validation/20260513T033702Z-kimi-k-bankload-profile-6tok/`
  changes the construction shape from many per-expert arrays plus
  `MLX.stacked` to one bank array per projection/suffix. It generated
  non-looping text at 0.39 tok/s with peak 3.15 GB / 1.0%, but still read
  127.76 GB of active stacked slices. Treat it as useful plumbing and a better
  profiling baseline, not the Kimi throughput fix.
- The metrics JSONL/bandwidth follow-up
  `docs/local/model-validation/20260513T034200Z-kimi-k-metrics-jsonl/`
  wrote machine-readable token/s, coherency, memory, and Activity Monitor gate
  records. The profile measured active bank reads at about 6,968 MB/s and
  bank-array construction at about 19,643 MB/s, which is not close to saturating
  unified memory bandwidth. That points the next optimization at routed decode
  orchestration, direct stacked-offset kernels, and graph/eval sequencing.
- The streaming-profile JSONL follow-up
  `docs/local/model-validation/20260513T034856Z-kimi-k-streaming-profile-jsonl/`
  adds the full top-20 streaming profile rows to `metrics.jsonl`, so future
  experiments can compare scheduler/kernel changes without parsing stderr.
- The no-eval-boundary follow-up
  `docs/local/model-validation/20260513T035500Z-kimi-k-no-each-layer-eval-after-envfix/`
  proves `MLXPRESS_STREAMING_EVAL_EACH_LAYER=0` now actually reaches the
  runtime when active expert streaming is enabled. Earlier no-eval attempts
  were misleading because the MLXPress session wrapper forced the variable
  back to `1`. This row reduced profile-internal time on a one-token prompt
  but left decode throughput at about 0.12 tok/s, so do not use it as a
  default-policy change or readiness claim.
- MiniMax differs materially: hidden 3072 instead of 7168, KV heads 8 instead
  of 64, simpler dense RoPE attention, an optional compiled router fast path,
  and no Kimi shared-expert add after every routed MoE block. Kimi additionally
  lacks the MLA absorb branch and direct stacked-offset JANGTQ kernels.

Functions or components still needed:

- Direct stacked-offset JANGTQ kernels that consume stacked safetensor slice
  metadata/expert offsets without building temporary active banks for every
  token.
- An active-expert scheduler/watcher that can prefetch or retain likely-hot
  layer/expert/projection slices under a strict Activity Monitor budget.
- MLA absorb decode for Kimi/DeepSeek-style partial-RoPE attention.
- Router readback minimization or compiled Kimi router parity, with the
  original score semantics preserved.
- A sustained decode benchmark row after each speed change, always recording
  prompt token/s, decode token/s, peak footprint, output coherency, and the
  streaming profile counters above.

Next-time Kimi notes:

- Do not call Kimi ready from low RAM alone. The hard gate is still low
  Activity Monitor RAM, coherent visible/reasoning output, multi-turn behavior,
  cache-stack hit correctness, and usable token/s together.
- Bank-load counters are the sanity check for the current stacked path:
  `tensor.stacked_bank_read` and `tensor.stacked_bank_array` should appear when
  `MLXPRESS_STREAMING_BANK_LOAD` is enabled. If only `tensor.stacked_read`
  appears after an edit, rebuild `--product mlxpress` explicitly before
  assuming the runtime path failed.
- Use `--metrics-jsonl` and `MLXPRESS_STREAMING_PROFILE_TOP=20` on speed rows.
  The JSONL file is the durable token/s and Activity Monitor record; the
  `streaming_profile` record and profile `bw=` fields are the first-order check
  for whether the row is bandwidth-bound or orchestration-bound.
- Use `scripts/summarize-mlxpress-metrics.py` for any before/after comparison.
  Do not judge scheduler or direct-kernel changes from only terminal scrolling;
  compare decode tok/s, peak percent, coherency, and top profile stages from
  the JSONL summaries. For Kimi, also compare `profile_read_mb` and
  `profile_read_mb_per_gen_token`; also compare
  `effective_read_mb_per_gen_token` because mmap page-ins can bypass explicit
  read counters. Low Activity Monitor RAM with high read MB per token means
  the runtime is still bounded by repeated safetensor streaming or mapped-page
  faults.
  For every family, compare `pagein_mb`, `swapin_mb`, and `swapout_mb` before
  claiming a slowdown is or is not OS-swap-related.
- When testing eval-boundary policy, confirm the env var is actually honored.
  With `MLXPRESS_STREAMING_EVAL_EACH_LAYER=0` the profile should not show
  `gateup.eval`, `down.eval`, `router.scores_eval`, `reduce.score_sum_eval`,
  or `kimi.layer_eval` in the active rows. A previous MLXPress wrapper bug
  made no-eval experiments look enabled while still forcing layer eval.
- The current Kimi bottleneck is still decode-layer routed work across
  60 MoE layers x top-k 8: active slice bytes, router readback, bank
  construction, Hadamard/TurboQuant gather, and shared expert build. A speed
  fix probably needs a direct stacked-offset kernel or scheduler, not another
  full-tensor mmap/load tweak.
- The current SSD/file-streaming diagnosis is captured in
  `docs/local/model-validation/20260513T041000Z-kimi-k-file-streaming-diagnosis/`.
  The one-token rows read 20.19 GB from active stacked expert slices while
  process peak stayed under 2 GB. That is the expected failure mode of a
  low-RAM but byte-heavy streaming path: not enough RAM is being used to cache
  the full model, but too many bytes are still being reread from model files.
- Fresh proof row
  `docs/local/model-validation/20260513T041500Z-kimi-k-system-pressure-profile/`
  confirms the distinction with system counters: 24.79 GB page-ins, 0.06 MB
  swap-in, and 0 MB swap-out. Treat Kimi's current sub-1 tok/s row as
  file-backed model streaming pressure until a later row disproves it.
- File-read gate row
  `docs/local/model-validation/20260513T042500Z-kimi-k-file-read-gate-report/`
  proves the runtime gate is active: the same low-RAM coherent Kimi row fails a
  1,000 MB/generated-token file-read gate at 20,190.94 MB/generated-token.
  Future scheduler/direct-kernel work should make this gate pass at a
  documented threshold.
- Active expert trace row
  `docs/local/model-validation/20260513T043500Z-kimi-k-active-expert-trace/`
  measured 180 calls, 1,440 routed slots, and 21.5% immediate expert reuse.
  This is now the reuse baseline for Kimi scheduler work.
- Active slice residency rows prove the cache path but not readiness:
  `docs/local/model-validation/20260513T043530Z-kimi-k-active-slice-residency-512mb/`
  had 0% hits, while
  `docs/local/model-validation/20260513T043713Z-kimi-k-active-slice-residency-8192mb/`
  had 21.5% byte hits and reduced file reads to 15,858.30 MB/generated-token.
  Decode stayed about 0.12 tok/s and peak rose to 11.35 GB, so direct
  stacked-offset dispatch remains the main Kimi speed path.
- Mach offset-span rows are the next native-compression diagnostic:
  `MLXPRESS_STREAMING_MACH_OFFSET_SPANS=1` /
  `JANGPRESS_STREAMING_MACH_OFFSET_SPANS=1` should produce
  `tensor.mach_offset_read` on first registration and `tensor.mach_offset_hit`
  on reuse. Treat it as created until a Kimi row beats the 32 GB file-backed
  active-window cache on decode token/s and effective read pressure while
  preserving coherent multi-turn output and low Activity Monitor RAM.
  First row
  `docs/local/model-validation/20260513T202041Z-kimi-k-mach-offset-spans-1tok/`
  is coherent and low-RAM but rejected: 0.368 tok/s decode, 6.02 GB / 1.834%
  peak Activity Monitor, 15,031.03 MB `tensor.mach_offset_read`, and
  24,819.72 MB/generated-token effective read pressure. It proves the bridge
  is measurable; it does not satisfy the speed/RAM/coherency trio.
  Warm row
  `docs/local/model-validation/20260513T202514Z-kimi-k-mach-offset-spans-2turn/`
  is also rejected: coherent `red` / `blue`, 20.88 GB / 6.364% peak Activity
  Monitor, but 0.209 aggregate tok/s, 60.73 GB resident Mach offset bytes, and
  30,871.54 MB/generated-token effective read pressure. Kimi needs bounded
  routing-aware residency/prefetch or persistent offset kernels, not unbounded
  Mach tile retention.
  Source follow-up: exact Mach offset spans now use direct safetensors
  `pread` into purgeable VM storage and can be capped with
  `MLXPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB` /
  `JANGPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB`. The next Kimi row must show
  whether this removes the extra copy cost and whether bounded retention helps
  without breaking the speed/RAM/coherency trio.
- False leads already tried: a large active expert cache raises peak footprint
  without improving average decode; active slice residency gets only 21.5%
  byte hits at an 8 GB budget; direct whole-tensor mmap dispatch is coherent
  but page-in bound and much slower; direct `MLXArray(Data, dtype:)` removed a
  copy but did not change throughput; layer cold-sweep did not reduce the Kimi
  small 43 GB strict-row peak; router materialization and chunk-local readback
  variants were worse and were reverted.
- Stacked tensors are allowed only as source files. Never create a permanent
  prestacked overlay or load a whole `[experts, ...]` tensor into MLX memory
  for MLXPress. The fallback path must read requested expert byte ranges, use
  the size-aware Darwin cached-`pread` policy, and release temporary banks
  between chunks.
- Kimi thinking-on remains parser/template blocked. The no-thinking rows do
  not prove reasoning mode or tool calls.
- Cache-stack proof still needs cold/warm rows with disk L2 and TurboQuant KV
  hits, plus deviation checks. The current profile rows are cold, short,
  no-thinking diagnostics.

Live config for stacked JANGTQ_K:

- top-level `model_type=kimi_k25`, inner text `model_type=kimi_k2`
- hidden 7168, layers 61, heads 64, KV heads 64
- routed layers 60, experts 384, `num_experts_per_tok=8`
- stacked `language_model.model.layers.N.mlp.switch_mlp` tensors for
  gate/up/down `tq_packed` and `tq_norms`
- mixed JANGTQ_K routed bits: gate/up 2-bit, down 4-bit
- `jangtq_runtime.safetensors` is now present for the stacked K bundle and
  contains `codebook.7168.2`, `signs.7168.42`, `codebook.2048.4`, and
  `signs.2048.42`. The deterministic missing-sidecar fallback remains
  supported, but this local Kimi K row no longer depends on it.

Implemented or currently proven:

- Factory support exists for `kimi_k25` wrapper unwrapping and
  `language_model.` key stripping in the imported library history.
- Active-expert streaming load proof exists for local Kimi small: 60 routed
  layers / 211 experts indexed, 75,960 per-expert tensors skipped during weight
  load, post-load Activity Monitor footprint about 1.23 GB / 0.9% with
  MLXPress, disk L2, and TurboQuant KV enabled.
- Default-policy proof:
  `docs/local/model-validation/20260512T222112Z-kimi-small-default-active/RESULT.md`
  selected active-expert streaming without `--active-expert-streaming on` and
  reported `stackedLayers=0`, so unstacked Kimi JANGTQ is not using the
  permanent prestack overlay as the normal path.
- No-cache active-slice read proof:
  `docs/local/model-validation/20260512T222606Z-kimi-small-default-active-nocache-read/RESULT.md`
  passes the one-token 30% Activity Monitor gate with peak footprint
  1.57 GB / 1.1%. This is default-path/load evidence only.
- Stacked Kimi JANGTQ_K low-RAM load and no-thinking multi-turn proof now
  exists in this combined repo. The loader indexes 60 stacked routed layers,
  skips 360 stacked routed tensors during generic weight load, and reads only
  requested expert slices by safetensor byte offset. Darwin slice reads use the
  size-aware cached-`pread` policy, which keeps `F_NOCACHE` by default for this
  oversized Kimi bundle.
- Sidecar-present proof rows now exist for stacked Kimi JANGTQ_K. The warm row
  `docs/local/model-validation/20260513T045340Z-kimi-k-sidecar-bankload-effective-gate-warm/`
  stays coherent and low-RAM but still fails at 20,190.94
  MB/generated-token, proving sidecar correctness is not the speed fix.
- Direct whole-tensor mmap proof
  `docs/local/model-validation/20260513T045423Z-kimi-k-direct-stacked-mmap-effective-gate/`
  is coherent and low-RAM but slower at 0.022 tok/s with 969,853.44
  MB/generated-token effective page-in pressure. Treat it as rejected
  diagnostic evidence, not a candidate production path.
- Default stacked exact two-turn row:
  `docs/local/model-validation/20260513T024727Z/` passed with cache stack
  enabled, active expert tensor cache budget 0 MB, output `red color` /
  `blue fruit`, post-load 144.9 MB, post-decode 1.80 GB, peak 3.26 GB
  / 1.0%, and avg decode telemetry 0.28 tok/s.
- Longer stacked no-thinking row:
  `docs/local/model-validation/20260513T024821Z/` passed with no-loop and
  minimum generation checks, output
  `Rain falls softly on the green grass.` /
  `The moon glows brightly in the night sky.`, post-load 145.8 MB,
  post-decode 2.81 GB, peak 4.68 GB / 1.4%, turn decode 0.33 and
  0.45 tok/s, avg 0.39 tok/s.
- `MLXPRESS_STREAMING_FAST_REDUCE_MAX_TOKENS=1` batches all top-k active
  experts for decode-sized chunks, avoiding the earlier serial top-k slot
  reduce. Cached safetensor file descriptors improved prompt prefill
  substantially, but decode remains sub-1 tok/s.
- Active slice construction now calls `MLXArray(Data, dtype:)` directly instead
  of copying streamed bytes into a typed Swift array first. Exact row
  `docs/local/model-validation/20260513T030410Z/` stayed coherent and
  low-RAM with peak 3.37 GB / 1.0% and avg decode 0.30 tok/s. This removes an
  avoidable copy but does not clear the decode-speed blocker.
- `MLXPRESS_STREAMING_EXPERT_CACHE_MB` exists as an explicit bounded
  active-slice cache knob. The default is 0 MB because an 8 GB cache experiment
  `docs/local/model-validation/20260513T024555Z/` raised peak footprint to
  11.22 GB while avg decode stayed about 0.30 tok/s.

Open or blocked:

- Kimi small now has strict coherent short no-thinking rows in this combined
  repo:
  `docs/local/model-validation/20260513T000500Z-kimi-small-base-thinking-alias-4tok/RESULT.md`
  for base active streaming and
  `docs/local/model-validation/20260513T000900Z-kimi-small-mlxpress-cache-thinking-alias-4tok/RESULT.md`
  for MLXPress/cache-stack on. Both emitted visible `4` with token/s and low
  Activity Monitor peak.
- Kimi small still has no strict coherent multi-turn row in this combined repo.
  The longer no-thinking short-answer row
  `docs/local/model-validation/20260512T222803Z-kimi-small-no-thinking-short-answer/RESULT.md`
  still fails the hard gate before stdout at 43.24 GB / 30.3%, so output
  validation is blocked by decode-memory residency.
- Treat Kimi as partial. Stacked no-thinking multi-turn is coherent and
  low-RAM, but decode is still far too slow, reasoning-on is not
  production-passing, and tool-call plus cold/warm cache hits still need proof.
  The current `mlxpress --inspect --json` readiness row marks
  `kimi-short-no-thinking-decode` as `partial`.
- Kimi thinking-on is blocked. Probe
  `docs/local/model-validation/20260513T025104Z/` stayed low RAM and produced
  coherent reasoning preview, but length-stopped at 48 generated tokens and
  leaked reasoning-style text into visible output instead of answering exactly
  `4`. Do not count it as a reasoning-ready proof.
- More aggressive `--mlxpress 80`, smaller `--prefill-step-size 8`, and
  precomputed active-expert index reuse did not reduce the strict Kimi
  short-answer peak; all still failed near 43.2 GB / 30.3%.
- Earlier malloc tracing at 512 MiB found repeated 0.59 GiB `concatenate_gpu`
  allocations through `StreamingTurboQuantSwitchGLU.callChunk` during
  DeepSeek/Kimi JANGTQ decode. After the `thinking` alias and single-expert
  `expandedDimensions` changes, the longer row still failed at 43.14 GB, but
  the new 512 MiB trace emitted no large allocation records. Continue the Kimi
  memory investigation across smaller Metal allocations, command-buffer
  lifetime, RoPE/KV update residency, and file/cache residency around
  active-streaming TQ matmul.
- A 128 MiB forced-longer trace
  `docs/local/model-validation/20260514T000210Z-kimi-small-forced-longer-malloc-trace-128m/RESULT.md`
  found the current dominant failure at router-index CPU readback:
  `StreamingTurboQuantSwitchGLU.reduced` calls
  `indicesFlat.reshaped([-1]).asArray(Int32.self)`, which realizes repeated
  `concatenate_gpu` allocations. The row also emitted repeated `&`, so this is
  both a memory and coherency blocker.
- The default router-readback materialization barrier
  `docs/local/model-validation/20260514T001355Z-kimi-small-forced-longer-router-barrier-trace-128m/RESULT.md`
  did not fix the row. Evaluating the original residual tensor first was worse
  and was reverted:
  `docs/local/model-validation/20260514T001948Z-kimi-small-forced-longer-original-x-barrier-trace-128m/RESULT.md`.
- Moving the router-index CPU readback inside the reduce chunk was also worse
  and was reverted:
  `docs/local/model-validation/20260514T002812Z-kimi-small-forced-longer-chunk-local-index-trace-128m/RESULT.md`.
  Next Kimi work should change the routing/readback strategy or streaming-MoE
  prefill graph residency rather than adding more eager eval or sliced-readback
  variants.
- Current source fix changes that residency boundary: `DeepseekV3JANGTQMoE`
  now chunks before `MoEGate`, so router eval, router-index readback,
  active-expert slice reads, TurboQuant gather, and the shared-expert MLP all
  run on small token windows. Defaults match the older Python/JangPress Kimi
  contract: 16-token chunks, `VMLX_JANGTQ_PREFILL_STEP` compatibility, and
  sync between chunks unless disabled.
- Validation row `docs/local/model-validation/20260513T012521Z/` shows
  pre-router chunking alone is not enough for longer Kimi decode: post-load
  remains low at 1.23 GB / 0.9%, but decode still fails at 43.2 GB / 30.3%
  and partial stdout is `uuualphaalp`. Treat the remaining failure as
  per-token decoder graph/Metal residency plus coherency during routed decode.
- Streaming JANGTQ decoder layers now materialize and clear layer output by
  default under `MLXPRESS_STREAMING_EVAL_DECODER_LAYER` /
  `MLXPRESS_STREAMING_EVAL_EACH_LAYER`, unless the caller explicitly sets the
  env override to `0`. The after-env-fix no-eval diagnostic proves the knob is
  respected, but it only covered a one-token coherent row. Longer strict Kimi
  rows are still needed before changing the default boundary policy.
- Diagnostic layer cold sweep
  `docs/local/model-validation/20260512T223823Z-kimi-small-no-thinking-short-answer-layer-sweep/RESULT.md`
  did not change that peak, so Kimi's current 30% miss is not fixed by simply
  advising loaded per-layer mmap ranges cold after forced layer eval.
- Report-only short-answer row
  `docs/local/model-validation/20260512T224306Z-kimi-small-no-thinking-short-answer-report-only/RESULT.md`
  continued past the memory gate but emitted no stdout or token/s telemetry
  before exit 137. Kimi output-channel/coherence testing remains blocked until
  the decode-residency issue is reduced enough for a normal strict row.
- Kimi Med needs local download or path before it can be tested.
- The large Kimi direction is the main pressure test: low RAM and faster token/s
  are the whole reason MLXPress exists here.

Must check while building:

- Factory unwrap: top-level `kimi_k25` conditional wrapper must instantiate the
  inner `kimi_k2` text model and strip `language_model.` from weight keys.
- Keep permanent prestack off for unstacked Kimi JANGTQ by default. The normal
  MLXPress path is compression-first canonical mmap residency; if active-expert
  streaming is enabled explicitly, label the row as fallback/diagnostic and
  require file/page-pressure evidence before drawing speed conclusions.
- For stacked Kimi JANGTQ2K bundles, stacked `switch_mlp` tensors are allowed
  only if the runtime reads one requested expert slice by safetensor byte
  offset. Reading the whole stacked tensor into an MLX array and then slicing
  is a high-RAM path and must remain blocked.
- For both stacked and unstacked Kimi, chunk before MoE routing during prefill;
  reducing inside `StreamingTurboQuantSwitchGLU` is too late because full
  prompt router readback has already built the large graph.
- During routed decode, keep a layer-output materialization boundary on by
  default until Kimi proves low peak plus coherent output in longer rows. The
  no-eval diagnostic is useful for profiling scheduler overhead, but the
  default should not be relaxed from one short `red` row.
- Cache mode: Kimi is dense/sliding-lineage attention, not Mamba SSM. Do not
  apply SSM companion disk rules unless a future config adds path-dependent
  state.
- Parser: use Kimi K2 tool-call format and Kimi thinking template; support
  `thinking` and `enable_thinking` aliases. The CLI now sends both because
  Kimi's standalone `chat_template.jinja` uses `thinking`, while
  `tokenizer_config.json` carries an `enable_thinking` bridge.
- KV memory: heads 64 and KV heads 64 mean KV cache is expensive; TurboQuant KV
  proof is mandatory, not optional.
- JANGTQ router/top-k: config has both `num_experts_per_tok=8` and `top_k=50`.
  Runtime routing must use the model's effective expert top-k, not blindly the
  config's retrieval/search top-k if those fields have different meanings.
- The current inspect row exposes the Kimi matmul path as
  `jangtq-hadamard-rotation`,
  `randomized-hadamard-pow2-blocks`,
  `turboquant-codebook-gather(tq_packed+tq_norms)`,
  `jangtq-runtime-signs-and-codebooks`,
  `routed-active-expert-slice-matmul`, and `fused-gate-up-swiglu-tq`; its
  position-vector contract is scalar offset plus partial RoPE head vector
  `qk=64`. Kimi also classifies as `mla-partial-rope-attention`, so cache
  hits must preserve the no-PE/partial-RoPE head split and not treat the row as
  ordinary full-head RoPE.
- Validation rows must include no-thinking, thinking, and tool-call prompts
  with cold/warm disk cache and token/s.

## Qwen3.6 MoE

Local bundle:

- `~/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK`

Live config:

- top-level `model_type=qwen3_5_moe`
- inner text `model_type=qwen3_5_moe_text`
- hidden 2048, layers 40, heads 16, KV heads 2
- experts 256, top-k 8
- MoE intermediate 512, shared expert intermediate 512
- `weight_format=mxtq`, uniform `mxtq_bits=2`

Implemented or currently proven:

- Imported model classes and JANGTQ support exist.
- Qwen XML-style reasoning and tool-call parser support exists in the common
  library surface.

Open or blocked:

- Needs combined-repo MLXPress-on multi-turn cache-stack proof.
- Hybrid/SSM notes from Osaurus must be reconciled against actual Qwen3.6
  model implementation before enabling full/partial disk hits.

Must check while building:

- If the model path has SSM or other path-dependent layers, full disk hits are
  unsafe without companion state; partial hits must be guarded.
- `rope_parameters` carries MRoPE fields on this family. Inspect now reads
  `rope_parameters` directly and reports MRoPE section/interleaving, instead
  of relying only on top-level `rope_theta` / `rope_scaling` keys.
- Qwen reasoning parser must preserve `.reasoning` and `.chunk` separately.
- Mixed media variants are not text-only if `vision_config` is present and
  bundle metadata does not explicitly set `has_vision=false`. They must expose
  VL attention, Qwen `[3,batch,seq]` position IDs, image/video `THW`, MRoPE
  delta vectors, media-salted cache keys, and a blocked media proof gate until
  real image/video cold/warm rows run.
- Uniform JANGTQ2 should not use the Hy3/MiniMax K mixed-bit path except through
  the common constructor with gate/up/down all equal.

## Qwen VL / MRoPE

Implemented or currently classified:

- The Qwen VL processors carry `cacheScopeSalt` from request context, and
  `computeCacheSalt` mixes media bytes, shape, dtype, reasoning/scope, and KV
  policy into cache identity.
- Qwen3 VL `getRopeIndex` builds `[3,batch,seq]` position IDs, image/video
  `THW` grid vectors, and MRoPE delta vectors. Inspect now exposes those as
  separate position-vector requirements instead of one generic RoPE row.

Must check while building:

- Text-only Qwen VL rows may use ordinary scalar position offsets, but real
  image/video rows must prove the 2D image and 3D video MRoPE grid path.
- Same text with different media must miss or use distinct salted entries;
  same text with same media must hit.
- Partial cache hits that would split a media-token span are invalid and must
  fall back to full prefill.
- Cache rows must record media processor settings, grid shapes, token counts,
  Activity Monitor peak, and token/s.

## ZAYA Text And VL

Local bundles:

- `~/models/JANGQ/ZAYA1-8B-JANGTQ_K`
- `~/models/JANGQ/ZAYA1-8B-JANGTQ4`
- `~/models/JANGQ/ZAYA1-VL-8B-JANGTQ_K`
- `~/models/JANGQ/ZAYA1-VL-8B-JANGTQ4`

Live text config:

- `model_type=zaya`
- hidden 2048, layers 80, heads 16, KV heads 2
- experts 16
- mixed JANGTQ_K routed bits gate/up 2-bit, down 4-bit
- attention/embed/lm head 8-bit, router/CCA/norms 16-bit

Implemented or currently proven:

- ZAYA split stacked JANGTQ tensor names are now included in mmap expert
  advice patterns.
- Zaya-specific cache classes exist: `ZayaCCACache`, `BatchZayaCCACache`.
- A short visible output smoke exists with thinking disabled, but reasoning-on
  is not confirmed.

Open or blocked:

- Need coherent reasoning-on text proof.
- Need VL image/video path proof for `zaya1_vl`, including media salt in cache
  keys.
- Inspect-only readiness rows for `ZAYA1-VL-8B-JANGTQ_K` and
  `ZAYA1-VL-8B-JANGTQ4` now block on `vl-vector-media-cache-proof` and
  `zaya-vl-media-cache`:
  `docs/local/model-validation/20260513T005500Z-zaya-vl-readiness-inspect/RESULT.md`.

Must check while building:

- CCA cache is path-dependent and must be handled like companion state, not as
  ordinary KV only.
- Disk L2 hits must include or reject CCA companion state.
- VL cache keys must include image/video content identity and processor config.
- Text and VL model factories must not share a cache entry unless media salt,
  tokenizer/template hash, and model hash match exactly.

## Ling / Bailing Hybrid

Local bundles:

- `~/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK`
- `~/models/dealign.ai/Ling-2.6-flash-MXFP4-CRACK`

Live JANGTQ config:

- `model_type=bailing_hybrid`
- hidden 4096, layers 32, heads 32, KV heads 32
- experts 256, top-k 8
- `weight_format=mxtq`
- routed experts 2-bit; attention/shared/dense/embed/lm head/MTP projection
  8-bit; norms/router/biases 16-bit

Implemented or currently proven:

- Bailing/Ling factory support exists in the combined library import.
- Prior Osaurus notes show topology-specific cache status support, but current
  combined-repo proof still needs to be rerun.

Open or blocked:

- Need MLXPress-on low-RAM multi-turn proof with JANGTQ2 and MXFP4 variants.
- Need determine whether any recurrent/hybrid state is path-dependent in the
  current implementation and whether disk L2 must carry companion state.

Must check while building:

- Heads 32 / KV heads 32 makes KV cache heavy; TurboQuant KV must be validated.
- Tool/reasoning format should be selected from the stamped capability, not
  generic JSON by accident.
- If hybrid state exists, full/partial disk hits need SSM-style guards and
  companion serialization.

## DeepSeek V4 Flash Exception

Local bundles:

- `~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K`
- `~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ2`

Live K config:

- `model_type=deepseek_v4`
- hidden 4096, layers 43, heads 64, KV heads 1
- top-k 6
- `weight_format=mxtq`
- routed experts 2-bit; attention/compressor/indexer/shared/embed/lm head
  8-bit; norms/router/hc 16-bit

Status:

- Treat as a special cache topology, not a template for Kimi/MiniMax/Hy3.
- Compressor/indexer attention has its own cache rules; do not generalize its
  disk-hit behavior to ordinary MoE or sliding attention models.

Must check while building:

- `DSV4_KV_MODE` and compressor/indexer cache settings must be included in
  cache keys.
- Reasoning/tool parser should use the DeepSeek family format.
- Full proof must include low-RAM, cold/warm cache, prompt/decode token/s, and
  no-loop output like every other row.
