# MLXPress Active Expert Scheduler Plan

Last updated: 2026-05-13

This plan is the implementation map for recovering the real MLXPress method:
compression-first canonical mmap residency for routed MoE weights. The intended
path keeps the model logically loaded, advises inactive routed pages cold, and
lets macOS reclaim/compress those pages so hot experts can be faulted back
quickly. The current Kimi active-streaming proof shows the failure mode clearly:
Activity Monitor RAM is low, swap-out is zero, but the runtime rereads about
20 GB of active stacked expert slices for one generated token. That is an
explicit fallback diagnostic, not the target architecture.

## Success Criteria

Every production row must prove all of these together:

- Activity Monitor peak stays below the family gate.
- Prompt and decode token/s are recorded.
- Visible and reasoning output are coherent, no-loop, and not max-token fake
  passes.
- Cache stack is enabled and cache hits are correct.
- `profile_read_mb_per_gen_token` and `effective_read_mb_per_gen_token` are
  below the row threshold.
- `pagein_mb`, `swapin_mb`, and `swapout_mb` are recorded.
- No permanent stacked tensor files are written as part of normal MLXPress
  operation.
- Active-expert streaming is off unless explicitly requested for a fallback or
  diagnostic row.

## Throughput Thesis

MLXPress should win because inactive routed experts are cheap to keep as
compressed/cold mmap pages compared with constantly moving their full
uncompressed buffers or rereading slices from SSD. A warm expert fault or
decompression can add latency, but the generated-token hot path should move a
much smaller effective working set: active routed experts, current attention
state, and cache blocks. If a row shows tens of GB of file reads or page-ins
per generated token, it is not proving this thesis even when Activity Monitor
looks low.

Current resident-loader blocker: the C++ safetensors mmap path is GPU-correct,
and the opt-in tensor-span mode now maps only each tensor's page-aligned byte
span instead of wrapping the whole shard in one Metal buffer. A first
MiniMax-Small tensor-span row still showed full Activity Monitor footprint
because MiniMax restacked non-stacked experts into full `switch_mlp` banks.
That variable is now removed in `MLXPRESS_RESIDENT_EXPERTS=1`: MiniMax
registers per-expert tensors into the streaming expert store and does not build
the full banks. The new load-only rows still report 35.2 GB footprint with
34.6 GB of live mmap-tracked Metal buffers. `MADV_PAGEOUT` and forced
`msync(MS_INVALIDATE | MS_ASYNC)` cold advice both return/succeed but do not
lower Activity Monitor footprint while all routed tensors have live Metal
buffers. The remaining production fix is a tensor-storage policy that does not
create live Metal buffers for the whole routed set, while still avoiding the
slow per-token active-bank reconstruction fallback.

Native compression primitive update: `JangPressMachCache` is restored in this
combined repo with MLXPress aliases (`MLXPressMachCache`,
`MLXPressMachConfig`, `MLXPressMachStats`). It allocates anonymous
`VM_FLAGS_PURGABLE` expert tiles, supports per-expert tensor components,
exposes acquire/release calls around hot routed experts, can build managed
MLXArray views over tile bytes, and releases cold tiles by the user compression
percentage. Focused tests prove registration, volatile accounting, hot pins,
unknown expert errors, component acquire/release, array view correctness, and
cold-percent release. The streaming expert store now has an opt-in bridge
(`MLXPRESS_STREAMING_MACH_ACTIVE_TENSORS=1`) that registers active tensor
bytes in `MLXPressMachCache`, exposes them through
`mlx_array_new_data_managed_payload`, and calls `releaseColdTiles` after
evaluated chunks. Strict MiniMax row
`docs/local/model-validation/20260513T120747Z-minimax-small-mach-active-tensors-noprofile-longer-generation/`
proved the bridge can preserve coherent two-turn output, pass the Activity
Monitor gate at 10.85 GB / 29.3%, and eliminate explicit active tensor file
reads. It is rejected as the MiniMax speed fix: decode was only 0.106 then
0.085 tok/s, with `reduce.call_chunk` dominating 802.8 s across 5,208 calls.
That means the fix has to restore resident compute semantics or move below the
Swift active-streaming call structure.

Active mmap fallback update: the new `mlx_array_new_mmap_file_region` C ABI is
now wired into `JANGTQStreamingExpertStore.load` for both unstacked per-expert
tensors and stacked expert slices when
`MLXPRESS_STREAMING_MMAP_ACTIVE_TENSORS=1`. MiniMax-Small longer row
`docs/local/model-validation/20260513T095717Z-minimax-small-streaming-mmap-active-tensors-longer-generation/`
proved the fallback keeps Activity Monitor low (2.66 GB / 7.2%), preserves
coherent multi-turn output, and eliminates explicit active-slice reads. It
still decodes at only 1.63 then 2.39 tok/s because it remains on the streaming
path: CPU router-index readback, temporary active-bank assembly, and per-chunk
reduce/eval dominate. An 8 GB tensor-residency variant stayed low-RAM but
regressed to about 1 tok/s, so bounded active tensor caching is not the speed
fix.

That narrows the old JangPress target: recovering 40+ tok/s requires resident
`TurboQuantSwitchGLU` semantics without full-bank heap materialization and
without live Metal buffers for the entire routed set. For non-stacked MiniMax
layout, the on-disk safetensors are expert-major and interleave `w1/w2/w3`;
same-projection expert tensors are not one contiguous file range. The next real
implementation is therefore either a direct offset-descriptor JANGTQ kernel
that reads expert-major/per-shard offsets using the GPU-side `rhsIndices`, or a
non-permanent page-backed stitched-bank loader that avoids copying the full
routed bank into ordinary heap/Metal-resident storage.

Offset-kernel status: `JANGTQStackedOffsetDescriptor` and
`JANGTQStreamingExperts.stackedOffsetDescriptors(...)` now classify true
`switch_mlp` stacked tensors (`stacked-contiguous`), MiniMax-style same-file
expert-major tensors (`expert-major-single-file-offsets`), and split-shard
expert-major tensors (`expert-major-multi-file-offsets`). The Metal kernels
accept `UInt32.max` sentinels for missing experts so per-shard outputs can be
summed. Focused tests cover descriptor layouts plus split-shard sentinel
summing for gate/up and down paths.

Release MiniMax-Small row
`docs/local/model-validation/20260513T110922Z-minimax-small-offset-kernels-multifile-release-longer-generation/`
proves the offset path is coherent and low-RAM with cache stack on: 2.70 GB /
7.3% Activity Monitor peak, zero explicit active-slice reads, 559.95
MB/generated-token effective read pressure, and no `stack.*` or
`router.indices_readback` profile rows. It still decodes at only 1.07 tok/s,
so offset dispatch fixed file/readback pressure but did not recover the old
resident JangPress speed. The no-layer-eval diagnostic improves decode only to
1.16 tok/s, so the next speed work must target the streaming call/kernel
structure or restore resident compute semantics without live full-bank Metal
buffers.

Scored-down kernel status: `JANGTQKernels.gatherTQTopKOffsetsScored` and the
opt-in `MLXPRESS_STREAMING_SCORED_DOWN_KERNELS=1` path now fuse router-score
reduction into the offset down-proj kernel. Focused kernel tests cover both
same-file expert-major offsets and split-shard sentinels against the existing
gather + score-sum reference. MiniMax row
`docs/local/model-validation/20260513T131500Z-minimax-small-offset-scored-down-longer-generation/`
passed coherent two-turn output, cache stack, zero explicit reads, 2.53 GB /
6.8% peak Activity Monitor footprint, 623.56 MB/generated-token effective read
pressure, and 1.150 tok/s aggregate decode. This is a real offset-path
improvement over the prior 1.071 tok/s row, but it is still far from the
resident JangPress speed target.

No-eval scored-down row
`docs/local/model-validation/20260513T132500Z-minimax-small-offset-scored-down-no-eval-longer-generation/`
is the current offset-graph upper bound: coherent two-turn output, zero
explicit reads, 2.64 GB / 7.1% peak Activity Monitor footprint, and 1.250 tok/s
aggregate decode. It is not production-safe because MLX allocator peak rose to
1.57 TB. This confirms materialization boundaries cost speed, but the scheduler
must be graph/allocator-bounded rather than a fixed no-eval or every-N-layer
policy.

Allocator-pressure eval scheduler status: created, not yet model-proven.
`MLXPRESS_STREAMING_EVAL_MLX_PEAK_MB` /
`JANGPRESS_STREAMING_EVAL_MLX_PEAK_MB` and
`MLXPRESS_STREAMING_EVAL_MLX_ACTIVE_MB` /
`JANGPRESS_STREAMING_EVAL_MLX_ACTIVE_MB` add an opt-in pressure boundary for
active-expert streaming rows. With `MLXPRESS_STREAMING_EVAL_EACH_LAYER=0`, the
runtime now keeps the no-eval graph shape until MLX allocator peak or active
memory crosses the configured MB limit, then materializes through the same
`MLX.eval` + `MLX.Memory.clearCache()` path used by forced per-layer eval.
Profile rows named `scheduler.allocator_eval` and
`scheduler.allocator_eval.layer_<n>.<phase>` mark each pressure-triggered
boundary; peak accounting resets after materialization by default. This is the
next experiment to run against MiniMax/Hy3/Kimi because it directly targets the
1.57 TB no-eval failure without reverting to every-layer eval overhead.

MiniMax allocator-pressure row
`docs/local/model-validation/20260513T134510Z-minimax-small-offset-scored-down-allocator-peak128/`
proved the scheduler path on real two-turn decode: coherent red-apple and
blue-sky answers, zero explicit reads, 844.42 MB/generated-token effective read
pressure, 0 MB swap-out, 2.53 GB / 6.8% peak Activity Monitor footprint, and
128.69 GB MLX peak with `scheduler.allocator_eval` count 73. Decode was only
1.241 tok/s aggregate, so the scheduler bounds graph residency but does not
recover the old 39-47 tok/s resident JangPress speed target.

Load-time resident-compute diagnostics narrow the path further. Row
`docs/local/model-validation/20260513T164210Z-minimax-small-loadtime-mach-stacks-resident-one-turn/`
materialized whole projection stacks into `MLXPressMachCache` and used
resident `TurboQuantSwitchGLU` compute with active streaming off. It reached
7.99 tok/s and coherent output, but failed low-footprint after GPU touch at
35.34 GB / 95.4% peak. Whole-projection purgeable anonymous stacks are too
coarse.

The first ephemeral MiniMax resident-compute baseline was
`docs/local/model-validation/20260513T170143Z-minimax-small-ephemeral-prestack-mmap-noprofile-2turn/`:
`MLXPRESS_PRESTACK=1`, `MLXPRESS_PRESTACK_EPHEMERAL=1`,
`MLX_SAFETENSORS_MMAP=1`, `MLX_SAFETENSORS_MMAP_START_COLD=1`, active
streaming off, cache stack on, and no generation profiler. It passed coherent
two-turn output, removed the temporary 32.8 GB overlay at exit, kept peak
Activity Monitor to 2.62 GB / 7.1%, used zero explicit reads, and decoded at
12.60 / 14.01 tok/s. It failed effective read pressure at 1,997.41
MB/generated-token.

The current best MiniMax resident-compute baseline is the no-cold follow-up
`docs/local/model-validation/20260513T170902Z-minimax-small-ephemeral-prestack-mmap-nocold-2turn/`.
It keeps the same ephemeral mmap-prestack storage but does not force pages cold
at load. It stayed coherent for two turns, removed the temporary overlay at
exit, kept peak Activity Monitor to 2.62 GB / 7.1%, used zero explicit reads,
passed effective read pressure at 662.31 MB/generated-token, and decoded at
13.89 / 14.49 tok/s. This proves forced cold-start advice is counterproductive
for this diagnostic path. MLXPress now exposes this recipe as
`--ephemeral-prestack on` so it is reproducible without a hand-written env-var
stack. The next implementation should preserve no-cold resident
`TurboQuantSwitchGLU` semantics and mmap-backed stacked storage, but must avoid
explicit normal-path overlays and explain the remaining gap to the old 39-47
tok/s target.

Strict CLI flag proof:
`docs/local/model-validation/20260513T174200Z-minimax-small-cli-ephemeral-prestack-nocold-strict-2turn/`.
It used `--ephemeral-prestack on`, cache stack, disk L2, TurboQuant KV, and
`--fail-on-length-stop`; both turns ended with `stop=stop`, coherency passed,
the temporary 32.8 GB overlay was cleaned up after mmap load with a process-exit
fallback, peak Activity Monitor was 2.63 GB / 7.093%, explicit reads were
0 MB/generated-token, effective read pressure passed at
780.76 MB/generated-token, and aggregate decode was 13.162 tok/s. This confirms
the option wiring but does not close the
speed gap.

KV-mode diagnostics split the remaining speed gap. Short no-cold row
`docs/local/model-validation/20260513T171226Z-minimax-small-ephemeral-prestack-mmap-nocold-kvnone-2turn/`
uses `--kv-cache none` with the same resident weight path and reaches
24.56 / 24.44 tok/s, so TurboQuant KV is a large short-turn latency cost. On a
longer coherent paragraph, however,
`docs/local/model-validation/20260513T171708Z-minimax-small-ephemeral-prestack-mmap-nocold-tqkv-long/`
gets 19.70 tok/s with TurboQuant KV, while
`docs/local/model-validation/20260513T171915Z-minimax-small-ephemeral-prestack-mmap-nocold-kvnone-long/`
gets 21.61 tok/s without it. That means TurboQuant KV explains about a 10%
sustained gap; the larger gap to the old 39-47 tok/s target is still resident
routed compute/eval behavior and explicit overlay architecture.

Pre-fix compiled MiniMax diagnostic:
`docs/local/model-validation/20260513T175200Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-profile-strict-2turn/`
tested `--compiled-decode on`, `--allow-minimax-compiled-decode`,
`--compiled-max-cache-length 512`, and `--kv-cache none` on the same no-cold
ephemeral-prestack path. The compiled closure engaged and reached 24.56 /
24.64 tok/s with peak Activity Monitor 2.58 GB / 6.943%, zero explicit reads,
and 158.58 MB/generated-token effective read pressure. It is rejected because
both turns visibly looped and length-stopped. Do not treat compiled MiniMax as
the scheduler speed solution unless it passes the normal strict coherency
gates.

Compiled MiniMax cache-promotion follow-up:
`docs/local/model-validation/20260513T180400Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-promotedcache-strict-2turn/`
patches the single-sequence compile path to promote plain `KVCacheSimple`
layers into `CompilableKVCache` before compiling. With that fix, the same
no-cold ephemeral-prestack + KV-none row passes strict two-turn coherency and
reaches 26.29 / 26.26 tok/s, peak Activity Monitor 2.64 GB / 7.109%, zero
explicit reads, and 845.69 MB/generated-token effective read pressure. This
moves the speed baseline up, but does not replace the scheduler/storage work:
it still disables TurboQuant KV and still depends on a temporary stacked
overlay. Longer guard row
`docs/local/model-validation/20260513T180900Z-minimax-small-cli-ephemeral-prestack-compiled-kvnone-promotedcache-longer-2turn/`
generated 107 coherent visible tokens over two turns at 24.15 / 25.62 tok/s
with peak Activity Monitor 2.79 GB / 7.510%, zero explicit reads, and 284.56
MB/generated-token effective read pressure. The accepted production path needs
TurboQuant KV/cache-stack parity, no explicit overlay, and resident graph/eval
or persistent/offset-addressed kernel work that moves toward the old 39-47
tok/s target.

Compiled TurboQuant-KV follow-up:
`TokenIterator.setupCompiledDecode` now mirrors the BatchEngine
TurboQuant-cache promotion: compress post-prefill `TurboQuantKVCache` layers,
promote them to `CompilableTurboQuantKVCache`, and compile against the promoted
cache reference. Strict MiniMax row
`docs/local/model-validation/20260513T181400Z-minimax-small-cli-ephemeral-prestack-compiled-tqkv-promotedcache-strict-2turn/`
passes with `default-kv=turboQuant(k=3,v=3)`, 26.95 / 24.11 tok/s, peak
Activity Monitor 2.53 GB / 6.818%, zero explicit reads, and 845.65
MB/generated-token effective read pressure. Longer cache-stack row
`docs/local/model-validation/20260513T181800Z-minimax-small-cli-ephemeral-prestack-compiled-tqkv-promotedcache-longer-2turn/`
generates 169 coherent no-loop visible tokens at 22.03 / 23.13 tok/s, peak
Activity Monitor 2.63 GB / 7.089%, zero explicit reads, and 180.10
MB/generated-token effective read pressure. This row was slowed by an
unrelated CLI policy bug: `--activity-gate` also tightened
`MLX.Memory.memoryLimit`, throttling compiled decode.

Activity-gate unthrottled follow-up:
`docs/local/model-validation/20260513T192000Z-minimax-small-mlxpress-cache-stack-compiled-tqkv-activitygate-unthrottled-2turn-stop/`
passes the actual cache-stack proof with TurboQuant KV, disk L2, compiled
decode, `--fail-on-length-stop`, and `--activity-gate 59`: 49.09 / 48.66
tok/s, both turns `stop=stop`, coherent no-loop visible output, peak Activity
Monitor 2.65 GB / 7.159%, zero explicit reads, and 0.09 MB/generated-token
effective read pressure. This recovers the old MiniMax JangPress target for
the explicit ephemeral-prestack diagnostic. Scheduler work remains required
only to remove the temporary overlay and provide a no-permanent-file resident
mmap/offset kernel path with the same speed/RAM/coherency behavior.

Cleanup-after-load follow-up:
`docs/local/model-validation/20260513T193500Z-minimax-small-mlxpress-ephemeral-cleanup-after-load-compiled-tqkv-2turn/`
proves the mapped overlay can be deleted immediately after load. The row logs
overlay creation and removal before decode, then generates coherent two-turn
output at 47.54 / 47.19 tok/s with TurboQuant KV, peak Activity Monitor
2.64 GB / 7.131%, zero explicit reads, and
585.69 MB/generated-token effective read pressure. This closes the
no-permanent-file behavior for the diagnostic path, but not the final
no-overlay production path because the 32.8 GB overlay is still written during
load.

The profiled long row
`docs/local/model-validation/20260513T172505Z-minimax-small-ephemeral-prestack-mmap-nocold-tqkv-long-profile/`
confirms that split. It stayed coherent at 19.07 tok/s, with
`decode.async_eval_submit` taking 6,161.4 ms total / 47.396 ms average, while
`decode.kv_quantize` took 352.6 ms total / 2.733 ms average. Do not treat
TurboQuant KV as the only sustained-speed blocker; the next real performance
work has to reduce the evaluated resident graph cost or introduce a compiled /
persistent resident-kernel path that is compatible with the low-footprint
mmap-prestack storage.

Active-shard filter status: created as a diagnostic and rejected as the MiniMax
speed fix. `MLXPRESS_STREAMING_OFFSET_ACTIVE_SHARD_FILTER=1` /
`JANGPRESS_STREAMING_OFFSET_ACTIVE_SHARD_FILTER=1` reads routed expert ids for
offset-dispatch chunks and filters split-shard offset files by a shared active
file set across paired gate/up/norm descriptors. The first implementation
filtered projections independently and failed before generation with
`mismatched offset gate/up shard groups for layer 61`; the corrected row
`docs/local/model-validation/20260513T135912Z-minimax-small-offset-active-shard-filter-peak128/`
passed coherent two-turn output, zero explicit reads, 843.77
MB/generated-token effective read pressure, 0 MB swap-out, and 2.44 GB / 6.6%
peak Activity Monitor footprint. It reduced MLX peak to 40.69 GB, but decode
fell to 1.134 tok/s because `router.indices_readback` dominated 33.95 s.
Conclusion: skip inactive offset shards only if shard selection stays on GPU;
CPU readback is not the speed path.

Hy3 K offset-alignment status: guarded for correctness, not yet fast.
`JANGTQStreamingExpertIndex.hasAlignedOffsetShardGroups(layerIdx:)` now checks
that gate/up/norm descriptor groups share the same file set before the fused
offset path is allowed. Layers that do not satisfy that invariant record
`offset_dispatch_unaligned_shards` and fall back instead of failing with
`mismatched offset gate/up shard groups`. Row
`docs/local/model-validation/20260513T141501Z-hy3-k-offset-alignment-fallback-no-thinking-short/`
proved the guard on Hy3-preview-JANGTQ_K: coherent two-turn `4` / `6`, cache
stack on, zero explicit reads, and 4.44 GB / 4.4% peak Activity Monitor
footprint. It is rejected as an MLXPress speed/read-pressure proof because
decode was only 0.106 then 0.102 tok/s and effective read pressure was
217,923 MB/generated token from page-ins. Hy3 K needs a shard-aware offset
kernel that keeps shard selection on GPU, or the resident-compute path, before
it can use MLXPress at usable speed.

Hy3 K active-shard filter row
`docs/local/model-validation/20260513T142627Z-hy3-k-offset-active-shard-filter-no-thinking-short/`
confirms CPU-side filtering is not enough. It preserved coherent `4` / `6`
output and low Activity Monitor footprint, but aggregate decode was only
0.118 tok/s and effective read pressure still failed at 163,286 MB/generated
token. The row reduced page-ins versus the unfiltered guard row, but the cost
moved to `router.indices_readback` (51.63 s over 474 calls). Do not make this
default for Hy3 or MiniMax; shard selection must happen on GPU or inside a
resident-compute path.

Hy3 K active-window offset-span row
`docs/local/model-validation/20260513T143837Z-hy3-k-offset-active-window-filter-no-thinking-short/`
narrows each offset mmap span to the selected active expert byte window. It
preserved coherent `4` / `6` output, cache stack/disk L2/TurboQuant KV, and
4.38 GB / 4.3% peak Activity Monitor footprint. It is a real diagnostic
improvement over CPU active-shard filtering: aggregate decode rose from 0.118
to 0.558 tok/s, page-ins fell from 326,573 MB to 90,082 MB, and
`tensor.offset_span_active_window` skipped 519,023 MB of mapped span bytes. It
is still rejected for readiness because effective read pressure remains
45,041 MB/generated token and `router.indices_readback` still costs 16.31 s
over 474 calls. The next implementation target remains GPU-side shard
selection or resident-weight compute semantics, not more CPU-side filtering.

Hy3 K flexible unaligned-shard grouping
`docs/local/model-validation/20260513T150028Z-hy3-k-flexible-offset-groups-no-thinking-short/`
lets offset dispatch group gate/up/down spans by shared expert coverage instead
of identical file sets. This is a compatibility fix for Hy3-style unaligned
shards: the row recorded `offset_dispatch_flexible_shard_groups=6`, stayed
coherent (`4` / `6`), kept cache stack/disk L2/TurboQuant KV on, and held peak
Activity Monitor footprint to 4.41 GB / 4.3%. It is rejected as a speed/read
pressure fix. Without active-window narrowing, it mapped 1,347,166 MB of offset
spans, caused 332,258 MB of page-ins, failed effective read pressure at
166,129 MB/generated token, and decoded at only 0.112 tok/s. This proves the
dispatcher can tolerate unaligned shard sets, but broad full-span mmap is not
the MLXPress method.

Hy3 K segmented active-window grouping
`docs/local/model-validation/20260513T152335Z-hy3-k-segmented-groups-fixed-no-thinking-short/`
fixes the next span granularity issue. Earlier segmented attempts
`20260513T151616Z` and `20260513T151837Z` decoded empty visible output because
the flexible offset group key used only file URLs; multiple single-expert
windows from the same shard file set overwrote one another. The fixed
implementation adds expert coverage to the group key and keeps one active
expert window per descriptor segment. The row stayed coherent (`4` / `6`) with
cache stack, disk L2, and TurboQuant KV on, held peak Activity Monitor footprint
to 4.39 GB / 4.3%, improved aggregate decode to 0.729 tok/s, and reduced
page-ins to 29,889 MB over two generated tokens. This is a real improvement
over active-window min/max spans (90,082 MB page-ins, 0.558 tok/s) and broad
flexible spans (332,258 MB page-ins, 0.112 tok/s). It is still rejected for
readiness: effective read pressure is 14,944 MB/generated token and
`router.indices_readback` remains the dominant synchronization cost at 5.43 s
over 474 calls. The next step is GPU-side active-window selection or resident
JangPress compute semantics.

Hy3 K after the activity-gate memory-limit fix
`docs/local/model-validation/20260513T201500Z-hy3-k-segmented-groups-unthrottled-no-thinking-short/`
rules out the MiniMax throttle as the Hy3 K speed blocker. The row preserved
coherent `4` / `6` output, cache stack/disk L2/TurboQuant KV, zero explicit
reads, and 4.41 GB / 4.336% peak Activity Monitor footprint, but aggregate
decode improved only from 0.729 to 0.792 tok/s while effective read pressure
remained failed at 14,953 MB/generated token. Top profile rows stayed
`router.indices_readback`, `reduce.call_chunk_scored`, and
`tensor.offset_span_mmap_array`. Do not spend more time on activity-gate
tuning for Hy3 K; the next fix must move active-window/shard selection onto
GPU or restore resident compute semantics.

## Components To Build

### JANGTQ Hadamard SIMD-Shuffle Parity

Status: created.

Implemented surface:

- `JANGTQKernelLibrary.hadamardShuffleLE1024`
- `JANGTQKernels.hadamardRotate` dispatch for `maxBlock <= 1024`
- `Tests/MLXLMCommonFocusedTests/JANGTQHadamardShuffleTests.swift`

Why:

MiniMax's routed intermediate activation rotation is 1536 = 1024 + 512. The
Python vMLX path uses SIMD lane shuffles for the first five Hadamard butterfly
stages on <=1024 blocks; the Swift path was still using the heavier
threadgroup-memory multiblock kernel for that hot dimension. This fix restores
that generic JANGTQ prerequisite and is independent of the compression-first
residency work. It is not itself proof that MiniMax or Kimi model decode is
fast; token/s still has to come from a real coherent generation row.

### File-Read Pressure Gate

Status: created.

Runtime surface:

- `mlxpress --file-read-gate-mb-per-token N`
- `mlxpress --file-read-gate-report-only`
- `file_read_pressure` JSONL records
- `effective_read_pressure` JSONL records, using max(explicit reads, page-ins)
- `scripts/summarize-mlxpress-metrics.py` fields:
  `profile_read_mb`, `profile_read_mb_per_gen_token`, `file_read_gate`,
  `effective_read_mb_per_gen_token`, and `effective_read_gate`

Use this gate before calling any streaming JANGTQ row ready. Low RAM alone is
not enough, and a mmap path is not ready unless the effective page-in gate also
passes.

### Active Expert Use Trace

Status: created; Kimi proof exists.

Needed helper:

- `MLXPressActiveExpertTrace`

Responsibilities:

- Record per layer, token chunk, and expert id for active routed experts.
- Count immediate reuse across consecutive calls for the same layer.
- Emit compact `active_expert_trace` JSONL records with top repeated experts.

Kimi proof:

- `docs/local/model-validation/20260513T043500Z-kimi-k-active-expert-trace/`
  produced coherent `red`, kept peak footprint at 1.76 GB / 0.5%, and
  recorded 180 trace calls, 1,440 routed slots, 1,440 unique expert touches,
  309 consecutive reuse touches, and reuse rate 0.215.

Why:

The scheduler now knows that this Kimi row has only about 21.5% immediate
expert reuse. That bounds how much a simple residency cache can help and
points the next fix toward direct stacked-offset dispatch, not unbounded
caching.

### Active Slice Residency Budget

Status: created as a bounded diagnostic; not a production speed fix.

Created helper:

- `MLXPressActiveSliceResidency`

Responsibilities:

- Keep active expert slices under a configurable MB budget.
- Track resident bytes by `(layer, expert, projection, suffix)`.
- Evict least-recently-used slices within the same
  `MLXPRESS_STREAMING_EXPERT_CACHE_MB` budget.
- Emit `active_slice_residency` JSONL records with hit rate, byte hit rate,
  resident bytes, stores, and evictions.

Kimi proof:

- `docs/local/model-validation/20260513T043530Z-kimi-k-active-slice-residency-512mb/`
  had 0.000 hit rate, 0.119 tok/s, and peak 2.47 GB / 0.8%. A 512 MB budget
  is too small for Kimi's reuse distance.
- `docs/local/model-validation/20260513T043713Z-kimi-k-active-slice-residency-8192mb/`
  had 0.215 hit rate and reduced file reads from 20,190.94 MB to
  15,858.30 MB per generated token, but decode stayed 0.117 tok/s and peak
  footprint rose to 11.35 GB / 3.5%.

Why:

The target is not maximum caching. The target is just enough hot-slice
residency to reduce file reads per token without expanding process footprint
into the full model size. The current diagnostic proves the cache wiring works
but is not enough for Kimi speed; direct stacked-offset dispatch is still the
main fix.

### Active Bank Cache

Status: created as a bounded diagnostic and rejected as a MiniMax speed fix.

Runtime surface:

- `MLXPRESS_STREAMING_BANK_CACHE_MB`
- `JANGPRESS_STREAMING_BANK_CACHE_MB`
- `active_slice_residency.bank_*` metrics
- `scripts/summarize-mlxpress-metrics.py` fields:
  `bank_resident_mb` and `bank_evictions`

MiniMax proof:

- `docs/local/model-validation/20260513T073632Z-minimax-small-streaming-8gb-tensor-8gb-bank-cache-glu/`
  used 8 GB tensor residency and 8 GB active-bank residency with active
  streaming, cache stack, TurboQuant KV, and a relaxed 59% Activity Monitor
  report gate. It stayed coherent and passed file/effective read-pressure
  gates, but decode averaged 1.094 tok/s and peak footprint rose to 18.44 GB /
  49.8%.
- Bank reuse was only 6 hits against 17,478 misses, with 15,663 bank
  evictions. Exact `(layer, projection, suffix, sortedExperts)` bank reuse is
  too sparse for MiniMax's routed sets.

Why:

Caching fully assembled active banks sounds like a cheap way to avoid repeated
bank construction, but the exact top-k set is not stable enough. It mostly
adds another 8 GB of resident MLX arrays. Keep the knob default-off, and do
not spend more time tuning the exact-bank cache unless a future family trace
shows high exact-set repetition.

The older JangPress MiniMax rows are the methodology target instead:
resident-weight compute with macOS reclaiming cold routed pages. That is why
the old notes show roughly 40-47 tok/s with about 5.4 GB RSS and 9.4 GB routed
resident after forced release, i.e. the practical "usable speed with far less
than 59% original routed footprint" target. However, those rows tracked RSS, not
Activity-Monitor-style `phys_footprint`, and one MiniMax row had a chunk
buffer output bug. Treat them as a speed/RSS target and mechanism clue, not as
current production proof. The current streaming path is a fallback and cannot
get there by reassembling active banks per token.

### Offset-Span Residency Diagnostic

Status: implemented as an opt-in diagnostic.

Knobs:

- `MLXPRESS_STREAMING_OFFSET_SPAN_CACHE_MB`
- `JANGPRESS_STREAMING_OFFSET_SPAN_CACHE_MB`
- profile rows: `tensor.offset_span_cache_hit`,
  `tensor.offset_span_cache_store`

Why:

The offset kernels already avoid explicit `pread` and avoid rebuilding
temporary active expert banks, but the current low-RAM rows still recreate the
same mmap-backed descriptor spans many times. This cache holds offset spans by
`(file, offset, byteCount, dtype)` under a byte budget, including
route-specific active expert windows. It is a direct test of the old JangPress
resident-compute method: keep reusable offset buffers warm and let macOS manage
page residency, without writing permanent stacked overlays.

Acceptance:

- MiniMax/Hy3/Kimi rows must record prompt and decode token/s.
- Activity Monitor peak must stay under the family gate.
- Effective read pressure must stay under the row gate.
- Output must pass coherent multi-turn/no-loop checks.

If decode remains around 1 tok/s, treat full offset-span residency as another
rejected streaming-side mitigation and move to resident/persistent kernel
storage instead of tuning more CPU-side filters.

Result:

MiniMax-Small row
`docs/local/model-validation/20260513T0913Z-minimax-small-offset-full-span-cache-128gb-noprofile-2turn/`
used `MLXPRESS_STREAMING_OFFSET_SPAN_CACHE_MB=131072` with streaming profile
off. It passed coherence, cache-stack, explicit-read, effective-read, and
Activity Monitor gates: peak footprint was 2.54 GB / 6.9%, and effective read
pressure was 826.25 MB/generated token. It failed the purpose of this
diagnostic because decode was only 0.823 then 0.816 tok/s. The profiled sibling
row `20260513T0910Z-minimax-small-offset-full-span-cache-128gb-2turn` had
98.85% offset-span cache hit rate by turn 2 with no evictions, yet still
decoded around 0.8 tok/s. This rejects full-span residency as the MiniMax speed
fix and narrows the next target to resident `TurboQuantSwitchGLU` semantics or
a lower-level persistent/offset-addressed JANGTQ kernel.

Kimi K active-window cache row
`docs/local/model-validation/20260513T195406Z-kimi-k-offset-active-window-cache32gb-short/`
used a 32 GB route-specific offset-span cache on the 328 GB Kimi K2.6
JANGTQ_K bundle. It produced a coherent 30-token sentence, held peak Activity
Monitor footprint to 2.41 GB / 0.735%, kept explicit reads at 0 MB/token, and
proved macOS can keep the logical resident set compressed. It is rejected as
readiness because decode was 0.321 tok/s and effective read pressure failed at
4,780.81 MB/generated token. The trace touched most of the routed expert set
over one short generation: 25,440 routed slots, 21,196 unique expert touches,
0.263 reuse rate, and 60,895 cache evictions. This rejects "just increase the
file-backed active-window LRU" as the Kimi fix. The next Kimi path must use
anonymous/purgeable resident storage or a persistent offset-addressed kernel
that does not refault from file-backed pages for every broad expert sweep.

### Mach Offset-Span Residency Diagnostic

Status: created, opt-in, needs Kimi proof.

Knobs:

- `MLXPRESS_STREAMING_MACH_OFFSET_SPANS`
- `JANGPRESS_STREAMING_MACH_OFFSET_SPANS`
- `MLXPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB`
- `JANGPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB`
- profile rows: `tensor.mach_offset_read`,
  `tensor.mach_offset_register`, `tensor.mach_offset_hit`,
  `tensor.mach_offset_array`, `tensor.mach_offset_evict`,
  `tensor.mach_offset_budget_skip`

Why:

Kimi showed that a file-backed active-window LRU can stay tiny in Activity
Monitor and still page-in too much data. This diagnostic registers exact
one-expert offset windows into `MLXPressMachCache`, keyed by
`(layer, expert, projection, suffix)`, so repeated expert touches can come from
anonymous purgeable memory instead of clean file-backed mappings. It is
deliberately narrower than the offset-span cache: coalesced multi-expert
windows and full descriptor spans still fall back to mmap/cache, because this
row is meant to isolate macOS native compression behavior from broad
file-backed remapping.

Implementation note after the first failed Kimi warm row: Mach offset
registration now reads the safetensors range directly into the purgeable VM
region through `JangPressMachCache.registerFilled`, avoiding the earlier
`pread` -> `Data` -> Mach copy sequence. A budget knob can force LRU removal or
fallback for exact offset tiles so the diagnostic no longer has to retain every
Kimi expert window.

Acceptance:

- First-turn `tensor.mach_offset_read` is allowed because the tile must be
  populated from real safetensors; warm or multi-turn rows must show
  `tensor.mach_offset_hit` and lower effective read pressure.
- Activity Monitor peak must remain below the Kimi family gate.
- Decode token/s must improve versus the 32 GB file-backed active-window row,
  not merely stay coherent.
- Output must pass visible/reasoning no-loop checks, with cache stack and
  TurboQuant KV enabled.

First result:

`docs/local/model-validation/20260513T202041Z-kimi-k-mach-offset-spans-1tok/`
passes wiring/coherence/RAM but fails speed and read pressure. It produced
coherent `red`, peak Activity Monitor 6.02 GB / 1.834%, 6,432 Mach registered
offset tensors, and 15,031.03 MB of `tensor.mach_offset_read`. Effective read
pressure improved versus the no-cache offset baseline but still failed at
24,819.72 MB/generated-token, and decode regressed from 0.955 to 0.368 tok/s.
Treat this as proof that the bridge is active and measurable, not a Kimi fix.
The next row must be warm/multi-turn and show `tensor.mach_offset_hit` raising
decode token/s while keeping Activity Monitor RAM low.

Warm result:

`docs/local/model-validation/20260513T202514Z-kimi-k-mach-offset-spans-2turn/`
confirms the current unbounded Mach offset path is not enough. It is coherent
(`red` / `blue`) and stays under the RAM gate at 20.88 GB / 6.364%, but
aggregate decode drops to 0.209 tok/s and turn 2 is only 0.144 tok/s. Resident
Mach offset bytes grow to 60.73 GB, file-read pressure is
30,363.52 MB/generated-token, and effective read pressure is
30,871.54 MB/generated-token. The file-read classifier now correctly reports
`rows=tensor.mach_offset_read`. Next scheduler work must be bounded and
routing-aware; simply retaining every exact offset tile is a rejected Kimi
shape.

Code follow-up now implemented:

- `JangPressMachCache.registerFilled` fills purgeable storage directly from the
  cached safetensors descriptor.
- `MLXPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB` /
  `JANGPRESS_STREAMING_MACH_OFFSET_SPAN_BUDGET_MB` cap exact Mach offset tiles.
- `tensor.mach_offset_evict` and `tensor.mach_offset_budget_skip` distinguish
  budget eviction from fallback. The next Kimi row should compare uncapped
  direct-fill versus a bounded run; either way it must improve decode token/s,
  stay low in Activity Monitor, and remain coherent.

### Active Offset-Window Coalescing

Status: created, opt-in only, needs model-row proof.

Knobs:

- `MLXPRESS_STREAMING_OFFSET_ACTIVE_WINDOW_COALESCE_MB`
- `JANGPRESS_STREAMING_OFFSET_ACTIVE_WINDOW_COALESCE_MB`
- profile row: `tensor.offset_span_active_coalesced_experts`

Why:

The one-expert-per-window active offset path is excellent for tiny Activity
Monitor footprint, but it creates many mmap windows and offset-kernel groups.
The old JangPress target is not "sub-5 GB at any speed"; it is usable decode
speed while staying below the family footprint gate. `JANGTQActiveOffsetWindow`
and `JANGTQStackedOffsetDescriptor.activeExpertByteWindows(...)` keep one
expert per window by default and can merge across a bounded byte gap only when
the env knob is set. This intentionally trades some extra mapped bytes for
fewer windows/groups, so every row must record token/s, effective read pressure,
and Activity Monitor footprint before treating it as progress.

First result:

`docs/local/model-validation/20260513T191100Z-hy3-k-offset-window-coalesce16-no-thinking-short/`
sets a 16 MB gap. It keeps Hy3 K coherent (`4` / `6`) and below the Activity
Monitor gate at 4.54 GB / 4.477%, and improves decode from the unthrottled
segmented row's 0.792 tok/s to 1.004 tok/s. It fails readiness because page-ins
rise to 58,690.22 MB and effective read pressure doubles to
29,345.11 MB/generated-token. Treat 16 MB as too broad; sweep smaller gaps
before assuming coalescing is the speed path.

Rejected regression row:

`docs/local/model-validation/20260513T192000Z-hy3-k-offset-window-coalesce0-current-no-thinking-short/`
was run while adjacent active ranges still merged by default. It stayed
coherent and low-footprint, but produced the same bad shape as the 16 MB row:
0.962 tok/s, 58,710.64 MB page-ins, and
29,355.32 MB/generated-token effective read pressure. That row proves that
adjacent coalescing is already too broad for Hy3's segmented offset path; the
default must preserve one window per active expert and all coalescing must be
explicit.

Corrected default row:

`docs/local/model-validation/20260513T193000Z-hy3-k-offset-window-default-no-coalesce-current-no-thinking-short/`
was run after default coalescing was disabled. It stayed coherent (`4` / `6`)
and low-footprint at 4.320% peak Activity Monitor footprint, and returned to
the lower segmented pressure shape: 0.781 tok/s decode, 22,803.84 MB mapped
spans, 29,897.86 MB page-ins, and 14,948.93 MB/generated-token effective read
pressure. This proves the regression fix, not readiness.

### Routing-Aware Prefetch Scheduler

Status: missing.

Needed helper:

- `MLXPressActiveExpertScheduler`
- `MLXPressActiveExpertPrefetch`

Responsibilities:

- Use router output for the current chunk to schedule gate/up/down reads for
  the current layer.
- Optionally prefetch the next layer or next token when the routing trace
  predicts reuse.
- Use bounded async `pread` workers and avoid blocking the decode stream on
  avoidable shard open/seek work.
- Never retain prefetched slices past budget.

Why:

Current Kimi streaming bank reads are sequential and correct, but they bypass
the resident-compression premise. Prefetch only helps as a fallback mitigation;
the primary scheduler should drive canonical mmap warm/cold advice and avoid
turning decode into repeated `pread` scatter.

### Direct Stacked-Offset JANGTQ Dispatch

Status: first offset-addressed JANGTQ kernels exist and are proven on
MiniMax-style expert-major tensors, including split safetensor shards. They are
a low-RAM/read-pressure fix, not yet a production-speed fix. The old
whole-tensor mmap prototype remains a false lead.

Needed helper/kernel surface:

- `JANGTQStackedOffsetDescriptor`
- `JANGTQKernels.fusedGateUpSwiGLUOffsets`
- `JANGTQKernels.gatherTQOffsets`
- `JANGTQKernels.gatherTQTopKOffsets`
- Follow-up: a resident/purgeable storage path or fused persistent-kernel path
  that avoids the remaining per-layer streaming call overhead.

Responsibilities:

- Consume stacked safetensor slice offsets and expert-local metadata without
  building temporary MLX bank arrays per token.
- Preserve randomized Hadamard signs, runtime codebooks, `tq_packed`,
  `tq_norms`, mixed gate/up/down bits, and fused gate/up SwiGLU semantics.
- Work for stacked and unstacked JANGTQ layouts through the same logical API.

Why:

Kimi currently builds active banks for every routed layer/chunk. That is
low-RAM, but byte-heavy. Direct stacked-offset dispatch is the path that can
reduce repeated file reads and MLX array construction together.

False lead already proven:

- `docs/local/model-validation/20260513T045423Z-kimi-k-direct-stacked-mmap-effective-gate/`
  enabled `MLXPRESS_STREAMING_DIRECT_STACKED=1` and fed whole stacked mmap
  tensors into the existing kernels. Output stayed coherent and peak footprint
  stayed 0.5%, but decode fell to 0.022 tok/s and effective read pressure
  failed at 969,853.44 MB/generated-token due page-ins. Do not promote this
  path; build the offset-descriptor kernel instead.

### Family Policy Layer

Status: partial.

Needed helper:

- `MLXPressActiveExpertFamilyPolicy`

Created behavior:

- Active-expert fallback reads are no longer unconditionally `F_NOCACHE`.
  `MLXPRESS_STREAMING_F_NOCACHE` can force the policy, and the default auto
  policy uses normal OS-cached `pread` when the indexed routed JANGTQ byte set
  fits under 70% of physical memory. Kimi-class oversized routed sets keep
  `F_NOCACHE` by default.

Current proof:

- `docs/local/model-validation/20260513T061054Z-minimax-small-streaming-oscache-hadamard/`
  selected `policy=os-cache` for MiniMax-Small (`streamable=32.81GB`,
  threshold 89.60GB). Coherence and low RAM passed, but read pressure and speed
  did not: 1.82 tok/s avg and 5,913 MB/generated-token. This proves the policy
  fix is necessary but not sufficient; the fallback still rebuilds too many
  active expert banks per token.
- New metrics rows must include Activity Monitor footprint and MLX allocator
  active/cache/peak bytes together. This is required before changing Metal
  external-buffer policy because MiniMax resident rows need to distinguish
  process footprint from MLX's own active buffer accounting.
- MiniMax non-stacked JANGTQ resident rows no longer have to build full
  `switch_mlp` banks. `MLXPRESS_RESIDENT_EXPERTS=1` registers the per-expert
  routed tensors into `JANGTQStreamingExpertStore`, and the focused
  `MiniMaxJANGTQResidentExpertTests` guard verifies sanitize does not leave
  per-expert staging keys or materialize `switch_mlp` banks. Fresh evidence:
  `docs/local/model-validation/20260513T090900Z-minimax-small-resident-experts-load-footprint/`
  still reports 35.2 GB post-load Activity Monitor footprint because the whole
  routed set has live mmap-backed Metal buffers.
- The tensor-span mmap diagnostic now confirms the same root cause from the
  opposite direction:
  `docs/local/model-validation/20260513T083103Z-minimax-small-jpreg-load-footprint-tensor-span-mmap/`
  loaded with `MLXPRESS_MMAP_TENSOR_BUFFERS=1`, post-load RSS 0.6 GB, post-load
  Activity Monitor footprint 35.0 GB, and only 1.3 GB of live mmap-tracked
  Metal buffers. The C++ tensor-span mmap path is therefore real and
  GPU-usable, but that old row was still before the resident-expert MiniMax
  sanitize fix.
- Cold advice is now env-selectable through
  `MLX_SAFETENSORS_MMAP_COLD_ADVICE=dontneed|pageout|force`. `dontneed` stays
  the default; `pageout` is a diagnostic; `force` uses the old JangPress
  `msync(MS_INVALIDATE | MS_ASYNC)` mechanism. A small GPU refault test passes
  after `force`, but full MiniMax resident load-only rows with `pageout` and
  `force` both remained at 35.2 GB footprint. This means cold advice alone is
  not enough when every routed tensor already owns a live Metal buffer.
- Short resident generation
  `docs/local/model-validation/20260513T093159Z-minimax-small-resident-experts-short-generation/`
  produced coherent `Green` / `Blue` with cache stack on and token/s recorded,
  but still failed the Activity Monitor gate at 35.25 GB / 95.2% peak and
  decoded one-token turns at 0.25 then 0.98 tok/s. The no-full-bank resident
  wiring is therefore semantically viable, but not low-RAM or fast enough.
- MiniMax streaming with an 8 GB active tensor cache
  (`docs/local/model-validation/20260513T070926Z-minimax-small-streaming-8gb-tensor-cache/`)
  passed low-RAM and read-pressure gates, but was slower than uncached
  streaming. That narrows the next scheduler/kernel task: reduce per-token
  router readback and active-bank construction, not just file reads.
- The same row with `MLXPRESS_STREAMING_EVAL_EACH_LAYER=0`
  (`docs/local/model-validation/20260513T071450Z-minimax-small-streaming-8gb-no-layer-eval/`)
  improved decode only from 1.17 to 1.21 tok/s, so eval-boundary tuning is
  secondary. The real fix needs a resident/offset active-expert dispatch that
  avoids CPU router readback and repeated temporary bank construction.
- New bounded eval scheduler: `MLXPRESS_STREAMING_EVAL_LAYER_STRIDE` /
  `JANGPRESS_STREAMING_EVAL_LAYER_STRIDE` keeps
  `MLXPRESS_STREAMING_EVAL_EACH_LAYER=0` as the full no-eval diagnostic, but
  lets runtime rows evaluate every Nth routed layer. This gives MiniMax/Kimi/Hy3
  a measurable middle point between per-layer materialization overhead and
  unbounded graph residency.
- MiniMax stride-4 row
  `docs/local/model-validation/20260513T124000Z-minimax-small-offset-kernels-eval-stride4-short/`
  rejected that middle point for the current offset path: peak Activity Monitor
  stayed low at 2.52 GB / 6.8% and explicit reads stayed zero, but aggregate
  decode was 0.909 tok/s and turn 1 length-stopped. The skipped work moved into
  `router.scores_eval` (63.2 s, 69.1 ms/call) and MLX allocator peak rose to
  114.57 GB. Eval-boundary tuning is therefore not the MiniMax speed fix.

Responsibilities:

- Kimi/DeepSeek MLA: account for partial-RoPE attention, 64 KV heads, shared
  expert MLP, and stacked JANGTQ_K.
- MiniMax: preserve mixed-bit gate/up 2-bit and down 4-bit, sigmoid+bias
  routing, and XML reasoning/tool parser behavior.
- Hy3/Hunyuan: keep MTP off for base decode, preserve mixed bits, and avoid
  full K overlay residency.
- ZAYA: include split stacked tensor names and CCA companion-state cache rules.
- Qwen3.6/Qwen VL: respect MRoPE vectors, media cache identity, and hybrid
  companion-state rules.

Why:

One generic scheduler cannot safely ignore attention and cache topology. The
shared mechanism should be generic, but policy must be family-aware.

### Watcher And Hard Guards

Status: partial.

Current guards:

- Activity Monitor gate.
- System pressure deltas in `metrics.jsonl`.
- File-read pressure gate.

Needed helper:

- `MLXPressRuntimePressureWatcher`

Responsibilities:

- Watch Activity Monitor footprint, page-in rate, swap-in/out rate, and
  file-read MB/token during generation.
- Force eviction or disable prefetch when pressure approaches the gate.
- Emit a clear JSONL reason when it changes policy.

Why:

MLXPress must adapt to users with less RAM. The scheduler should not require a
fixed cache size that works only on this machine.

## Kimi First Milestones

1. Add `MLXPressActiveExpertTrace` and record reuse stats for the existing
   Kimi stacked path without changing behavior. Done.
2. Run a short Kimi row with trace + file-read gate report-only and document
   per-layer/expert reuse. Done.
3. Add budgeted active-slice residency using the trace. Start with a very small
   budget and prove Activity Monitor does not regress. Diagnostic done:
   512 MB gets no hits; 8 GB gets 21.5% hits but no speed gain.
4. Prototype direct stacked-offset JANGTQ dispatch for Kimi gate/up/down
   packed tensors. The whole-tensor mmap prototype is done and rejected; the
   next version must avoid touching unrelated expert pages.
5. Run the same prompt set:
   - one-token exact answer
   - six-token non-loop row
   - multi-turn no-thinking row
   - thinking-on parser row
   - cold/warm cache-stack row
6. Promote only if file-read MB/token drops, decode tok/s rises, Activity
   Monitor stays low, and coherency holds.

## Current Kimi Baseline To Beat

Use these as the first regression targets:

- `docs/local/model-validation/20260513T041500Z-kimi-k-system-pressure-profile/`
  - coherent `red`
  - decode 0.119 tok/s
  - peak 1.77 GB / 0.5%
  - 20,190.94 MB active expert reads per generated token
  - 24,790.98 MB page-ins
  - 0 MB swap-out
- `docs/local/model-validation/20260513T042500Z-kimi-k-file-read-gate-report/`
  - coherent `red`
  - file-read gate failed at 20,190.94 MB/generated-token against a
    1,000 MB/generated-token diagnostic threshold
- `docs/local/model-validation/20260513T043500Z-kimi-k-active-expert-trace/`
  - coherent `red`
  - trace calls 180, routed slots 1,440, immediate reuse rate 0.215
  - confirms simple active residency can only remove a minority of reads
- `docs/local/model-validation/20260513T043713Z-kimi-k-active-slice-residency-8192mb/`
  - coherent `red`
  - 8 GB residency budget reduced file reads to 15,858.30 MB/generated-token
  - decode stayed 0.117 tok/s and peak rose to 11.35 GB / 3.5%
  - still blocked by the file-read gate and decode speed
- `docs/local/model-validation/20260513T045340Z-kimi-k-sidecar-bankload-effective-gate-warm/`
  - sidecar present and valid for Kimi JANGTQ_K
  - coherent `red`
  - decode 0.117 tok/s
  - explicit and effective read gate failed at 20,190.94 MB/generated-token
- `docs/local/model-validation/20260513T045423Z-kimi-k-direct-stacked-mmap-effective-gate/`
  - coherent `red`
  - decode 0.022 tok/s
  - explicit read gate passed falsely at 0 MB/generated-token
  - effective read gate failed at 969,853.44 MB/generated-token from page-ins

## Current Kimi Mach Residency Evidence

- Chunked direct-fill into `MLXPressMachCache` is a real wiring fix. Before
  chunking, full Kimi routed descriptors could fail registration with
  `mmap failed: errno=22`; after chunking, the one-layer row
  `docs/local/model-validation/20260513T212055Z-kimi-k-fullspan-prewarm-1layer-after-chunked-pread/`
  registered all six spans / 5.26 GB and generated coherent `red`.
- The high-priority volatile cold state plus chunked refault fixed the
  eight-layer acquire failure in
  `docs/local/model-validation/20260513T212755Z-kimi-k-fullspan-prewarm-8layer-after-coldstate-refault/`.
  That row still decoded at only 0.303 tok/s, so it is a residency-wiring
  proof, not a speed proof.
- `MLXPRESS_MACH_RELEASE_POLICY=pageout` is a dead end for this path. The
  Kimi pageout row decoded at 0.136 tok/s and raised footprint versus the
  default volatile/LIFO release policy.
- Partial full-span prewarm with no active trace/readback removes explicit
  `pread` counters but not OS page pressure. The 8-layer no-readback eval row
  had 0 MB explicit reads and still failed effective read pressure at
  901,350.47 MB/generated-token because 52 non-prewarmed layers fell back to
  file-backed mmap spans.
- Full 60-layer eager prewarm is also rejected:
  `docs/local/model-validation/20260513T214811Z-kimi-k-fullspan-60layer-no-readback-eval-generation/`
  prewarmed 360 spans / 315.48 GB in 454.36 s and kept post-load footprint to
  a 416.2 MB increase, but load pressure recorded 341.40 GB page-ins and
  78.57 GB swapouts. Generation then failed the peak Activity Monitor gate at
  98.65 GB / 30.1% before producing a token; the row was interrupted at
  `turns=0`, `gen_tokens=0`, `decode_tps=0.000`. The next Kimi scheduler
  direction must be selective/predictive residency or a lower-level persistent
  offset-kernel path, not eager registration of the full routed set.

## Do Not Do

- Do not make active-slice `pread` streaming the default success path.
- Do not load the whole stacked expert bank into anonymous MLX memory.
- Do not call full routed-span Mach prewarm production-ready from post-load
  footprint alone; the 60-layer Kimi row failed load pressure, peak footprint,
  and token/s.
- Do not force OS-cached reads for oversized bundles just to hide explicit file
  reads. Always gate effective page-in pressure too.
- Do not create permanent prestacked overlays as the normal path.
- Do not call a row ready from coherent output alone.
- Do not optimize Kimi by breaking reasoning parser, tool parser, cache hit
  correctness, or VL/media cache identity.
