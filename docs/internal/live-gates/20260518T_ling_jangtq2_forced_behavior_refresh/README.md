# Ling JANGTQ2 Forced-Behavior Refresh

Timestamp: 2026-05-18 15:59 PDT

Purpose: prove the Ling/Bailing runtime path can produce coherent visible
output without hidden sampler guards, forced repetition penalties, forced
reasoning closure, or Osaurus-side reasoning-to-content repair.

Command:

```sh
env BENCH_MODEL=/Users/eric/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK \
    BENCH_NO_GUARD_SAMPLING=1 \
    BENCH_MAX_TOKENS=256 \
    swift run -c release RunBench \
    > docs/internal/live-gates/20260518T_ling_jangtq2_forced_behavior_refresh/no_guard_sampling.log 2>&1
```

Result summary:

- Model loaded as `BailingHybridModel` with `glm4` tool format and
  `deepseek_r1` reasoning stamp.
- Greedy no-repetition row used `temp=0.0`, `topP=1.0`, `topK=0`,
  `minP=0.0`, `rep=nil`, and omitted `enable_thinking`; it stopped normally
  at 37.3 tok/s with no loop, BOS repetition, reasoning leak, or visible marker
  leak.
- Explicit thinking-on row used `temp=0.6`, `rep=1.0`,
  `enable_thinking=true`; it stopped normally at 37.5 tok/s with coherent
  visible output and no forced `</think>` repair.
- Ling Russian stress row used `temp=0.7`, `rep=nil`,
  `enable_thinking=false`; it stopped normally at 38.1 tok/s with no loop, BOS
  repetition, or leak.

Boundary:

This is vmlx runtime proof for the no-hidden-guard Ling row. It does not close
the Osaurus app/API production gate by itself; Osaurus still needs long-output,
cache, SSM companion, memory, saved-setting carryover, and route artifacts.
