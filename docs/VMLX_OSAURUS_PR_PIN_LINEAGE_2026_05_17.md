# vMLX Swift / Osaurus PR Pin Lineage - 2026-05-17

This note records the resolver truth for recent user-authored Osaurus PRs that
matter to the `vmlx-swift` consolidation work. It intentionally separates PR
titles and intermediate bump commits from the package revisions that were
actually resolved at the merge or current PR head.

Source commands used for this pass:

```sh
gh pr view -R osaurus-ai/osaurus <pr> --json number,title,state,url,headRefOid,mergeCommit,commits,files
git -C /Users/eric/osaurus-staging show <commit>:osaurus.xcworkspace/xcshareddata/swiftpm/Package.resolved
git -C /Users/eric/osaurus-staging show <commit>:App/osaurus.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved
```

## Resolver Pins

| Osaurus PR | State / commit inspected | vmlx-swift-lm resolved revision | Other runtime pins | What vmlx-swift must carry forward |
| --- | --- | --- | --- | --- |
| #1037 `Bump vmlx-swift-lm to b9da180 (Ling/ZAYA hardening + BatchEngine lifecycle)` | merged, merge `f083d46` | `b9da180158365c20a0fab130217e4fa50b8ec674` | `mlx-swift 0a56f904`; `swift-transformers b4a094b`; `Jinja 58d21aa` via old `swift-jinja` identity | Ling/Bailing multi-turn cache fixes, ZAYA text/CCA cache path, BatchEngine lifecycle/fairness/shutdown, Bailing recurrent GLA fused kernel, TaskCoalescer teardown tombstones, Jinja parser-fix mirror policy. |
| #1057 `feat(runtime): bump vmlx-swift-lm to cb8b3df + MiniMax speed fix` | merged, merge `7e963ce` | `78cf6ac9dd1742c51a8f737bd4abe6c68282072e` | `mlx-swift 0a56f904`; `swift-transformers 087a66b`; `Jinja 58d21aa`; `swift-huggingface 0.9.0` | Final resolver is newer than the title. Carry MiniMax streaming/lifecycle fixes, Hy3 reasoning-effort cache salt, ZAYA1-VL detection/hardening, typed load configuration, blank-output handling, terminal stats, and added-token/Jinja compatibility pins. |
| #1066 `chore(runtime): pin DSV4 vmlx update` | merged, merge `b52e0a7` | `ad1d23199b056ed502124717e6ca8877f2fb303a` | `mlx-swift 0a56f904`; `swift-transformers 087a66b`; `Jinja 58d21aa`; `swift-huggingface 0.9.0` | DSV4 fallback/tokenizer wiring, solo cache gate, swift-transformers tokenizer fallback, and runtime policy tests for local tokenizer loading. |
| #1073 `Nemotron Omni live voice input path` | merged, merge `27f3573` | `6561a72f93d6cd5e0202e8067b53fed5cf21a660` | `mlx-swift 0a56f904`; `swift-transformers 087a66b`; `Jinja 58d21aa`; `swift-huggingface 0.9.0` | Omni live voice PCM/pre-encoded audio path, Parakeet/RADIO/EVS rows, media-cache token-aware restore, DSV4 causal pool and overlap-compressor fixes, and live voice TTFT trace fields. |
| #1110 `Harden DSV4 reasoning gates and runtime proof` | open, head `b0a96dd49a91fd646d7e229ba15a26aa0343d428` | `2cc64dd30f9faa877d4c5ecced63ab4ac9467df4` | `mlx-swift 0a56f904`; `swift-transformers 087a66b`; `Jinja 58d21aa`; `swift-huggingface 0.9.0` | Native DSV4 chat encoder/tokenizer bridge, DSV4 live diagnostics, runtime pin check script, and PR-level release-readiness proof. PR was reported `DIRTY` by GitHub at inspection time, so do not treat it as merged release state. |

## Constant Pins To Preserve

The later PRs converge on this runtime chain:

| Package | Canonical location | Revision |
| --- | --- | --- |
| `mlx-swift` | `https://github.com/osaurus-ai/mlx-swift` | `0a56f9041d56b4b8161f67a6cbd540ae66efc9fd` |
| `Jinja` | `https://github.com/osaurus-ai/Jinja.git` | `58d21aa5b69fdd9eb7e23ce2c3730f47db8e0c9d` |
| `swift-transformers` | `https://github.com/osaurus-ai/swift-transformers` | `087a66b17e482220b94909c5cf98688383ae481a` |
| `swift-huggingface` | `https://github.com/huggingface/swift-huggingface.git` | `b721959445b617d0bf03910b2b4aced345fd93bf` / `0.9.0` |

## Consolidation Implications

`vmlx-swift` is not production-ready for Osaurus just because the package
graph builds. The single package must retain the behavior that these split pins
currently provide:

- Jinja `58d21aa` compatibility for list concatenation in model templates.
- Swift-transformers tokenizer fixes used by MiniMax, DSV4, Qwen, and Omni
  wrapper-token paths.
- MLX/runtime behavior from the pinned `mlx-swift` submodule and Cmlx checkout.
- `vmlx-swift-lm` fixes for Bailing/Ling hybrid cache offsets, ZAYA CCA cache
  state, Gemma4 heterogeneous SWA cache handling, Hy3 mixed q/k/v fallback,
  DSV4 CSA/HSA/SWA state, Qwen hybrid SSM cache repair, and Omni Parakeet/RADIO
  media handling.

Any Osaurus repin to a single `vmlx-swift` dependency should verify this exact
lineage with a resolver check, not by PR title. #1057 is the concrete warning:
its title names `cb8b3df`, while the merge-commit resolver pins `78cf6ac`.
