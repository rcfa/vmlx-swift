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

## Authored PR Scope

Live GitHub inspection on 2026-05-17 found these user-authored Osaurus PRs in
the May 1-May 17 runtime window. Merged PRs define package resolver lineage;
closed/unmerged PRs are context only.

| PR | State | Head / merge inspected | Why it matters for `vmlx-swift` |
| --- | --- | --- | --- |
| #967 `feat: Nemotron-3 Hybrid + storage fix + multimodal API + Bug 1/3a stability fixes` | merged | head `18eace7`; merge `01a1194` | First window where Nemotron hybrid, multimodal API, storage, `mlx-swift`, Jinja, swift-transformers, and `vmlx-swift-lm` pins were all moving. Also shows early app/workspace resolver skew. |
| #990 `feat(api): OpenAI input_audio + video_url content parts -> vmlx audios/videos` | closed/unmerged | head `e090726` | API-surface context only. Do not treat it as shipped resolver state. |
| #993 `fix(preflight): reject JANGTQ Mistral 3 / Laguna before vmlx loads` | merged | head `3a297e`; merge `8fab9f4` | Preflight/fail-fast runtime policy plus converged Jinja package identity. |
| #998 `fix(quality): revert default KV mode .turboQuant(4,4) -> .none - fixes degenerate-repetition looping` | merged | head `642c158`; merge `5eefccc` | Explicit warning against fake global KV defaults: quality fix was restoring native/default cache semantics, not hiding bad output behind sampler guards. Also shows app/workspace `vmlx-swift-lm` skew. |
| #1037 `Bump vmlx-swift-lm to b9da180 (Ling/ZAYA hardening + BatchEngine lifecycle)` | merged | head `4ea3488`; merge `f083d46` | Ling/Bailing, ZAYA, BatchEngine lifecycle, and topology-aware cache hardening. |
| #1057 `feat(runtime): bump vmlx-swift-lm to cb8b3df + MiniMax speed fix` | merged | head `70287ee`; merge `7e963ce` | MiniMax speed/lifecycle plus parser/tokenizer/Jinja compatibility. Title SHA does not match final resolver SHA. |
| #1066 `chore(runtime): pin DSV4 vmlx update` | merged | head `91b72e4`; merge `b52e0a7` | DSV4 tokenizer/cache/runtime pin work. |
| #1073 `Nemotron Omni live voice input path` | merged | head `e2723b0`; merge `27f3573` | Omni live voice path, Parakeet/RADIO/media-cache token-aware restore, and DSV4 pool/compressor fixes. |
| #1110 `Harden DSV4 reasoning gates and runtime proof` | open | head `b0a96dd` | Current DSV4 proof branch. Not merged release state; useful for the next Osaurus switch PR readiness bar. |

## Resolver Pins

| Osaurus PR | State / commit inspected | vmlx-swift-lm resolved revision | Other runtime pins | What vmlx-swift must carry forward |
| --- | --- | --- | --- | --- |
| #967 `feat: Nemotron-3 Hybrid + storage fix + multimodal API + Bug 1/3a stability fixes` | merged, merge `01a1194` | App: `a196800`; workspace: `a7db6e5` | App: `mlx-swift 0a56f904`, `swift-jinja 0aeefad`, `swift-transformers b38443e`; workspace: `mlx-swift 02b01f0`, `swift-jinja 0aeefad`, `swift-transformers b38443e` | Early resolver skew means the app and workspace may have exercised different engine commits. Use this as a warning that Osaurus must pin the consolidated package from one resolver path only. |
| #993 `fix(preflight): reject JANGTQ Mistral 3 / Laguna before vmlx loads` | merged, merge `8fab9f4` | `89f8114d9a7bcbfd0ffde5b989e3f3cc76cfe2b3` | `mlx-swift 0a56f904`; `Jinja 58d21aa`; `swift-transformers b38443e` | Preflight/load-policy behavior and converged Jinja package identity. |
| #998 `fix(quality): revert default KV mode .turboQuant(4,4) -> .none - fixes degenerate-repetition looping` | merged, merge `5eefccc` | App: `2e61c12`; workspace: `89f8114` | Both: `mlx-swift 0a56f904`; `Jinja 58d21aa`; `swift-transformers b38443e` | Never restore hidden global KV defaults as a quality patch. The consolidated engine must use explicit settings, model metadata, and real cache compatibility checks. |
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

## Dependency Commit Windows

The resolver lineage above implies these upstream commit windows that the
single `vmlx-swift` package must subsume or vendor:

| Dependency | Window inspected | Commit payload |
| --- | --- | --- |
| `osaurus-ai/mlx-swift` | `02b01f0...0a56f904` | 2 commits: fork submodule switch and `mx::malloc` tracer / Bug 2 support. Current `vmlx-swift` uses the local Cmlx/MLX submodule checkout rather than a SwiftPM resolver entry, so Osaurus switch validation must compare behavior, not package identity alone. |
| `osaurus-ai/Jinja` | `0aeefad...58d21aa` | 1 commit: for-loop iterable accepts binary expressions. Vendored Jinja now also has `tojson(separators:)` for compact JSON tool schemas. |
| `osaurus-ai/swift-transformers` | `b38443e...087a66b` | 2 commits: use Osaurus Jinja and skip unused placeholder added tokens in delimiter regex. The vendored tokenizer source skips `<unusedN>` added tokens before building `addedTokensRegex`. |
| `osaurus-ai/vmlx-swift-lm` | `a196800...2cc64dd` | 181 commits across Mistral3/Laguna, MiniMax, Ling/Bailing, ZAYA, Gemma4, Hy3, DSV4, Omni, cache I/O, media salt, reasoning/tool parsers, and live diagnostics. The current `vmlx-swift` docs/tests cover many of those surfaces, but several live model rows remain open before a default Osaurus switch. |

## Vendored Parity Checks

Current consolidated-package status for the split dependency fixes:

| Surface | Status in `vmlx-swift` | Evidence |
| --- | --- | --- |
| Jinja `tojson(separators=(',', ':'))` | implemented and tested | `DeepseekV4ChatTemplateFallbackFocusedTests.tojsonAcceptsPythonSeparatorsKwarg` |
| Jinja binary expressions in `for` iterables | fixed in this pass and tested | `DeepseekV4ChatTemplateFallbackFocusedTests.forLoopIterableAcceptsBinaryExpression` |
| Jinja loop `if` filter after the binary-expression fix | tested to avoid regression | `DeepseekV4ChatTemplateFallbackFocusedTests.forLoopIfClauseRemainsLoopFilter` |
| swift-transformers unused placeholder skip | static-present in vendored tokenizer | `Vendors/swift-transformers/Sources/Tokenizers/Tokenizer.swift` filters `isUnusedPlaceholderAddedToken` before regex build |
| Focused artifact | passed | `docs/local/production-readiness/20260517T2200_jinja_pin_parity/deepseek_v4_jinja_fallback.log` passes 10/10 rows under full Xcode toolchain |

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
