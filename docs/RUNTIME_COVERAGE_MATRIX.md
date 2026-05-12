# Runtime coverage matrix

This matrix defines what must be checked before Osaurus can treat this repo as
the single Swift runtime dependency.

## Architecture buckets

| Bucket | Model families to cover | Engine concerns | Osaurus concerns |
|---|---|---|---|
| Dense KV | Qwen dense, Mistral dense, Llama-style dense | EOS/stop tokens, generation_config, compiled decode, paged KV | model defaults, streaming stats, stop/cancel UI |
| Dense MoE | Qwen3.5/3.6 MoE, Laguna, Gemma MoE | top-k override, compiled router, routed expert dispatch, SwitchGLU variants | server knobs, generation defaults, decode speed regression |
| Sliding/rotating KV | Gemma 4 | RotatingKVCache, PLE config, image-token path for VLM variants | load errors, VLM detection, media gating |
| Hybrid SSM/Mamba | MiniMax, Nemotron-H, Jamba, FalconH1 | MambaCache/CacheList, SSM companion disk store, rederive/full-hit safety | reasoning/tool parser stamps, prefix cache logs, stop button terminal state |
| Linear attention | Ling/Bailing hybrid, LFM-style | ArraysCache restore, path-dependent cache gating, compile safety | Ling non-reasoning policy, prefix cache reporting |
| CCA | ZAYA text/VL | ZayaCCACache, CCA state restore, JANGTQ4/MXFP4/JANGTK correctness | reasoning toggle, text vs VL media path, TTFT attribution |
| MLA/compressor | DSV4, Nemotron compressor variants | MLA/compressor state, JANGTQ-K routed bit plan, stop tokens | reasoning parser, cache restore, coherent UI output |
| Image VL | ZAYA1-VL, Qwen VL, Gemma VLM, Pixtral/Mistral VL | image token IDs, media salt, vision tower dtype/pass-through | drag/drop gating, image content mapping |
| Video VL | Qwen VL, SmolVLM, omni | video URL/materialization, frame extraction, media salt | content part mapping, large payload behavior |
| Audio/omni | Nemotron-Omni | audio URL/materialization, resample path, Parakeet/audio encoder | input_audio mapping, format canonicalization |
| Tool calling | MiniMax, Qwen, Mistral, Laguna, Gemma | parser stamps, content/reasoning rail routing, tool result turns | OpenAI tool_calls, tool_call_id, UI/tool loop |

## Per-family proof checklist

For every family above, record:

- Model path and exact bundle config files used.
- vmlx SHA, mlx-swift SHA, swift-transformers SHA, Jinja SHA.
- Whether generation_config was loaded and which fields applied:
  temperature, top_p, top_k, min_p, repetition_penalty, max_new_tokens.
- Reasoning parser source: JANG stamp, model_type heuristic, or none.
- Tool parser source: JANG stamp, model_type heuristic, or none.
- Cache type created by `newCache`.
- Cache coordinator tier hit or miss for turn 1, turn 2, and turn 3.
- Prompt tokens, promptMs, TTFT, decode tok/s, stop reason, generated tokens.
- Peak resident memory during load and after first token.
- Whether output was coherent and stopped naturally.

## Standard turns

Use the same conversation shape for every model where possible:

1. `hi`
2. `what did I just say?`
3. `answer in one short sentence and do not use tools`
4. stop/cancel mid-generation

For reasoning-capable families, also run:

1. thinking off: `hi`
2. thinking on: `hi`
3. thinking on follow-up after prior reasoning: `what did I ask?`

For media families, also run:

1. text only
2. one image
3. same image on next turn
4. different image on next turn
5. one unsupported media type to verify rejection

## DSV4-specific checks

DSV4 and DSV4 Flash JANGTQ-K must be validated as their own bucket, not inferred
from Qwen or MiniMax:

- JANGTQ-K routed bit plan is preserved.
- MLA/compressor cache state restores without cross-turn corruption.
- EOS/stop tokens include the family-specific sentence-end token.
- Reasoning parser opens/closes correctly under the DSV4 template.
- Prompt cache hit is demonstrated across a growing chat.
- Direct engine smoke and Osaurus app/API smoke both produce coherent output.

## Failure classification

Every failure should be labeled as one of:

- `load`: config, safetensors, mmap, JANG sidecar, memory cap
- `prepare`: tokenizer, Jinja, content part mapping, media processor
- `prefill`: cache restore, RoPE position, compile path, media salt
- `decode`: router, top-k, sampling defaults, stop tokens, parser rail
- `ui`: stop button, reasoning pane, streaming stats, attachment gating
- `package`: wrong SHA, local path, stale URL, missing pushed dependency

Do not merge a runtime PR with an unlabeled failure. If a failure is out of
scope, record the exact scope decision and the model families it affects.

