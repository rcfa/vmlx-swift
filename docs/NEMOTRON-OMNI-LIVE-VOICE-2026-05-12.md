# Nemotron Omni Live Voice Status - 2026-05-12

## Scope

This note tracks the current live voice path for agentic call-style input:
Nemotron Omni audio input, Parakeet/RADIO encoder handling, prefix-cache
isolation, Osaurus handoff requirements, and measured local behavior.

2026-05-17 consolidation update: the live voice bench executables described
here now live in this package as `OmniAudioLatencyBench` and
`OmniAudioChunkStabilityBench`. The original 2026-05-12 measurements below are
kept as historical evidence; current `vmlx-swift` release-built evidence is
tracked in `docs/VMLX_ACTIVE_MODEL_PRODUCTION_SCOPE_2026_05_17.md`.

## Implemented

- `UserInput.Audio.preEncoded(samples:sampleRate:embedding:)` exists for live
  voice handoff when another component has already produced Parakeet/sound
  projection embeddings.
- `NemotronHOmniProcessor` preserves pre-encoded audio embeddings while still
  carrying the waveform for media salt and fallback behavior.
- `NemotronHOmni.prepare(_:)` uses `audio.preEncodedEmbedding` when present and
  otherwise runs the mel + Parakeet + sound projection encoder path.
- The Python vMLX Omni dispatcher hashes user media bytes/paths into its
  session signature. Same text plus different prior audio/image/video now
  resets the Omni session instead of reusing stale media-conditioned state.
- Osaurus now retains raw live voice PCM separately from the STT worker drain
  and attaches a WAV to voice sends when the selected model advertises audio.

## Bench Evidence

Command:

```sh
BENCH_OMNI=1 \
BENCH_MAX_TOKENS=8 \
BENCH_MODEL=<local Nemotron-Omni-Nano-JANGTQ-CRACK> \
/usr/bin/time -l swift run RunBench
```

Artifact:

```text
build/evidence-20260512/nemotron_omni_live_voice_jangtq2_clean_20260512_155446.log
```

Inputs:

- Model: local `Nemotron-Omni-Nano-JANGTQ-CRACK` bundle, `weight_format=mxtq`,
  `mxtq_bits=2`, `config_omni.json` present.
- Audio fixture: `Tests/MLXLMTests/Resources/audio_only.mov`.

Result summary:

| Row | Result | Time | Throughput |
|---|---:|---:|---:|
| text-only single-turn | PASS | 0.23s | 61.5 tok/s |
| text-only multi-turn x3 | PASS | 0.55s | 62.1 tok/s |
| image single-turn | PASS | 1.57s | 43.5 tok/s |
| image multi-turn x2 | PASS | 3.01s | 44.5 tok/s |
| video encoder smoke | PASS | 0.33s | n/a |
| audio encoder smoke | PASS | 0.29s | n/a |
| video LMInput end-to-end | PASS | 15.22s | 13.1 tok/s |
| audio LMInput end-to-end | PASS | 1.56s | 52.3 tok/s |
| reasoning OFF | PASS | 0.19s | 60.2 tok/s |
| reasoning ON/OFF/ON toggle | PASS | 0.52s | 57.5 tok/s |
| mixed image + audio | PASS | 2.88s | 41.0 tok/s |
| media-salt isolation audio A vs B | PASS | 2.18s | 56.0 tok/s |
| hybrid SSM warm-pass parity | PASS | 0.37s | 61.1 tok/s |

Overall: `13 passed, 0 failed`, `bench_exit=0`, load time `2.37s`.

Memory:

- System memory free before: 93%.
- System memory free after: 93%.
- Maximum resident set size: `7,740,358,656` bytes.
- Peak memory footprint reported by `/usr/bin/time -l`: `79,949,715,888`
  bytes.
- Swaps during command: `0`.

## Latency Interpretation

This bench proves real audio input encoding/splice/generation and multi-turn
cache behavior. It is not yet a TTFAB benchmark because it does not synthesize
or stream output audio.

Current useful input-side measurements:

- Parakeet encoder smoke for the fixture: `0.29s`.
- Full audio `LMInput` end-to-end row with 8-token decode: `1.56s`.
- Mixed image + audio row with 8-token decode: `2.88s`.

For live calls, the `preEncoded` path is important because it can remove the
mel + Parakeet encoder cost from the model turn if a streaming voice component
has already produced embeddings. The remaining unresolved latency question is
not "can Omni consume audio" but "can we produce stable streaming audio
embeddings and TTS first audio fast enough for natural turn-taking."

## Remaining Hookups

- Add a true TTFAB bench: audio input clip -> model first visible token ->
  TTS first audio byte. Record VAD endpoint delay, model first token, TTS
  first audio byte, and total wall time separately.
- Add an Osaurus app-level live voice trace that logs:
  `voice_snapshot_ms`, `wav_bytes`, `input_audio_materialize_ms`,
  `prompt_prepare_ms`, `first_token_ms`, and response stream first chunk.
- Add a direct `preEncoded` route from an embedding-producing voice component
  into `UserInput.Audio.preEncoded`. The current Osaurus patch sends a WAV
  attachment, which is correct but still pays model-side audio encoding.
- Keep prefix/cache keys media-aware. Any future embedding-only path must hash
  the original waveform or a stable embedding digest in addition to text.
- Add TTS TTFA/TTFAB measurement. Nemotron Omni does not include a neural TTS
  decoder in this runtime; call output speech must be handled by a separate TTS
  backend.
- Fix or route around local Swift test-runner blockers before claiming unit
  tests are green:
  - `vmlx-swift-lm`: `swift test` currently fails on `no such module 'XCTest'`.
  - `osaurus-staging`: `swift test` currently fails on `no such module 'Testing'`.

## Library Boundary

- `vmlx-swift-lm`: code changes are required and implemented for pre-encoded
  audio and Omni prepare.
- `vmlx_engine` Python: code changes are required and implemented for
  media-aware Omni session signatures and `audio/mp4 -> .m4a`.
- `osaurus-staging`: code changes are required and implemented for retaining
  live voice PCM and attaching WAV audio to audio-capable voice sends.
- `swift-jinja`, `swift-transformers`, and `mlx-swift`: no direct code changes
  were required in this pass. Existing package pins should still be checked
  before PR merge.
