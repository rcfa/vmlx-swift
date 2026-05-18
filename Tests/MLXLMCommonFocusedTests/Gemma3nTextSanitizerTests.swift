// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLLM
import Testing

@Suite("Gemma 3n text sanitizer")
struct Gemma3nTextSanitizerTests {
    @Test("conditional generation text weights are remapped to the text-only module")
    func conditionalGenerationTextWeightsAreRemappedToTextOnlyModule() throws {
        let model = try makeTinyModel()
        let sanitized = model.sanitize(weights: [
            "language_model.model.embed_tokens.weight": MLXArray.zeros(
                [12, 4], dtype: .float32),
            "language_model.model.embed_tokens.scales": MLXArray.zeros(
                [12, 1], dtype: .float32),
            "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros(
                [4, 4], dtype: .float32),
            "audio_tower.encoder.weight": MLXArray.zeros([1], dtype: .float32),
            "vision_tower.encoder.weight": MLXArray.zeros([1], dtype: .float32),
            "embed_audio.embedding.weight": MLXArray.zeros([1], dtype: .float32),
            "embed_vision.embedding.weight": MLXArray.zeros([1], dtype: .float32),
        ])

        #expect(sanitized["language_model.embed_tokens.weight"] != nil)
        #expect(sanitized["language_model.embed_tokens.scales"] != nil)
        #expect(sanitized["language_model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(sanitized["language_model.model.embed_tokens.weight"] == nil)
        #expect(sanitized["audio_tower.encoder.weight"] == nil)
        #expect(sanitized["vision_tower.encoder.weight"] == nil)
        #expect(sanitized["embed_audio.embedding.weight"] == nil)
        #expect(sanitized["embed_vision.embedding.weight"] == nil)
        #expect(sanitized["language_model.embed_tokens.weight"]?.dim(0) == 10)
    }

    @Test("fully wrapped text weights strip both model prefixes")
    func fullyWrappedTextWeightsStripBothModelPrefixes() throws {
        let model = try makeTinyModel()
        let sanitized = model.sanitize(weights: [
            "model.language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros(
                [4, 4], dtype: .float32),
            "model.language_model.layers.1.self_attn.k_proj.weight": MLXArray.zeros(
                [4, 4], dtype: .float32),
        ])

        #expect(sanitized["language_model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(sanitized["language_model.layers.1.self_attn.k_proj.weight"] != nil)
        #expect(sanitized["model.language_model.model.layers.0.self_attn.q_proj.weight"] == nil)
    }

    private func makeTinyModel() throws -> Gemma3nTextModel {
        let data = Data(
            """
            {
              "model_type": "gemma3n",
              "hidden_size": 4,
              "num_hidden_layers": 2,
              "intermediate_size": [8, 8],
              "num_attention_heads": 2,
              "head_dim": 2,
              "rms_norm_eps": 0.000001,
              "vocab_size": 10,
              "num_key_value_heads": 1,
              "num_kv_shared_layers": 0,
              "query_pre_attn_scalar": 2.0,
              "vocab_size_per_layer_input": 12,
              "sliding_window": 8,
              "max_position_embeddings": 64,
              "rope_local_base_freq": 10000.0,
              "rope_theta": 1000000.0,
              "final_logit_softcapping": 30.0,
              "layer_types": ["sliding_attention", "full_attention"],
              "activation_sparsity_pattern": [1.0, 1.0],
              "hidden_size_per_layer_input": 4,
              "altup_num_inputs": 1,
              "altup_coef_clip": null,
              "altup_correct_scale": true,
              "altup_active_idx": 0,
              "laurel_rank": 2,
              "rope_scaling": null,
              "sliding_window_pattern": 2
            }
            """.utf8)
        let config = try JSONDecoder().decode(Gemma3nTextConfiguration.self, from: data)
        return Gemma3nTextModel(config: config)
    }
}
