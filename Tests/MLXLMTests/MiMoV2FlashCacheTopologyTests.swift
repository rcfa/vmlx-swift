import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
import Testing

@Suite("MiMo V2.5 hybrid full/SWA cache topology")
struct MiMoV2FlashCacheTopologyTests {
    private static let minimalMiMoV25Config = #"""
    {
      "model_type": "mimo_v2",
      "num_experts_per_tok": 2,
      "hybrid_layer_pattern": [0, 1, 1, 0],
      "moe_layer_freq": [0, 1, 1, 1],
      "add_swa_attention_sink_bias": true,
      "add_full_attention_sink_bias": false,
      "sliding_window_size": 128,
      "vocab_size": 256,
      "hidden_size": 32,
      "intermediate_size": 64,
      "moe_intermediate_size": 16,
      "num_hidden_layers": 4,
      "num_attention_heads": 8,
      "num_key_value_heads": 4,
      "n_shared_experts": 1,
      "n_routed_experts": 4,
      "routed_scaling_factor": 1.0,
      "topk_method": "noaux_tc",
      "scoring_func": "sigmoid",
      "norm_topk_prob": true,
      "n_group": 1,
      "topk_group": 1,
      "max_position_embeddings": 4096,
      "layernorm_epsilon": 1e-6,
      "rope_theta": 10000000,
      "swa_rope_theta": 10000,
      "swa_num_attention_heads": 8,
      "swa_num_key_value_heads": 8,
      "head_dim": 4,
      "v_head_dim": 4,
      "swa_head_dim": 4,
      "swa_v_head_dim": 4,
      "partial_rotary_factor": 0.5,
      "attention_value_scale": 0.707
    }
    """#

    @Test("model_type=mimo_v2 dispatches to MiMoV2FlashModel")
    func modelTypeAliasDispatches() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(Self.minimalMiMoV25Config.utf8),
            modelType: "mimo_v2")

        #expect(model is MiMoV2FlashModel)
    }

    @Test("per-layer KV heads and cache classes match full/SWA layer pattern")
    func perLayerKVHeadsAndCacheTopologyFollowHybridPattern() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))
        let model = MiMoV2FlashModel(config)

        #expect(model.modelType == "mimo_v2")
        #expect(model.kvHeads == [4, 8, 8, 4])

        let cache = model.newCache(parameters: GenerateParameters(
            maxTokens: 8,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3)))
        #expect(cache.count == 4)
        #expect(cache[0] is KVCacheSimple)
        #expect(cache[1] is RotatingKVCache)
        #expect(cache[2] is RotatingKVCache)
        #expect(cache[3] is KVCacheSimple)
        #expect(!cache.contains { $0 is TurboQuantKVCache })
    }

    @Test("configuration decodes MiMo V2.5 attention value scale")
    func configurationDecodesAttentionValueScale() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(Self.minimalMiMoV25Config.utf8))

        #expect(config.attentionValueScale == 0.707)
    }

    @Test("JANGTQ config builds TurboQuant switch MLP and drops tq_bits metadata")
    func jangtqConfigBuildsTurboQuantSwitchMLPAndDropsTQBitsMetadata() throws {
        let configJSON = Self.minimalMiMoV25Config.replacingOccurrences(
            of: #""attention_value_scale": 0.707"#,
            with: """
      "attention_value_scale": 0.707,
      "weight_format": "mxtq",
      "mxtq_bits": {
        "routed_expert": {
          "gate_proj": 2,
          "up_proj": 2,
          "down_proj": 4
        }
      },
      "mxtq_seed": 7
""")
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(configJSON.utf8))
        #expect(config.weightFormat == "mxtq")
        #expect(config.usesTurboQuantRoutedExperts)
        #expect(config.routedExpertQuantization(layerIndex: 1, projection: "gate_proj").bits == 2)
        #expect(config.routedExpertQuantization(layerIndex: 1, projection: "up_proj").bits == 2)
        #expect(config.routedExpertQuantization(layerIndex: 1, projection: "down_proj").bits == 4)
        let model = MiMoV2FlashModel(config)

        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let gate = try #require(
            leaves["model.layers.1.mlp.switch_mlp.gate_proj"] as? TurboQuantSwitchLinear)
        let up = try #require(
            leaves["model.layers.1.mlp.switch_mlp.up_proj"] as? TurboQuantSwitchLinear)
        let down = try #require(
            leaves["model.layers.1.mlp.switch_mlp.down_proj"] as? TurboQuantSwitchLinear)

        #expect(gate.bits == 2)
        #expect(up.bits == 2)
        #expect(down.bits == 4)

        let sanitized = model.sanitize(weights: [
            "model.layers.1.mlp.switch_mlp.gate_proj.tq_packed": MLXArray.ones(
                [4, 2, 2], dtype: .uint32),
            "model.layers.1.mlp.switch_mlp.gate_proj.tq_norms": MLXArray.ones(
                [4, 2, 1], dtype: .float32),
            "model.layers.1.mlp.switch_mlp.gate_proj.tq_bits": MLXArray(Int32(2)),
            "model.layers.1.mlp.switch_mlp.up_proj.tq_packed": MLXArray.ones(
                [4, 2, 2], dtype: .uint32),
            "model.layers.1.mlp.switch_mlp.up_proj.tq_norms": MLXArray.ones(
                [4, 2, 1], dtype: .float32),
            "model.layers.1.mlp.switch_mlp.up_proj.tq_bits": MLXArray(Int32(2)),
            "model.layers.1.mlp.switch_mlp.down_proj.tq_packed": MLXArray.ones(
                [4, 2, 2], dtype: .uint32),
            "model.layers.1.mlp.switch_mlp.down_proj.tq_norms": MLXArray.ones(
                [4, 2, 1], dtype: .float32),
            "model.layers.1.mlp.switch_mlp.down_proj.tq_bits": MLXArray(Int32(4)),
            "audio_encoder.input_local_transformer.layers.0.input_layernorm.weight":
                MLXArray.ones([32], dtype: .float32),
            "encoder.layers.0.self_attn.q_proj.weight": MLXArray.ones([32, 32], dtype: .float32),
            "speech_embeddings.0.weight": MLXArray.ones([16, 32], dtype: .float32),
            "visual.blocks.0.attn.qkv.weight": MLXArray.ones([32, 32], dtype: .float32),
        ])

        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.tq_packed"] != nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.tq_norms"] != nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.tq_bits"] == nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.up_proj.tq_bits"] == nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.down_proj.tq_bits"] == nil)
        #expect(
            sanitized["audio_encoder.input_local_transformer.layers.0.input_layernorm.weight"]
                == nil)
        #expect(sanitized["encoder.layers.0.self_attn.q_proj.weight"] == nil)
        #expect(sanitized["speech_embeddings.0.weight"] == nil)
        #expect(sanitized["visual.blocks.0.attn.qkv.weight"] == nil)
    }
}
