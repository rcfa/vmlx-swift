// Copyright © 2026 Jinho Jang. All rights reserved.

import Foundation
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("DiffusionGemma configuration and registry")
struct DiffusionGemmaConfigurationTests {
    @Test("HF DiffusionGemma config decodes nested text model and canvas fields")
    func decodesHFConfigShape() throws {
        let config = try JSONDecoder.json5().decode(
            DiffusionGemmaConfiguration.self,
            from: Data(Self.minimalConfig.utf8))

        #expect(config.modelType == "diffusion_gemma")
        #expect(config.canvasLength == 256)
        #expect(config.boiTokenId == 255_999)
        #expect(config.eoiTokenId == 258_882)
        #expect(config.imageTokenId == 258_880)
        #expect(config.textConfig.modelType == "diffusion_gemma_text")
        #expect(config.textConfig.hiddenSize == 2816)
        #expect(config.textConfig.numHiddenLayers == 30)
        #expect(config.textConfig.numExperts == 128)
        #expect(config.textConfig.topKExperts == 8)
        #expect(config.textConfig.moeIntermediateSize == 704)
        #expect(config.textConfig.enableMoeBlock)
        #expect(config.textConfig.layerTypes.filter { $0 == "full_attention" }.count == 5)
    }

    @Test("LLM factory source registers diffusion_gemma without aliasing to Gemma4 AR model")
    func factorySourceRegistersDiffusionGemma() throws {
        let source = try Self.repositoryFile("Libraries/MLXLLM/LLMModelFactory.swift")

        #expect(source.contains(#""diffusion_gemma": create("#))
        #expect(source.contains("DiffusionGemmaConfiguration.self"))
        #expect(source.contains("DiffusionGemmaForBlockDiffusion.init"))
        #expect(!source.contains(#""diffusion_gemma": create(Gemma4TextConfiguration.self"#))
    }

    @Test("DiffusionGemma source fails closed before autoregressive TokenIterator generation")
    func prepareSourceFailsClosed() throws {
        let source = try Self.repositoryFile("Libraries/MLXLLM/Models/DiffusionGemma.swift")

        #expect(source.contains("DiffusionGemma requires block-diffusion denoising generation"))
        #expect(source.contains("throw DiffusionGemmaRuntimeError.blockDiffusionGenerationNotImplemented"))
        #expect(!source.contains("return encoder(input.text"))
    }

    private static func repositoryFile(_ path: String) throws -> String {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: repoRoot.appending(path: path), encoding: .utf8)
    }

    private static let minimalConfig = #"""
    {
      "architectures": ["DiffusionGemmaForBlockDiffusion"],
      "model_type": "diffusion_gemma",
      "boi_token_id": 255999,
      "eoi_token_id": 258882,
      "image_token_id": 258880,
      "canvas_length": 256,
      "tie_word_embeddings": true,
      "text_config": {
        "model_type": "diffusion_gemma_text",
        "hidden_size": 2816,
        "num_hidden_layers": 30,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "intermediate_size": 2112,
        "moe_intermediate_size": 704,
        "num_experts": 128,
        "top_k_experts": 8,
        "vocab_size": 262144,
        "sliding_window": 1024,
        "final_logit_softcapping": 30.0,
        "attention_bias": false,
        "attention_k_eq_v": false,
        "layer_types": [
          "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
        ],
        "rope_parameters": {
          "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional"
          },
          "sliding_attention": {
            "rope_theta": 10000.0,
            "rope_type": "default"
          }
        }
      }
    }
    """#
}
