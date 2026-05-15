// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CoreGraphics
import Foundation
@testable import MLXVLM
import Testing

@Suite("VL image extent and shape guard contracts")
struct VLShapeGuardFocusedTests {
    @Test("QwenVL.intExtent accepts finite positive dimensions")
    func qwenVLExtentAcceptsFinitePositiveDimensions() throws {
        let (height, width) = try QwenVL.intExtent(CGSize(width: 1024.7, height: 768.3))
        #expect(height == 768)
        #expect(width == 1025)
    }

    @Test("QwenVL.intExtent rejects non-finite and non-positive dimensions")
    func qwenVLExtentRejectsInvalidDimensions() throws {
        let invalidSizes = [
            CGSize(width: CGFloat.infinity, height: 768),
            CGSize(width: 1024, height: CGFloat.infinity),
            CGSize(width: 1024, height: CGFloat.nan),
            CGSize(width: 0, height: 768),
            CGSize(width: 1024, height: 0),
            CGSize(width: -1024, height: 768),
        ]
        for size in invalidSizes {
            #expect(throws: VLMError.self) {
                _ = try QwenVL.intExtent(size)
            }
        }
    }

    @Test("Qwen, ZAYA, GLM, Gemma, LFM, and Smol VL processors use extent guard")
    func vlProcessorsUseFiniteExtentGuard() throws {
        let guardedFiles = [
            "Libraries/MLXVLM/Models/Qwen2VL.swift",
            "Libraries/MLXVLM/Models/Qwen25VL.swift",
            "Libraries/MLXVLM/Models/Qwen3VL.swift",
            "Libraries/MLXVLM/Models/Zaya1VL.swift",
            "Libraries/MLXVLM/Models/GlmOcr.swift",
            "Libraries/MLXVLM/Models/Gemma4.swift",
            "Libraries/MLXVLM/Models/LFM2VL.swift",
            "Libraries/MLXVLM/Models/SmolVLM2.swift",
        ]

        for file in guardedFiles {
            let source = try Self.source(file)
            #expect(source.contains("QwenVL.intExtent("), "\(file) must route image extents through QwenVL.intExtent")
            #expect(!source.contains("height: Int(size.height), width: Int(size.width)"))
            #expect(!source.contains("height: Int(extent.height)"))
            #expect(!source.contains("let width = Int(image.extent.width)"))
            #expect(!source.contains("let height = Int(image.extent.height)"))
            #expect(!source.contains("(Int(ci.extent.width), Int(ci.extent.height))"))
        }
    }

    @Test("SmolVLM2 tile path stays throwing so invalid extents surface as errors")
    func smolVLM2TilePathStaysThrowing() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/SmolVLM2.swift")
        #expect(source.contains(
            "func tiles(from originalImage: CIImage) throws -> (tiles: [CIImage], rows: Int, cols: Int)"))
        #expect(source.contains("try QwenVL.intExtent(originalImage.extent.size)"))
        #expect(source.contains("try tiles(from: image)"))
    }

    @Test("disk restore keeps rank guard for 2D arrays before sequence-dim access")
    func diskRestoreKeepsRankGuardBeforeDimAccess() throws {
        let source = try Self.source("Libraries/MLXLMCommon/Cache/CacheHelpers.swift")
        #expect(source.contains("keys.shape.count >= 3"))
        #expect(source.contains("values.shape.count >= 3"))
        #expect(source.contains("Need >= 3D. Falling back to fresh prefill."))
    }

    @Test("ZAYA1-VL JANGTQ-K nested routed bits preserve gate-up and down widths")
    func zaya1VLNestedRoutedBitsDecode() throws {
        let config = try JSONDecoder.json5().decode(
            Zaya1VLConfiguration.self,
            from: Data(Self.minimalZaya1VLJANGTQKConfig.utf8))
        let text = config.makeZayaTextConfiguration()
        #expect(text.mxtqBits == 2)
        #expect(text.mxtqGateUpBits == 2)
        #expect(text.mxtqDownBits == 4)
        #expect(config.mxtqSeed == 123)
        #expect(text.mxtqSeed == 123)
    }

    private static func source(_ relativePath: String) throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: repo.appendingPathComponent(relativePath), encoding: .utf8)
    }

    private static let minimalZaya1VLJANGTQKConfig = #"""
    {
      "model_type": "zaya1_vl",
      "architectures": ["Zaya1VLForConditionalGeneration"],
      "hidden_size": 2048,
      "num_hidden_layers": 2,
      "num_attention_heads": 16,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "num_query_groups": 2,
      "max_position_embeddings": 32768,
      "rotary_base": 1000000,
      "rope_pct": 0.5,
      "ffn_hidden_size": 4096,
      "zaya_mlp_expansion": 256,
      "zaya_expert_layout": "split_switch_mlp",
      "norm_epsilon": 1e-6,
      "clamp_temp": false,
      "projector_hidden_act": "gelu",
      "num_experts": 16,
      "moe_router_topk": 1,
      "cca": true,
      "zaya_use_eda": true,
      "zaya_use_mod": true,
      "scale_residual_merge": true,
      "residual_in_fp32": false,
      "tie_word_embeddings": true,
      "vision_lora": false,
      "vocab_size": 262272,
      "image_token_id": 151655,
      "vision_start_token_id": 151652,
      "vision_end_token_id": 151653,
      "weight_format": "mxtq",
      "mxtq_seed": 123,
      "mxtq_bits": {
        "routed_expert": {
          "gate_proj": 2,
          "up_proj": 2,
          "down_proj": 4
        },
        "attention": 8,
        "router": 16,
        "embed_tokens": 8,
        "lm_head": 8,
        "cca_conv": 16,
        "norms_residual": 16
      },
      "vision_config": {
        "model_type": "qwen2_5_vl",
        "depth": 2,
        "hidden_size": 128,
        "intermediate_size": 256,
        "out_hidden_size": 2048,
        "num_heads": 4,
        "patch_size": 14,
        "spatial_patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "window_size": 112,
        "fullatt_block_indexes": [1],
        "tokens_per_second": 4,
        "in_chans": 3,
        "layer_norm_eps": 1e-6,
        "skip_vision": false,
        "hidden_act": "silu"
      }
    }
    """#
}
