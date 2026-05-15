// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
@testable import MLXLLM
import Testing

@Suite("Zaya config focused decode")
struct ZayaConfigDecodeFocusedTests {
    @Test("flat and per-role mxtq_bits decode to routed expert bits")
    func flatAndPerRoleBitsDecode() throws {
        let flat = try JSONDecoder().decode(
            ZayaConfiguration.self,
            from: Data(#"{"model_type":"zaya","mxtq_bits":2}"#.utf8))
        #expect(flat.textConfig.mxtqBits == 2)
        #expect(flat.textConfig.mxtqGateUpBits == 2)
        #expect(flat.textConfig.mxtqDownBits == 2)

        let perRole = try JSONDecoder().decode(
            ZayaConfiguration.self,
            from: Data(#"{"model_type":"zaya","mxtq_bits":{"routed_expert":4,"attention":8}}"#.utf8))
        #expect(perRole.textConfig.mxtqBits == 4)
        #expect(perRole.textConfig.mxtqGateUpBits == 4)
        #expect(perRole.textConfig.mxtqDownBits == 4)
    }

    @Test("nested JANGTQ_K projection bits preserve gate-up and down widths")
    func nestedProjectionBitsDecode() throws {
        let json = """
        {
          "model_type": "zaya",
          "weight_format": "mxtq",
          "mxtq_bits": {
            "routed_expert": {
              "gate_proj": 2,
              "up_proj": 2,
              "down_proj": 4
            },
            "attention": 8,
            "router": 16
          }
        }
        """
        let cfg = try JSONDecoder().decode(ZayaConfiguration.self, from: Data(json.utf8))
        #expect(cfg.textConfig.mxtqBits == 2)
        #expect(cfg.textConfig.mxtqGateUpBits == 2)
        #expect(cfg.textConfig.mxtqDownBits == 4)
    }

    @Test("nested JANGTQ_K projection bits reject mismatched gate and up")
    func nestedProjectionBitsRejectMismatchedGateUp() throws {
        let json = """
        {
          "model_type": "zaya",
          "weight_format": "mxtq",
          "mxtq_bits": {
            "routed_expert": {
              "gate_proj": 2,
              "up_proj": 4,
              "down_proj": 4
            }
          }
        }
        """
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(ZayaConfiguration.self, from: Data(json.utf8))
        }
    }

    @Test("non-integer mxtq_bits dictionaries without routed_expert are rejected")
    func invalidBitsDictionaryRejectsMissingRoutedExpert() throws {
        let json = """
        {
          "model_type": "zaya",
          "weight_format": "mxtq",
          "mxtq_bits": {
            "attention": "eight"
          }
        }
        """
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(ZayaConfiguration.self, from: Data(json.utf8))
        }
    }
}
