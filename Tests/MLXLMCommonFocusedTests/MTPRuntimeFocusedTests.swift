// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing

@Suite("MTP runtime metadata")
struct MTPRuntimeFocusedTests {
    @Test("Qwen-style preserved MTP bundle is detected but not auto-enabled")
    func qwenPreservedMTPBundleIsDetectedButNotAutoEnabled() throws {
        let root = try makeTemporaryBundle(name: "qwen-mtp-detected")
        defer { try? FileManager.default.removeItem(at: root) }

        try writeJSON([
            "model_type": "qwen3_vl",
            "text_config": [
                "model_type": "qwen3",
                "num_hidden_layers": 48,
                "mtp_num_hidden_layers": 1,
            ] as [String: Any],
        ], to: root.appendingPathComponent("config.json"))
        try writeJSON([
            "runtime": [
                "total_weight_bytes": 17_820_460_160,
                "total_weight_gb": 16.6,
                "bundle_has_mtp": true,
                "mtp_layers": 1,
                "mtp_mode": "preserved_enabled",
            ] as [String: Any],
        ], to: root.appendingPathComponent("jang_config.json"))
        try writeJSON([
            "weight_map": [
                "mtp.fc.weight": "model-00029-of-00029.safetensors",
                "mtp.layers.0.self_attn.q_proj.weight": "model-00029-of-00029.safetensors",
                "mtp.layers.0.mlp.down_proj.weight": "model-00029-of-00029.safetensors",
                "vision_tower.blocks.0.attn.qkv.weight": "model-00001-of-00029.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00029.safetensors",
            ] as [String: Any],
        ], to: root.appendingPathComponent("model.safetensors.index.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)

        #expect(status.bundleHasMTP)
        #expect(status.configuredLayers == 1)
        #expect(status.tensorCount == 3)
        #expect(status.visionTensorCount == 1)
        #expect(status.mode == .preservedEnabled)
        #expect(status.hasCompleteMTPArtifact)
        #expect(status.requiresAcceptRejectBeforeEnable)
        #expect(!status.speculativeDecodeEnabled)
        #expect(!status.canAutoLaunchMTP)
        #expect(status.configEvidence.contains("text_config.mtp_num_hidden_layers=1"))
        #expect(status.statusLine.contains("accept/reject required"))
    }

    @Test("MTP config without tensors is reported as metadata-only")
    func configOnlyMTPIsMetadataOnlyMissingWeights() throws {
        let root = try makeTemporaryBundle(name: "qwen-mtp-missing-weights")
        defer { try? FileManager.default.removeItem(at: root) }

        try writeJSON([
            "text_config": [
                "mtp_num_hidden_layers": 1,
            ] as [String: Any],
        ], to: root.appendingPathComponent("config.json"))
        try writeJSON([
            "weight_map": [
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
            ] as [String: Any],
        ], to: root.appendingPathComponent("model.safetensors.index.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)

        #expect(!status.bundleHasMTP)
        #expect(status.configuredLayers == 1)
        #expect(status.tensorCount == 0)
        #expect(status.mode == .metadataOnlyMissingWeights)
        #expect(!status.hasCompleteMTPArtifact)
        #expect(!status.speculativeDecodeEnabled)
        #expect(!status.canAutoLaunchMTP)
    }

    @Test("JANG runtime parses MTP activation metadata")
    func jangRuntimeParsesMTPActivationMetadata() throws {
        let config = try JangLoader.parseConfig(from: [
            "runtime": [
                "total_weight_bytes": 17_820_460_160,
                "total_weight_gb": 16.6,
                "bundle_has_mtp": true,
                "mtp_layers": 1,
                "mtp_mode": "preserved_enabled",
            ] as [String: Any],
        ])

        #expect(config.runtime.totalWeightBytes == 17_820_460_160)
        #expect(config.runtime.bundleHasMTP)
        #expect(config.runtime.mtpLayers == 1)
        #expect(config.runtime.mtpMode == .preservedEnabled)
    }

    @Test("ModelConfiguration carries MTP status into resolved configuration")
    func modelConfigurationCarriesMTPStatusIntoResolvedConfiguration() {
        let root = URL(fileURLWithPath: "/tmp/qwen-mtp")
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 333,
            mode: .preservedEnabled,
            tensorSamples: ["mtp.fc.weight"],
            visionTensorSamples: ["vision_tower.blocks.0.attn.qkv.weight"],
            configEvidence: ["text_config.mtp_num_hidden_layers=1"])
        let configuration = ModelConfiguration(
            directory: root,
            mtpStatus: status)

        let resolved = configuration.resolved(modelDirectory: root, tokenizerDirectory: root)

        #expect(configuration.mtpStatus == status)
        #expect(resolved.mtpStatus == status)
        #expect(resolved.mtpStatus?.requiresAcceptRejectBeforeEnable == true)
    }

    @Test("optional real local MTP bundle inspection")
    func optionalRealLocalMTPBundleInspection() throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_MTP_REAL_BUNDLE"],
            !path.isEmpty
        else {
            return
        }

        let status = try MTPBundleInspector.inspect(
            modelDirectory: URL(fileURLWithPath: path))

        #expect(status.bundleHasMTP)
        #expect(status.configuredLayers > 0)
        #expect(status.tensorCount > 0)
        #expect(status.hasCompleteMTPArtifact)
        #expect(!status.canAutoLaunchMTP)
        if ProcessInfo.processInfo.environment["VMLX_MTP_REAL_BUNDLE_EXPECTS_VL"] == "1" {
            #expect(status.visionTensorCount > 0)
            #expect(status.bundleHasVision)
        }
    }

    private func makeTemporaryBundle(name: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func writeJSON(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }
}
