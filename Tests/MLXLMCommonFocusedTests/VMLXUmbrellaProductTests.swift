// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing
import VMLX

@Suite("VMLX umbrella product")
struct VMLXUmbrellaProductTests {
    @Test("umbrella re-exports Osaurus runtime modules")
    func reexportsRuntimeModules() {
        let _: MLXArray.Type = MLXArray.self
        let _: ModelContext.Type = ModelContext.self
        let _: UserInput.Type = UserInput.self
        let _: GenerateParameters.Type = GenerateParameters.self
        let _: LLMRegistry.Type = LLMRegistry.self
        let _: MediaProcessing.Type = MediaProcessing.self
        let _: AutoTokenizer.Type = AutoTokenizer.self
        let _: Template.Type = Template.self
        let _: VMLXServerRuntimeSettings.Type = VMLXServerRuntimeSettings.self
        let _: GenerationConfigFile.Type = GenerationConfigFile.self
        let _: JangConfig.Type = JangConfig.self
        let _: JangCapabilities.Type = JangCapabilities.self
        let _: ParserResolution.Type = ParserResolution.self
        let _: ToolCallFormat.Type = ToolCallFormat.self
        let _: ReasoningParser.Type = ReasoningParser.self
        let _: MTPBundleStatus.Type = MTPBundleStatus.self
        let _: MTPBundleStatusSnapshot.Type = MTPBundleStatusSnapshot.self
        let _: NativeMTPTuning.Type = NativeMTPTuning.self
        let _: NativeMTPTuningSnapshot.Type = NativeMTPTuningSnapshot.self
    }

    @Test("umbrella exposes MTP tuning snapshot JSON for Osaurus")
    func exposesMTPTuningSnapshotJSON() throws {
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 333,
            mode: .preservedEnabled,
            tensorSamples: ["model.layers.64.mtp_fc.weight"],
            visionTensorSamples: ["vision_tower.blocks.0.attn.qkv.weight"],
            configEvidence: ["tuning_file=vmlx_mtp_tuning.json"],
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 2,
                verifierMode: "chunk_commit",
                validated: true,
                outputEquivalent: true,
                blocked: false,
                cacheMode: "paged+ssm",
                artifact: "docs/internal/release-gates/qwen36_mxfp4/result.json",
                baselineTokensPerSecond: 24.65,
                bestTokensPerSecond: 45.71,
                speedupVsBaseline: 1.85))

        let snapshot = status.snapshot
        #expect(snapshot.canAutoLaunch)
        #expect(snapshot.hasUsableNativeMTPTuning)
        #expect(!snapshot.requiresNativeMTPTuningBeforeAutoLaunch)
        #expect(snapshot.tuning?.usableBestDepth == 2)

        let data = try JSONEncoder().encode(snapshot)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["can_auto_launch"] as? Bool == true)
        #expect(object["has_usable_native_mtp_tuning"] as? Bool == true)
        #expect(object["requires_native_mtp_tuning_before_auto_launch"] as? Bool == false)
        #expect(object["bundle_has_vision"] as? Bool == true)

        let tuning = try #require(object["tuning"] as? [String: Any])
        #expect(tuning["file"] as? String == "vmlx_mtp_tuning.json")
        #expect(tuning["best_depth"] as? Int == 2)
        #expect(tuning["usable_best_depth"] as? Int == 2)
        #expect(tuning["verifier_mode"] as? String == "chunk_commit")
    }
}
