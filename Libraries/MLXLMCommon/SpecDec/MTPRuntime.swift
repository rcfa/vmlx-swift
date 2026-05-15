// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Runtime mode stamped by a model bundle or inferred from its files.
///
/// `preserved_enabled` means the converted artifact carries MTP metadata and
/// tensors, not that speculative MTP decode is live. The engine may only launch
/// MTP automatically when the mode proves a verified accept/reject runtime.
public enum MTPRuntimeMode: String, Codable, Sendable, Equatable {
    case none
    case preservedDisabled = "preserved_disabled"
    case preservedEnabled = "preserved_enabled"
    case metadataOnlyMissingWeights = "metadata_only_missing_weights"
    case enabled
    case speculativeVerified = "speculative_verified"
    case unknown

    public init(rawMode: String?) {
        let normalized =
            (rawMode ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: " ", with: "_")

        switch normalized {
        case "", "none", "off", "false":
            self = .none
        case "disabled", "preserved_disabled", "preserved_off":
            self = .preservedDisabled
        case "preserved", "preserved_enabled", "runtime_unwired", "available":
            self = .preservedEnabled
        case "metadata_only", "metadata_only_missing_weights", "config_only":
            self = .metadataOnlyMissingWeights
        case "enabled", "speculative_enabled", "accept_reject", "accept_reject_enabled":
            self = .enabled
        case "speculative_verified", "verified", "verified_accept_reject":
            self = .speculativeVerified
        default:
            self = .unknown
        }
    }

    public var hasSpeculativeAcceptReject: Bool {
        switch self {
        case .enabled, .speculativeVerified:
            true
        case .none, .preservedDisabled, .preservedEnabled, .metadataOnlyMissingWeights,
            .unknown:
            false
        }
    }
}

/// No-load MTP status derived from `config.json`, `jang_config.json`, and tensor
/// names. This type is safe to expose through Osaurus capability/status APIs.
public struct MTPBundleStatus: Codable, Sendable, Equatable {
    public let bundleHasMTP: Bool
    public let configuredLayers: Int
    public let tensorCount: Int
    public let visionTensorCount: Int
    public let mode: MTPRuntimeMode
    public let tensorSamples: [String]
    public let visionTensorSamples: [String]
    public let configEvidence: [String]

    public init(
        bundleHasMTP: Bool = false,
        configuredLayers: Int = 0,
        tensorCount: Int = 0,
        visionTensorCount: Int = 0,
        mode: MTPRuntimeMode = .none,
        tensorSamples: [String] = [],
        visionTensorSamples: [String] = [],
        configEvidence: [String] = []
    ) {
        self.bundleHasMTP = bundleHasMTP
        self.configuredLayers = configuredLayers
        self.tensorCount = tensorCount
        self.visionTensorCount = visionTensorCount
        self.mode = mode
        self.tensorSamples = tensorSamples
        self.visionTensorSamples = visionTensorSamples
        self.configEvidence = configEvidence
    }

    public var hasCompleteMTPArtifact: Bool {
        bundleHasMTP && configuredLayers > 0 && tensorCount > 0
    }

    public var speculativeDecodeEnabled: Bool {
        hasCompleteMTPArtifact && mode.hasSpeculativeAcceptReject
    }

    public var canAutoLaunchMTP: Bool {
        speculativeDecodeEnabled
    }

    public var requiresAcceptRejectBeforeEnable: Bool {
        hasCompleteMTPArtifact && !mode.hasSpeculativeAcceptReject
    }

    public var bundleHasVision: Bool {
        visionTensorCount > 0
    }

    public var statusLine: String {
        let base = "mtp: \(mode.rawValue), layers=\(configuredLayers), tensors=\(tensorCount)"
        if speculativeDecodeEnabled {
            return "\(base), speculative=on"
        }
        if requiresAcceptRejectBeforeEnable {
            return "\(base), speculative=off (accept/reject required)"
        }
        if configuredLayers > 0 && tensorCount == 0 {
            return "\(base), speculative=off (metadata only; MTP weights missing)"
        }
        return "\(base), speculative=off"
    }
}

/// Contract object for future MTP decode wiring. It intentionally contains no
/// MLX arrays because draft cache/state must stay separate from accepted base KV.
public struct MTPDraftStateContract: Codable, Sendable, Equatable {
    public let mode: MTPRuntimeMode
    public let draftTokenLimit: Int
    public let cacheIsSeparateFromBase: Bool
    public let acceptedTokensOnlyEnterBaseCache: Bool

    public init(
        mode: MTPRuntimeMode,
        draftTokenLimit: Int,
        cacheIsSeparateFromBase: Bool = true,
        acceptedTokensOnlyEnterBaseCache: Bool = true
    ) {
        self.mode = mode
        self.draftTokenLimit = draftTokenLimit
        self.cacheIsSeparateFromBase = cacheIsSeparateFromBase
        self.acceptedTokensOnlyEnterBaseCache = acceptedTokensOnlyEnterBaseCache
    }

    public static func inactive(mode: MTPRuntimeMode = .none) -> MTPDraftStateContract {
        MTPDraftStateContract(mode: mode, draftTokenLimit: 0)
    }
}

/// No-load MTP/VL inspector. It reads JSON metadata and safetensors headers only;
/// it never materializes tensors or changes the active generation path.
public enum MTPBundleInspector {
    public static func inspect(
        modelDirectory: URL,
        jangConfig suppliedJangConfig: JangConfig? = nil
    ) throws -> MTPBundleStatus {
        let config = try loadJSONObjectIfExists(modelDirectory.appendingPathComponent("config.json"))
        let jangConfig = suppliedJangConfig ?? (try? JangLoader.loadConfig(at: modelDirectory))
        let (configuredLayers, evidence, mtpLayerPrefixes) = configuredMTPLayers(
            config: config,
            jangConfig: jangConfig)
        let tensorNames = try loadTensorNames(from: modelDirectory)
        let mtpNames = tensorNames.filter { isMTPName($0, mtpLayerPrefixes: mtpLayerPrefixes) }
        let visionNames = tensorNames.filter(isVisionName)

        let runtimeMode = jangConfig?.runtime.mtpMode ?? .none
        let runtimeBundleHasMTP = jangConfig?.runtime.bundleHasMTP ?? false
        let bundleHasMTP = runtimeBundleHasMTP || !mtpNames.isEmpty

        let mode: MTPRuntimeMode
        if runtimeMode != .none {
            mode = runtimeMode
        } else if !bundleHasMTP && configuredLayers > 0 {
            mode = .metadataOnlyMissingWeights
        } else if bundleHasMTP && configuredLayers > 0 {
            mode = .preservedEnabled
        } else {
            mode = .none
        }

        return MTPBundleStatus(
            bundleHasMTP: bundleHasMTP,
            configuredLayers: configuredLayers,
            tensorCount: mtpNames.count,
            visionTensorCount: visionNames.count,
            mode: mode,
            tensorSamples: Array(mtpNames.sorted().prefix(8)),
            visionTensorSamples: Array(visionNames.sorted().prefix(8)),
            configEvidence: evidence)
    }

    private static func configuredMTPLayers(
        config: [String: Any]?,
        jangConfig: JangConfig?
    ) -> (Int, [String], [String]) {
        var layers = 0
        var evidence: [String] = []
        var layerPrefixes: [String] = []

        func note(_ value: Int?, _ key: String) {
            guard let value, value > 0 else { return }
            layers = max(layers, value)
            evidence.append("\(key)=\(value)")
        }

        if let config {
            note(intValue(config["num_nextn_predict_layers"]), "num_nextn_predict_layers")
            note(intValue(config["mtp_num_hidden_layers"]), "mtp_num_hidden_layers")

            if let textConfig = config["text_config"] as? [String: Any] {
                note(
                    intValue(textConfig["num_nextn_predict_layers"]),
                    "text_config.num_nextn_predict_layers")
                note(
                    intValue(textConfig["mtp_num_hidden_layers"]),
                    "text_config.mtp_num_hidden_layers")
            }

            let topBaseLayers = intValue(config["num_hidden_layers"])
            let textConfig = config["text_config"] as? [String: Any]
            let textBaseLayers = intValue(textConfig?["num_hidden_layers"])
            for baseLayer in [topBaseLayers, textBaseLayers].compactMap({ $0 }) where baseLayer > 0 {
                layerPrefixes.append("model.layers.\(baseLayer).")
                layerPrefixes.append("language_model.model.layers.\(baseLayer).")
            }
        }

        if let runtimeLayers = jangConfig?.runtime.mtpLayers, runtimeLayers > 0 {
            layers = max(layers, runtimeLayers)
            evidence.append("jang_config.runtime.mtp_layers=\(runtimeLayers)")
        }

        return (layers, Array(Set(evidence)).sorted(), Array(Set(layerPrefixes)).sorted())
    }

    private static func loadTensorNames(from directory: URL) throws -> [String] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if let index = try loadJSONObjectIfExists(indexURL),
            let weightMap = index["weight_map"] as? [String: Any]
        {
            return Array(weightMap.keys)
        }

        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])) ?? []
        let safetensors = contents.filter { $0.pathExtension == "safetensors" }

        var names: [String] = []
        for file in safetensors {
            names.append(contentsOf: try safetensorsHeaderNames(file))
        }
        return names
    }

    private static func safetensorsHeaderNames(_ url: URL) throws -> [String] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            return []
        }
        var headerLength: UInt64 = 0
        for (index, byte) in lengthData.enumerated() {
            headerLength |= UInt64(byte) << UInt64(index * 8)
        }
        guard headerLength > 0, headerLength <= 64 * 1024 * 1024 else {
            return []
        }
        guard let headerData = try handle.read(upToCount: Int(headerLength)),
            headerData.count == Int(headerLength),
            let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            return []
        }
        return header.keys.filter { $0 != "__metadata__" }
    }

    private static func isMTPName(_ name: String, mtpLayerPrefixes: [String]) -> Bool {
        let lower = name.lowercased()
        if lower.hasPrefix("mtp.") || lower.hasPrefix("model.mtp_layers.") {
            return true
        }
        if lower.contains(".mtp.") || lower.contains(".mtp_layers.") {
            return true
        }
        if lower.contains("nextn") || lower.contains("next_n") {
            return true
        }
        return mtpLayerPrefixes.contains { name.hasPrefix($0) }
    }

    private static func isVisionName(_ name: String) -> Bool {
        let lower = name.lowercased()
        return lower.hasPrefix("vision_tower.")
            || lower.contains(".vision_tower.")
            || lower.hasPrefix("visual.")
            || lower.contains(".visual.")
            || lower.hasPrefix("vision_model.")
            || lower.contains(".vision_model.")
            || lower.hasPrefix("multi_modal_projector.")
            || lower.hasPrefix("mm_projector.")
            || lower.hasPrefix("image_newline")
    }

    private static func loadJSONObjectIfExists(_ url: URL) throws -> [String: Any]? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private static func intValue(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        if let value = value as? Double { return Int(value) }
        if let value = value as? Float { return Int(value) }
        if let value = value as? NSNumber { return value.intValue }
        if let value = value as? String { return Int(value) }
        return nil
    }
}
