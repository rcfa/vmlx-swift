// Copyright © 2024 Apple Inc.

import Foundation

/// Tristate capability flag for host/UI status surfaces.
///
/// `unknown` is intentional: a parser or model-type heuristic is not the same
/// thing as a production claim that a model is trained for a modality.
public enum ModelRuntimeCapabilitySupport: String, Codable, Sendable, Equatable {
    case supported
    case unsupported
    case unknown

    static func from(_ value: Bool?) -> Self? {
        guard let value else { return nil }
        return value ? .supported : .unsupported
    }
}

/// Codable capability snapshot for Osaurus-style model status endpoints.
///
/// This is a no-load data surface. It combines the resolved model configuration,
/// JANG capability stamps, bundle `generation_config.json` defaults, and native
/// MTP status into a single JSON shape without changing decode behavior.
public struct ModelRuntimeCapabilitySnapshot: Codable, Sendable, Equatable {
    public let modelName: String
    public let modelType: String?
    public let family: String?
    public let modality: String?
    public let cacheType: String?

    public let supportsText: ModelRuntimeCapabilitySupport
    public let supportsVision: ModelRuntimeCapabilitySupport
    public let supportsVideo: ModelRuntimeCapabilitySupport
    public let supportsAudio: ModelRuntimeCapabilitySupport
    public let supportsTools: ModelRuntimeCapabilitySupport
    public let supportsReasoning: ModelRuntimeCapabilitySupport
    public let supportsNativeMTP: ModelRuntimeCapabilitySupport

    public let thinkInTemplate: Bool?
    public let toolParser: String?
    public let reasoningParser: String?
    public let generationDefaults: GenerationConfigFile?
    public let nativeMTP: MTPBundleStatusSnapshot?

    public init(
        configuration: ModelConfiguration,
        capabilities: JangCapabilities? = nil,
        modelType: String? = nil
    ) {
        self.modelName = configuration.name
        self.modelType = modelType
        self.family = capabilities?.family
        self.modality = capabilities?.modality
        self.cacheType = capabilities?.cacheType
        self.supportsText = Self.textSupport(capabilities)
        self.supportsVision = Self.visionSupport(
            capabilities: capabilities,
            mtpStatus: configuration.mtpStatus)
        self.supportsVideo = Self.videoSupport(capabilities)
        self.supportsAudio = Self.audioSupport(capabilities)
        self.supportsTools = Self.toolSupport(
            capabilities: capabilities,
            toolCallFormat: configuration.toolCallFormat)
        self.supportsReasoning = Self.reasoningSupport(
            capabilities: capabilities,
            reasoningParserName: configuration.reasoningParserName)
        self.supportsNativeMTP =
            configuration.mtpStatus?.canAutoLaunchMTP == true ? .supported : .unsupported
        self.thinkInTemplate = capabilities?.thinkInTemplate
        self.toolParser = capabilities?.toolParser ?? configuration.toolCallFormat?.rawValue
        self.reasoningParser = capabilities?.reasoningParser ?? configuration.reasoningParserName
        self.generationDefaults = configuration.generationDefaults
        self.nativeMTP = configuration.mtpStatus?.snapshot
    }

    public init(
        resolvedConfiguration: ResolvedModelConfiguration,
        capabilities: JangCapabilities? = nil,
        modelType: String? = nil
    ) {
        self.modelName = resolvedConfiguration.name
        self.modelType = modelType
        self.family = capabilities?.family
        self.modality = capabilities?.modality
        self.cacheType = capabilities?.cacheType
        self.supportsText = Self.textSupport(capabilities)
        self.supportsVision = Self.visionSupport(
            capabilities: capabilities,
            mtpStatus: resolvedConfiguration.mtpStatus)
        self.supportsVideo = Self.videoSupport(capabilities)
        self.supportsAudio = Self.audioSupport(capabilities)
        self.supportsTools = Self.toolSupport(
            capabilities: capabilities,
            toolCallFormat: resolvedConfiguration.toolCallFormat)
        self.supportsReasoning = Self.reasoningSupport(
            capabilities: capabilities,
            reasoningParserName: resolvedConfiguration.reasoningParserName)
        self.supportsNativeMTP =
            resolvedConfiguration.mtpStatus?.canAutoLaunchMTP == true ? .supported : .unsupported
        self.thinkInTemplate = capabilities?.thinkInTemplate
        self.toolParser = capabilities?.toolParser ?? resolvedConfiguration.toolCallFormat?.rawValue
        self.reasoningParser = capabilities?.reasoningParser
            ?? resolvedConfiguration.reasoningParserName
        self.generationDefaults = resolvedConfiguration.generationDefaults
        self.nativeMTP = resolvedConfiguration.mtpStatus?.snapshot
    }

    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case modelType = "model_type"
        case family
        case modality
        case cacheType = "cache_type"
        case supportsText = "supports_text"
        case supportsVision = "supports_vision"
        case supportsVideo = "supports_video"
        case supportsAudio = "supports_audio"
        case supportsTools = "supports_tools"
        case supportsReasoning = "supports_reasoning"
        case supportsNativeMTP = "supports_native_mtp"
        case thinkInTemplate = "think_in_template"
        case toolParser = "tool_parser"
        case reasoningParser = "reasoning_parser"
        case generationDefaults = "generation_defaults"
        case nativeMTP = "native_mtp"
    }
}

private extension ModelRuntimeCapabilitySnapshot {
    static func textSupport(_ capabilities: JangCapabilities?) -> ModelRuntimeCapabilitySupport {
        ModelRuntimeCapabilitySupport.from(capabilities?.supportsText) ?? .supported
    }

    static func visionSupport(
        capabilities: JangCapabilities?,
        mtpStatus: MTPBundleStatus?
    ) -> ModelRuntimeCapabilitySupport {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsVision) {
            return explicit
        }
        if mtpStatus?.bundleHasVision == true {
            return .supported
        }
        return supportFromModality(
            capabilities?.modality,
            supportedTokens: ["vision", "image", "images", "vl", "vlm", "multimodal", "omni"],
            unsupportedTokens: ["text"])
    }

    static func videoSupport(_ capabilities: JangCapabilities?) -> ModelRuntimeCapabilitySupport {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsVideo) {
            return explicit
        }
        return supportFromModality(
            capabilities?.modality,
            supportedTokens: ["video", "videos", "omni"],
            unsupportedTokens: ["text", "audio"])
    }

    static func audioSupport(_ capabilities: JangCapabilities?) -> ModelRuntimeCapabilitySupport {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsAudio) {
            return explicit
        }
        return supportFromModality(
            capabilities?.modality,
            supportedTokens: ["audio", "speech", "voice", "omni"],
            unsupportedTokens: ["text", "vision", "image", "images", "vl", "vlm", "video"])
    }

    static func toolSupport(
        capabilities: JangCapabilities?,
        toolCallFormat: ToolCallFormat?
    ) -> ModelRuntimeCapabilitySupport {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsTools) {
            return explicit
        }
        if capabilities?.toolParser != nil || toolCallFormat != nil {
            return .unknown
        }
        return .unknown
    }

    static func reasoningSupport(
        capabilities: JangCapabilities?,
        reasoningParserName: String?
    ) -> ModelRuntimeCapabilitySupport {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsThinking) {
            return explicit
        }
        let stamp = capabilities?.reasoningParser ?? reasoningParserName
        guard let stamp else { return .unknown }
        return ReasoningParser.fromCapabilityName(stamp) == nil ? .unsupported : .supported
    }

    static func supportFromModality(
        _ modality: String?,
        supportedTokens: Set<String>,
        unsupportedTokens: Set<String>
    ) -> ModelRuntimeCapabilitySupport {
        let tokens = modalityTokens(modality)
        if !tokens.isDisjoint(with: supportedTokens) {
            return .supported
        }
        if !tokens.isDisjoint(with: unsupportedTokens) {
            return .unsupported
        }
        return .unknown
    }

    static func modalityTokens(_ modality: String?) -> Set<String> {
        guard let modality else { return [] }
        return Set(
            modality.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init))
    }
}
