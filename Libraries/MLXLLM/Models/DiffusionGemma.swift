// Copyright © 2026 Jinho Jang (eric@jangq.ai)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum DiffusionGemmaRuntimeError: Error, LocalizedError, Sendable, Equatable {
    case blockDiffusionGenerationNotImplemented

    public var errorDescription: String? {
        switch self {
        case .blockDiffusionGenerationNotImplemented:
            return "DiffusionGemma requires block-diffusion denoising generation; autoregressive token iteration is not supported for this model."
        }
    }
}

public struct DiffusionGemmaConfiguration: Codable, Sendable {
    public let modelType: String
    public let textConfig: Gemma4TextConfiguration
    public let canvasLength: Int
    public let boiTokenId: Int
    public let eoiTokenId: Int
    public let imageTokenId: Int
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case canvasLength = "canvas_length"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case imageTokenId = "image_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "diffusion_gemma"
        self.canvasLength = try container.decodeIfPresent(Int.self, forKey: .canvasLength) ?? 256
        self.boiTokenId = try container.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255_999
        self.eoiTokenId = try container.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? 258_882
        self.imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258_880
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.textConfig = try Gemma4TextConfiguration(from: decoder)
    }
}

/// DiffusionGemma scaffold.
///
/// The checkpoint is not a normal autoregressive Gemma4 decoder: prompt tokens
/// populate an encoder KV cache, then a fixed-length canvas is denoised with
/// bidirectional decoder attention and self-conditioning. This type registers
/// the real config/model family without allowing the generic TokenIterator path
/// to produce misleading autoregressive output before the denoising loop lands.
public class DiffusionGemmaForBlockDiffusion: Module, LLMModel {
    @ModuleInfo(key: "encoder") public var encoder: Gemma4TextModel

    public let config: DiffusionGemmaConfiguration
    public var vocabularySize: Int { config.textConfig.vocabSize }
    public var canvasLength: Int { config.canvasLength }

    public init(_ config: DiffusionGemmaConfiguration) {
        self.config = config
        self._encoder.wrappedValue = Gemma4TextModel(config.textConfig)
        super.init()
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        throw DiffusionGemmaRuntimeError.blockDiffusionGenerationNotImplemented
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        encoder.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        var encoderWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            let prefixes = [
                "model.encoder.language_model.",
                "encoder.language_model.",
            ]
            if let prefix = prefixes.first(where: { key.hasPrefix($0) }) {
                let suffix = String(key.dropFirst(prefix.count))
                encoderWeights["encoder.model.\(suffix)"] = value
            }
        }
        return encoderWeights
    }

    public var loraLayers: [Module] {
        encoder.loraLayers
    }
}
