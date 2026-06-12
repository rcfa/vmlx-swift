// Copyright © 2026 Osaurus

import Foundation
import MLX
import MLXLMCommon

public enum Model2VecStaticEmbeddingError: LocalizedError {
    case missingWeights(URL)
    case missingEmbeddingTensor(URL)
    case invalidEmbeddingRank([Int])
    case emptyBatch

    public var errorDescription: String? {
        switch self {
        case .missingWeights(let url):
            return "Missing Model2Vec weights at \(url.path)"
        case .missingEmbeddingTensor(let url):
            return "Missing Model2Vec 'embeddings' tensor in \(url.path)"
        case .invalidEmbeddingRank(let shape):
            return "Model2Vec embeddings tensor must be rank 2, got shape \(shape)"
        case .emptyBatch:
            return "Model2Vec embedding batch must contain at least one text"
        }
    }
}

public struct Model2VecStaticEmbeddingConfiguration: Decodable, Sendable {
    public let normalize: Bool?
    public let hiddenDim: Int?

    enum CodingKeys: String, CodingKey {
        case normalize
        case hiddenDim = "hidden_dim"
    }
}

/// Lightweight Model2Vec/static-embedding pipeline for bundles such as
/// `minishlab/potion-base-4M`.
///
/// This keeps static embeddings inside vmlx-swift instead of pulling the
/// separate `swift-embeddings` package and its external transformer graph into
/// Osaurus.
public actor Model2VecStaticEmbeddingPipeline {
    public let dimension: Int

    private let embeddings: MLXArray
    private let tokenizer: any Tokenizer
    private let unknownTokenId: Int?
    private let normalize: Bool

    public init(
        embeddings: MLXArray,
        tokenizer: any Tokenizer,
        normalize: Bool
    ) throws {
        guard embeddings.shape.count == 2 else {
            throw Model2VecStaticEmbeddingError.invalidEmbeddingRank(embeddings.shape)
        }
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.unknownTokenId = tokenizer.unknownTokenId
        self.normalize = normalize
        self.dimension = embeddings.shape[1]
        eval(embeddings)
    }

    public static func load(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader
    ) async throws -> Model2VecStaticEmbeddingPipeline {
        let weightsURL = directory.appending(component: "model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw Model2VecStaticEmbeddingError.missingWeights(weightsURL)
        }

        let configURL = directory.appending(component: "config.json")
        let config = try? JSONDecoder.json5().decode(
            Model2VecStaticEmbeddingConfiguration.self,
            from: Data(contentsOf: configURL)
        )
        let weights = try loadArrays(url: weightsURL)
        guard let embeddings = weights["embeddings"] else {
            throw Model2VecStaticEmbeddingError.missingEmbeddingTensor(weightsURL)
        }
        let tokenizer = try await tokenizerLoader.load(from: directory)
        return try Model2VecStaticEmbeddingPipeline(
            embeddings: embeddings,
            tokenizer: tokenizer,
            normalize: config?.normalize ?? false
        )
    }

    public func embed(text: String) throws -> [Float] {
        try embed(texts: [text])[0]
    }

    public func embed(texts: [String]) throws -> [[Float]] {
        guard !texts.isEmpty else {
            throw Model2VecStaticEmbeddingError.emptyBatch
        }
        return try texts.map(embedOne(_:))
    }

    private func embedOne(_ text: String) throws -> [Float] {
        let tokens = tokenizer.encode(text: text, addSpecialTokens: false).filter { token in
            if let unknownTokenId {
                return token != unknownTokenId
            }
            return true
        }

        guard !tokens.isEmpty else {
            return Array(repeating: 0, count: dimension)
        }

        let ids = MLXArray(tokens.map(Int32.init))
        var vector = embeddings.take(ids, axis: 0).mean(axis: 0)
        if normalize {
            vector = vector.l2Normalized()
        }
        eval(vector)
        return vector.asArray(Float.self)
    }
}
