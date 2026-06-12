// Copyright © 2026 Jinho Jang (eric@jangq.ai)
//
// DiffusionGemma block-diffusion engine tests.
//
// Covers the ordered KV read APIs the diffusion decoder relies on, the
// block-diffusion sampling primitives, the DiffusionGemma model family,
// and the BlockDiffusionTokenIterator generation flow.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// MARK: - Ordered KV read APIs (Task 1)

@Test(.serialized)
func testRotatingKVCacheTemporallyOrderedKV() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let cache = RotatingKVCache(maxSize: 4, keep: 0)

    // 6 single-token updates whose key payload encodes the position.
    for position in 0 ..< 6 {
        let keys = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position)))
        let values = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position) + 100))
        _ = cache.update(keys: keys, values: values)
    }
    #expect(cache.offset == 6)

    let ordered = try #require(cache.temporallyOrderedKV())
    #expect(ordered.keys.dim(2) == 4)
    let keyPositions = ordered.keys[0, 0, 0..., 0].asArray(Float.self)
    #expect(keyPositions == [2, 3, 4, 5])
    let valuePositions = ordered.values[0, 0, 0..., 0].asArray(Float.self)
    #expect(valuePositions == [102, 103, 104, 105])

    // Read must not mutate rotation state.
    #expect(cache.offset == 6)
    let again = try #require(cache.temporallyOrderedKV())
    #expect(again.keys[0, 0, 0..., 0].asArray(Float.self) == [2, 3, 4, 5])
}

@Test(.serialized)
func testRotatingKVCacheTemporallyOrderedKVBeforeWrap() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let cache = RotatingKVCache(maxSize: 8, keep: 0)
    for position in 0 ..< 3 {
        let keys = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position)))
        let values = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position)))
        _ = cache.update(keys: keys, values: values)
    }

    let ordered = try #require(cache.temporallyOrderedKV())
    #expect(ordered.keys.dim(2) == 3)
    #expect(ordered.keys[0, 0, 0..., 0].asArray(Float.self) == [0, 1, 2])
    #expect(cache.offset == 3)
}

@Test(.serialized)
func testKVCacheSimpleReadKV() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let cache = KVCacheSimple()
    for position in 0 ..< 3 {
        let keys = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position)))
        let values = MLXArray.full([1, 1, 1, 2], values: MLXArray(Float(position)))
        _ = cache.update(keys: keys, values: values)
    }

    let read = try #require(cache.readKV())
    #expect(read.keys.dim(2) == 3)
    #expect(read.keys[0, 0, 0..., 0].asArray(Float.self) == [0, 1, 2])
    #expect(cache.offset == 3)

    let empty = KVCacheSimple()
    #expect(empty.readKV() == nil)
    #expect(RotatingKVCache(maxSize: 4).temporallyOrderedKV() == nil)
}

// MARK: - Block-diffusion primitives (Task 2)

@Test
func testBlockDiffusionTemperatureSchedule() {
    // First denoising step (curStep == maxSteps) runs at tMax.
    #expect(
        blockDiffusionTemperature(curStep: 48, maxSteps: 48, tMin: 0.4, tMax: 0.8) == 0.8)
    // Annealing toward tMin as curStep approaches 0.
    let nearEnd = blockDiffusionTemperature(curStep: 1, maxSteps: 48, tMin: 0.4, tMax: 0.8)
    #expect(abs(nearEnd - (0.4 + 0.4 / 48)) < 1e-6)
    // Midpoint.
    let mid = blockDiffusionTemperature(curStep: 24, maxSteps: 48, tMin: 0.4, tMax: 0.8)
    #expect(abs(mid - 0.6) < 1e-6)
}

@Test(.serialized)
func testCanvasTokenEntropy() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // Near-one-hot logits → entropy ≈ 0; uniform logits → entropy ≈ ln(V).
    let vocab = 16
    var oneHot = [Float](repeating: -1e9, count: vocab)
    oneHot[3] = 0
    let uniform = [Float](repeating: 0, count: vocab)
    let logits = MLXArray(oneHot + uniform).reshaped(1, 2, vocab)

    let entropy = canvasTokenEntropy(processedLogits: logits)
    #expect(entropy.shape == [1, 2])
    let values = entropy[0].asArray(Float.self)
    #expect(values[0] < 1e-3)
    #expect(abs(values[1] - log(Float(vocab))) < 1e-3)
}

@Test(.serialized)
func testEntropyBoundAcceptMask() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // Acceptance at sorted rank k requires the SUM of all lower entropies
    // to stay within the bound (cum_k − sorted_k = Σ_{i<k}).
    // Entropies [0.0, 5.0, 0.2]: ranks give Σ_{i<k} = [0, 0, 0.2]
    // → with bound 0.1 accept positions 0 and 2, reject the 5.0 token.
    let entropy = MLXArray([Float(0.0), 5.0, 0.2]).reshaped(1, 3)
    let mask = entropyBoundAcceptMask(tokenEntropy: entropy, entropyBound: 0.1)
    #expect(mask.shape == [1, 3])
    let accepted = mask[0].asArray(Bool.self)
    #expect(accepted == [true, false, true])

    // The lowest-entropy token is always accepted, even when above the bound.
    let high = MLXArray([Float(3.0), 4.0]).reshaped(1, 2)
    let highMask = entropyBoundAcceptMask(tokenEntropy: high, entropyBound: 0.1)
    #expect(highMask[0].asArray(Bool.self) == [true, false])
}

@Test
func testStableConfidentStopper() {
    var stopper = StableConfidentStopper(stabilityThreshold: 1, confidenceThreshold: 0.005)

    let canvasA: [Int32] = [1, 2, 3]
    let canvasB: [Int32] = [1, 2, 4]

    // First step: no history yet → not stable, even when confident.
    #expect(stopper.shouldStop(argmaxCanvas: canvasA, meanEntropy: 0.001) == false)
    // Second step, same canvas, confident → stop.
    #expect(stopper.shouldStop(argmaxCanvas: canvasA, meanEntropy: 0.001) == true)

    stopper.reset()
    #expect(stopper.shouldStop(argmaxCanvas: canvasA, meanEntropy: 0.001) == false)
    // Canvas changed → not stable.
    #expect(stopper.shouldStop(argmaxCanvas: canvasB, meanEntropy: 0.001) == false)
    // Stable but not confident → keep denoising.
    #expect(stopper.shouldStop(argmaxCanvas: canvasB, meanEntropy: 0.5) == false)
    // Stable and confident → stop.
    #expect(stopper.shouldStop(argmaxCanvas: canvasB, meanEntropy: 0.0001) == true)
}

// MARK: - generation_config.json diffusion fields (Task 4)

@Test
func testGenerationConfigFileDecodesDiffusionFields() throws {
    // Verbatim from the local DiffusionGemma bundles.
    let json = """
        {
          "confidence_threshold": 0.005,
          "eos_token_id": [1, 106, 50],
          "max_denoising_steps": 48,
          "max_new_tokens": 256,
          "pad_token_id": 0,
          "sampler_config": {
            "_cls_name": "EntropyBoundSamplerConfig",
            "entropy_bound": 0.1
          },
          "stability_threshold": 1,
          "t_max": 0.8,
          "t_min": 0.4,
          "transformers_version": "5.8.0.dev0"
        }
        """
    let config = try JSONDecoder().decode(
        GenerationConfigFile.self, from: Data(json.utf8))
    #expect(config.eosTokenIds?.values == [1, 106, 50])
    #expect(config.maxNewTokens == 256)
    #expect(config.maxDenoisingSteps == 48)
    #expect(config.samplerConfig?.entropyBound == 0.1)
    #expect(config.tMin == 0.4)
    #expect(config.tMax == 0.8)
    #expect(config.stabilityThreshold == 1)
    #expect(config.confidenceThreshold == 0.005)
    #expect(config.padTokenId == 0)

    let defaults = BlockDiffusionParameters(canvasLength: 256)
        .resolving(generationConfig: config)
    #expect(defaults.canvasLength == 256)
    #expect(defaults.maxDenoisingSteps == 48)
    #expect(defaults.entropyBound == 0.1)
    #expect(defaults.eosTokenIds == Set([1, 106, 50]))
    #expect(defaults.padTokenId == 0)
}
