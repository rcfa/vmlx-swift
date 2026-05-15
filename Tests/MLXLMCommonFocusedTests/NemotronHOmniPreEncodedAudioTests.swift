// Copyright © 2026 Osaurus AI. All rights reserved.

import MLX
import MLXLMCommon
import MLXVLM
import Testing

private struct FocusedOmniTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [1, 18, 27, 2]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<image>": 18
        case "<so_embedding>": 27
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 18: "<image>"
        case 27: "<so_embedding>"
        default: String(id)
        }
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [1, 18, 27, 2]
    }
}

@Suite("Nemotron H Omni pre-encoded audio")
struct NemotronHOmniPreEncodedAudioTests {
    @Test("live audio buffer keeps full snapshot while streaming chunks")
    func liveAudioBufferSnapshotAndStreamingCursor() {
        let buffer = NemotronHOmniLiveAudioBuffer(sampleRate: 4)

        #expect(buffer.snapshot().samples == [])
        #expect(buffer.durationSeconds == 0)

        buffer.append([1, 2, 3])
        let firstChunk = buffer.consumeAvailableSamples()
        #expect(firstChunk.samples == [1, 2, 3])
        #expect(firstChunk.sampleRate == 4)
        #expect(abs(firstChunk.durationSeconds - 0.75) < 0.0001)
        #expect(buffer.consumeAvailableSamples().samples == [])

        buffer.append([4, 5])
        #expect(buffer.consumeAvailableSamples().samples == [4, 5])
        #expect(buffer.snapshot().samples == [1, 2, 3, 4, 5])

        buffer.resetConsumeCursor()
        #expect(buffer.consumeAvailableSamples().samples == [1, 2, 3, 4, 5])

        buffer.clear()
        #expect(buffer.snapshot().samples == [])
        #expect(buffer.retainedSampleCount == 0)
    }

    @Test("processor preserves caller supplied Parakeet embedding")
    func processorPreservesPreEncodedAudioEmbedding() async throws {
        try await FocusedMLXTestSupport.withLock {
        let processor = NemotronHOmniProcessor(
            NemotronHOmniProcessorConfiguration(),
            tokenizer: FocusedOmniTokenizer())
        let samples = [Float](repeating: 0.05, count: 1_600)
        let embedding = MLXArray.zeros([5, 2_688])

        let input = UserInput(
            prompt: "What did the caller say?",
            audios: [
                .preEncoded(samples: samples, sampleRate: 16_000, embedding: embedding)
            ])
        let lmInput = try await processor.prepare(input: input)

        #expect(lmInput.audio?.waveform.shape == [1, samples.count])
        #expect(lmInput.audio?.sampleRate == 16_000)
        #expect(lmInput.audio?.preEncodedEmbedding?.shape == [5, 2_688])
        #expect(lmInput.mediaTokenIds == [18, 27])
        }
    }
}
