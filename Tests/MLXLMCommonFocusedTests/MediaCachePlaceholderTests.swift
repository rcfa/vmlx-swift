// Copyright © 2026 Osaurus AI. All rights reserved.

import MLX
import Testing

@testable import MLXLMCommon

@Suite("media cache placeholder policy")
struct MediaCachePlaceholderTests {
    @Test("known media tokens allow text-only cache-hit suffixes")
    func knownMediaTokensAllowTextOnlySuffixes() {
        FocusedMLXTestSupport.withLock {
        let waveform = MLXArray([Float](repeating: 0.0, count: 8)).reshaped([1, 8])
        MLX.eval(waveform)

        let unknown = LMInput(
            text: .init(tokens: MLXArray([Int32(1), 27, 2])),
            audio: .init(waveform: waveform))
        #expect(!unknown.cacheHitSuffixContainsMediaPlaceholder([]))
        #expect(unknown.cacheHitSuffixContainsMediaPlaceholder([1, 2]))

        let known = LMInput(
            text: .init(tokens: MLXArray([Int32(1), 27, 2])),
            audio: .init(waveform: waveform),
            mediaTokenIds: [27])
        #expect(known.cacheHitSuffixContainsMediaPlaceholder([27, 2]))
        #expect(!known.cacheHitSuffixContainsMediaPlaceholder([2, 3]))
        #expect(!LMInput(
            text: .init(tokens: MLXArray([Int32(1)])),
            mediaTokenIds: [27]
        ).cacheHitSuffixContainsMediaPlaceholder([27]))
        }
    }
}
