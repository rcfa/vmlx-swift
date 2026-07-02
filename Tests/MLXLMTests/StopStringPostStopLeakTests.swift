// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// Regression tests for the post-stop-string leak: after a stop string
/// matched mid-stream, `TextToolTokenLoopHandler.onGenerationEnd` used to
/// flush the detokenizer's ~24-char held-back tail — text chronologically
/// AFTER the match — back through a stop matcher whose buffer had been
/// cleared at match time, appending mangled post-stop text to the response.
///
/// Live repro (osaurus solo path, Mistral-Small, temperature 0):
///   stop=["five"]  → "one, two, three, four,  six, "   (post-stop " six, " leaked)
///   stop=["three"] → "one,two,our,fi"                  (post-stop tail leaked mangled)
struct StopStringPostStopLeakTests {

    /// Deterministic tokenizer: each token id maps to a fixed text piece;
    /// decode is a plain concatenation. This makes the
    /// `NaiveStreamingDetokenizer` 24-char hold-back window (the leak's
    /// raw material) fully predictable.
    struct FixedPieceTokenizer: MLXLMCommon.Tokenizer {
        let pieces: [String]

        var vocabularySize: Int { pieces.count }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { pieces.indices.contains($0) ? pieces[$0] : "" }.joined()
        }

        func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }
        func convertIdToToken(_ id: Int) -> String? {
            pieces.indices.contains(id) ? pieces[id] : nil
        }

        var bosToken: String? = nil
        var eosToken: String? = nil
        var eosTokenId: Int? { nil }
        var unknownToken: String? = nil
        var unknownTokenId: Int? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// "one, " ... "ten." as token ids 0...9.
    static let countingPieces = [
        "one, ", "two, ", "three, ", "four, ", "five, ",
        "six, ", "seven, ", "eight, ", "nine, ", "ten.",
    ]

    /// Drive the handler exactly like `generateLoopTask` does: feed tokens
    /// through `onToken` until it returns false (stop hit / terminated),
    /// then call `onGenerationEnd`. Returns the visible content collected
    /// (a) after the token loop and (b) after the end-of-stream flush.
    private func run(
        stopStrings: [String],
        tokens: [Int]
    ) -> (afterLoop: String, afterFlush: String, stopHit: Bool) {
        var handler = TextToolTokenLoopHandler(
            tokenizer: FixedPieceTokenizer(pieces: Self.countingPieces),
            format: .json,
            tools: nil,
            reasoningParser: nil,
            stopStringMatcher: StopStringMatcher(stopStrings: stopStrings)
        )
        var collected = ""
        let emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult = {
            event in
            if case .chunk(let text) = event {
                collected += text
            }
            return .enqueued(remaining: .max)
        }
        for token in tokens {
            if !handler.onToken(token, emit: emit) {
                break
            }
        }
        let afterLoop = collected
        handler.onGenerationEnd(emit: emit)
        return (afterLoop, collected, handler.stopSequenceHit)
    }

    @Test("Stop match mid-stream truncates exactly; end-of-stream flush leaks nothing")
    func postStopTailDoesNotLeak() {
        let result = run(stopStrings: ["three"], tokens: Array(0 ..< 10))
        #expect(result.stopHit)
        // Everything before the match, nothing after it.
        #expect(result.afterLoop == "one, two, ")
        // onGenerationEnd must not append the detokenizer's held tail
        // (post-stop text) after the truncation point.
        #expect(result.afterFlush == "one, two, ")
    }

    @Test("Stop string 'five' — the live osaurus repro shape")
    func liveReproShape() {
        let result = run(stopStrings: ["five"], tokens: Array(0 ..< 10))
        #expect(result.stopHit)
        #expect(result.afterFlush == "one, two, three, four, ")
    }

    @Test("No stop configured: flush emits the full text (no over-suppression)")
    func noStopEmitsEverything() {
        let result = run(stopStrings: [], tokens: Array(0 ..< 10))
        #expect(!result.stopHit)
        #expect(result.afterFlush == Self.countingPieces.joined())
    }

    @Test("Unmatched stop: flush emits the full text and matcher tail in order")
    func unmatchedStopEmitsEverythingInOrder() {
        let result = run(stopStrings: ["XYZZY"], tokens: Array(0 ..< 10))
        #expect(!result.stopHit)
        #expect(result.afterFlush == Self.countingPieces.joined())
    }

    @Test("Stop string arriving only in the end-of-stream flush still truncates")
    func stopInFlushTailStillTruncates() {
        // "ten." sits inside the detokenizer's 24-char hold-back window at
        // end of stream, so the matcher first sees it during
        // onGenerationEnd's detokenizer flush — which must still truncate.
        let result = run(stopStrings: ["ten."], tokens: Array(0 ..< 10))
        #expect(result.afterFlush == Self.countingPieces.dropLast().joined())
    }
}
