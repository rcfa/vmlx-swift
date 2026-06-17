//
//  BPETokenizer.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 18/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import VMLXHub

/// A pair of byte/token strings used in Byte-Pair Encoding (BPE) merge operations.
struct BytePair: Hashable, Sendable {
    let a: String
    let b: String
    init(_ a: String, _ b: String) {
        self.a = a
        self.b = b
    }

    init(tuple: [String]) {
        a = tuple[0]
        b = tuple[1]
    }

    static func == (lhs: BytePair, rhs: BytePair) -> Bool {
        lhs.a == rhs.a && lhs.b == rhs.b
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(a)
        hasher.combine(b)
    }
}

/// A Byte-Pair Encoding (BPE) tokenizer implementation.
///
/// BPE tokenizers learn to merge the most frequently occurring pairs of characters
/// or character sequences. This implementation supports various BPE-based models
/// including GPT-2, RoBERTa, and other transformer models.
class BPETokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    let bpeRanks: [BytePair: Int]
    private let tokensToIds: [NSString: Int]
    private let idsToTokens: [Int: NSString]

    /// The total number of tokens in the vocabulary.
    var vocabCount: Int { tokensToIds.count }

    /// The beginning-of-sequence token string, if defined.
    let bosToken: String?

    /// The numeric ID of the beginning-of-sequence token, if defined.
    let bosTokenId: Int?

    /// The end-of-sequence token string, if defined.
    let eosToken: String?

    /// The numeric ID of the end-of-sequence token, if defined.
    let eosTokenId: Int?

    /// The unknown token string used for out-of-vocabulary words.
    let unknownToken: String?

    /// The numeric ID of the unknown token.
    let unknownTokenId: Int?

    /// Whether consecutive unknown tokens should be fused together.
    let fuseUnknownTokens: Bool

    static func mergesFromConfig(_ config: Config?) -> [[String]]? {
        guard let config else { return nil }

        if let merges = config.array() {
            return merges.reduce(into: [[String]]()) { result, element in
                if let val: [String] = element.get() { // New format (pushed with tokenizers >= 0.20.0): each merge is a list of 2 items
                    result.append(val)
                }
                if let val: String = element.get() { // legacy
                    result.append(val.unicodeScalars.split(separator: " ", omittingEmptySubsequences: false).map { String($0) })
                }
            }
        }

        return nil
    }

    /// Initializes a BPE tokenizer from configuration data.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data containing vocabulary and merges
    ///   - addedTokens: Additional tokens to include in the vocabulary
    /// - Throws: `TokenizerError` if required configuration is missing
    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        guard let merges = Self.mergesFromConfig(tokenizerData.model.merges) else { fatalError("BPETokenizer requires merges") }
        guard let vocab = tokenizerData.model.vocab.dictionary() else {
            throw TokenizerError.missingVocab
        }
        var bpeRanks: [BytePair: Int] = [:]
        for (i, merge) in merges.enumerated() {
            let bp = BytePair(tuple: merge)
            bpeRanks[bp] = i
        }
        self.bpeRanks = bpeRanks

        let addedTokens = addedTokens.reduce(into: [BinaryDistinctString: Config]()) { result, element in
            result[BinaryDistinctString(element.key)] = .init(element.value)
        }
        tokensToIds = vocab.merging(addedTokens) { $1 }.reduce(into: [NSString: Int]()) { result, element in
            result[element.key.nsString] = element.value.integer()
        }

        idsToTokens = tokensToIds.reduce(into: [Int: NSString]()) { result, element in
            result[element.value] = element.key
        }

        // Populate tokens
        if let unknownToken = TokenizerModel.unknownToken(from: tokenizerConfig) {
            self.unknownToken = unknownToken
            unknownTokenId = tokensToIds[unknownToken as NSString]
        } else {
            unknownToken = nil
            unknownTokenId = nil
        }

        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]

        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken == nil ? nil : tokensToIds[bosToken! as NSString]

        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)
    }

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID, or the unknown token ID if not found
    func convertTokenToId(_ token: String) -> Int? {
        tokensToIds[token as NSString] ?? unknownTokenId
    }

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is invalid
    func convertIdToToken(_ id: Int) -> String? {
        idsToTokens[id] as String?
    }

    func byteEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.map { token -> String in
            return Array(token.utf8).compactMap { byteEncoder[$0] }.joined()
        }
    }

    func hexaEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.flatMap { token -> [String] in
            return Array(token.utf8).map { String(format: "<0x%02X>", $0) }
        }
    }

    private func getPairs(word: [String]) -> Set<BytePair> {
        var s = Set<BytePair>()
        for i in 0..<word.count - 1 {
            let bp = BytePair(
                word[i],
                word[i + 1]
            )
            s.insert(bp)
        }
        return s
    }

    /// Byte-Pair Encoding of a single pre-token.
    ///
    /// Linear-ish merge: a doubly-linked list of symbols plus a min-heap of
    /// candidate adjacent merges keyed by `(rank, leftIndex)`. Each merge is
    /// O(log n) and there are O(n) merges, so an N-symbol token is O(n log n)
    /// rather than O(n²) — this matters for long whitespace-free pre-tokens
    /// (e.g. compact tool JSON), which a naive per-round rescan tokenizes in
    /// seconds.
    ///
    /// Merges happen in rank order, and a pair formed by a merge always
    /// outranks the merge that created its components, so popping the
    /// globally-lowest `(rank, leftIndex)` reproduces the canonical BPE result
    /// ("merge all occurrences of the min-rank pair per round, left to right").
    /// Stale heap entries (a node consumed by an earlier merge, or whose pair
    /// rank no longer matches the candidate) are skipped on pop.
    func bpe(token: String) -> String {
        let parts0 = Array(token).map { String($0) }
        let n = parts0.count
        if n <= 1 { return token }

        var parts = parts0
        var prev = [Int](repeating: 0, count: n)
        var next = [Int](repeating: 0, count: n)
        var alive = [Bool](repeating: true, count: n)
        for i in 0..<n {
            prev[i] = i - 1
            next[i] = (i + 1 == n) ? -1 : (i + 1)
        }

        // Binary min-heap of (rank, left), ordered by rank then left index.
        var heap: [(rank: Int, left: Int)] = []
        heap.reserveCapacity(n)
        func before(_ a: (rank: Int, left: Int), _ b: (rank: Int, left: Int)) -> Bool {
            a.rank != b.rank ? a.rank < b.rank : a.left < b.left
        }
        func push(_ left: Int) {
            guard left >= 0 else { return }
            let r = next[left]
            guard r >= 0, let rank = bpeRanks[BytePair(parts[left], parts[r])] else { return }
            heap.append((rank, left))
            var i = heap.count - 1
            while i > 0 {
                let parent = (i - 1) / 2
                if before(heap[i], heap[parent]) {
                    heap.swapAt(i, parent)
                    i = parent
                } else {
                    break
                }
            }
        }
        func pop() -> (rank: Int, left: Int)? {
            guard let top = heap.first else { return nil }
            let last = heap.removeLast()
            if !heap.isEmpty {
                heap[0] = last
                var i = 0
                let count = heap.count
                while true {
                    let l = 2 * i + 1
                    let r = 2 * i + 2
                    var m = i
                    if l < count, before(heap[l], heap[m]) { m = l }
                    if r < count, before(heap[r], heap[m]) { m = r }
                    if m == i { break }
                    heap.swapAt(i, m)
                    i = m
                }
            }
            return top
        }

        for i in 0..<n where next[i] >= 0 { push(i) }

        while let cand = pop() {
            let l = cand.left
            guard alive[l] else { continue }
            let r = next[l]
            guard r >= 0, alive[r] else { continue }
            // Skip stale entries: the live pair at this node must still carry the
            // rank this candidate was queued with.
            guard let curRank = bpeRanks[BytePair(parts[l], parts[r])], curRank == cand.rank
            else { continue }

            // Merge r into l; r leaves the list.
            parts[l] = parts[l] + parts[r]
            alive[r] = false
            let rn = next[r]
            next[l] = rn
            if rn >= 0 { prev[rn] = l }

            // New adjacencies created by the merge.
            push(prev[l])
            push(l)
        }

        // Walk the surviving list from the head (node 0 is never consumed —
        // it is never the right element of any merge).
        var result: [String] = []
        result.reserveCapacity(n)
        var i = 0
        while i >= 0 {
            result.append(parts[i])
            i = next[i]
        }
        return result.joined(separator: " ")
    }

    /// Tokenizes input text using the BPE algorithm.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of BPE token strings
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        let bpeTokens = bpe(token: text).split(separator: " ").map { String($0) }
        for token in bpeTokens {
            if convertTokenToId(token) != unknownTokenId {
                tokens.append(token)
            } else {
                // TODO: if config.byte_fallback is False, append the unknown token instead
                tokens.append(contentsOf: hexaEncode(text: token))
            }
        }
        return tokens
    }
}
