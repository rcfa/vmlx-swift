import Foundation
import MLXLLM
import Testing

@testable import MLXLMCommon

/// Pins the gemma `<end_of_turn>` `extraEOSTokens` contract: gemma-4 must NOT declare it, gemma-3 must.
///
/// These live in their own file rather than alongside `StopStringMatcherTests` on purpose. They and the
/// `generation_config.json` eos-gate tests are offered as two independent upstream PRs, so each patch has
/// to apply to a pristine tree on its own; if both appended to the tail of the same suite they could not
/// be applied sequentially (an add/add conflict at the same anchor). Separate files keep them orthogonal
/// in any order, at the cost of the small tokenizer stub below.
@Suite("gemma <end_of_turn> stale extraEOSTokens")
struct Gemma4StaleEOSTests {

    /// A tokenizer that resolves nothing — standing in for gemma-4's, which has no `<end_of_turn>`.
    private struct NoSpecialTokenTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// Gemma-4 renamed its turn-end token, so `<end_of_turn>` is absent from its vocabulary.
    /// Declaring it in `extraEOSTokens` is NOT inert: that loop passes `allowExactTextFallback: true`,
    /// so an entry the tokenizer cannot resolve becomes a live TEXT stop-string — able to truncate a
    /// legitimate answer that happens to type those characters.
    @Test("gemma-4 configs declare no <end_of_turn> extra EOS token")
    func gemma4ConfigsDropStaleTurnEnd() {
        for config in [
            LLMRegistry.gemma4_27b_it_4bit,
            LLMRegistry.gemma4_12b_it_4bit,
            LLMRegistry.gemma4_27b_it_qat_4bit,
        ] {
            #expect(!config.extraEOSTokens.contains("<end_of_turn>"))
        }
    }

    /// `<end_of_turn>` IS still the real turn-end token for gemma-3 / gemma-3n — it must stay.
    @Test("gemma-3 configs keep <end_of_turn>")
    func gemma3ConfigsKeepTurnEnd() {
        #expect(LLMRegistry.gemma3_1B_qat_4bit.extraEOSTokens.contains("<end_of_turn>"))
        #expect(LLMRegistry.gemma3n_E4B_it_lm_bf16.extraEOSTokens.contains("<end_of_turn>"))
    }

    /// Behavioural pin: on a tokenizer that cannot resolve `<end_of_turn>` (as gemma-4's cannot),
    /// gemma-4 must yield no such text stop-string, while gemma-3 still must.
    @Test("<end_of_turn> resolves to a live text stop for gemma-3 but not gemma-4")
    func turnEndTextStopOnlyForGemma3() {
        let gemma4 = resolveStopSequences(
            modelConfiguration: LLMRegistry.gemma4_27b_it_4bit,
            tokenizer: NoSpecialTokenTokenizer())
        #expect(!gemma4.textStopStrings.contains("<end_of_turn>"))

        let gemma3 = resolveStopSequences(
            modelConfiguration: LLMRegistry.gemma3_1B_qat_4bit,
            tokenizer: NoSpecialTokenTokenizer())
        #expect(gemma3.textStopStrings.contains("<end_of_turn>"))
    }
}
