import Foundation
import Testing
@testable import MLXLMCommon

@Suite("Gemma4 tool-marker stripping with no tools in scope")
struct Gemma4NoToolsMarkerStripTests {
    // Reproduces the leak: thinking-on + zero tools offered + model
    // hallucinates a Gemma tool call -> control markers must NOT reach
    // visible text. Feeds the exact streamed body observed live.
    @Test("Gemma4 control markers are stripped from visible text when no tools are offered")
    func stripsMarkersWithoutTools() {
        let processor = ToolCallProcessor(format: .gemma4, tools: nil, stripOnly: true)
        var visible = ""
        for chunk in ["<|tool_call>", "call:osaurus_status{}", "<tool_call|>"] {
            if let out = processor.processChunk(chunk) { visible += out }
        }
        if let tail = processor.processEOS() { visible += tail }
        #expect(!visible.contains("<|tool_call>"), "leaked start marker: \(visible)")
        #expect(!visible.contains("<tool_call|>"), "leaked end marker: \(visible)")
        #expect(!visible.contains("call:"), "leaked bare-call marker: \(visible)")
        // No tools were offered, so no tool call should be surfaced.
        #expect(processor.toolCalls.isEmpty, "must not fabricate tool calls when none offered")
    }
}
