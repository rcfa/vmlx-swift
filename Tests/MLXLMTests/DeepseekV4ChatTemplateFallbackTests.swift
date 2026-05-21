// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import VMLXJinja
import MLXLMCommon
import XCTest

private extension Template {
    func renderDSV4(_ context: [String: Any]) throws -> String {
        var values: [String: Value] = [:]
        for (key, value) in context {
            values[key] = try Value(any: value)
        }
        return try render(values)
    }
}

final class DeepseekV4ChatTemplateFallbackTests: XCTestCase {
    func testMinimalFallbackMatchesCanonicalChatModeMultiTurn() throws {
        let template = try Template(ChatTemplateFallbacks.dsv4Minimal)
        let rendered = try template.renderDSV4([
            "messages": [
                ["role": "user", "content": "Turn 1."],
                ["role": "assistant", "content": "Answer 1."],
                ["role": "user", "content": "Turn 2."],
            ],
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])

        let canonical = DeepseekV4ChatEncoder().encode(
            messages: [
                .init(role: .user, content: "Turn 1."),
                .init(role: .assistant, content: "Answer 1."),
                .init(role: .user, content: "Turn 2."),
            ],
            thinkingMode: .chat)

        XCTAssertEqual(rendered, canonical)
        XCTAssertTrue(
            rendered.contains(
                "<\u{FF5C}User\u{FF5C}>Turn 1.<\u{FF5C}Assistant\u{FF5C}></think>Answer 1.<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>"),
            rendered.debugDescription)
        XCTAssertTrue(
            rendered.hasSuffix("<\u{FF5C}User\u{FF5C}>Turn 2.<\u{FF5C}Assistant\u{FF5C}></think>"),
            rendered.debugDescription)
    }

    func testMinimalFallbackMatchesCanonicalThinkingModeWithDroppedEarlierReasoning() throws {
        let template = try Template(ChatTemplateFallbacks.dsv4Minimal)
        let rendered = try template.renderDSV4([
            "messages": [
                ["role": "user", "content": "Turn 1."],
                [
                    "role": "assistant",
                    "reasoning_content": "Earlier reasoning must be dropped.",
                    "content": "Answer 1.",
                ],
                ["role": "user", "content": "Turn 2."],
            ],
            "add_generation_prompt": true,
            "enable_thinking": true,
            "reasoning_effort": "high",
        ])

        let canonical = DeepseekV4ChatEncoder().encode(
            messages: [
                .init(role: .user, content: "Turn 1."),
                .init(
                    role: .assistant,
                    content: "Answer 1.",
                    reasoningContent: "Earlier reasoning must be dropped."),
                .init(role: .user, content: "Turn 2."),
            ],
            thinkingMode: .thinking,
            reasoningEffort: .high,
            dropEarlierReasoning: true)

        XCTAssertEqual(rendered, canonical)
        XCTAssertFalse(rendered.contains("Earlier reasoning must be dropped."))
        XCTAssertTrue(
            rendered.hasSuffix("<\u{FF5C}User\u{FF5C}>Turn 2.<\u{FF5C}Assistant\u{FF5C}><think>"),
            rendered.debugDescription)
    }

    func testMinimalFallbackMatchesCanonicalToolResultMerge() throws {
        let template = try Template(ChatTemplateFallbacks.dsv4Minimal)
        let rendered = try template.renderDSV4([
            "messages": [
                ["role": "user", "content": "Check Paris weather."],
                [
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [[
                        "id": "call_weather",
                        "type": "function",
                        "name": "get_weather",
                        "arguments": ["city": "Paris"],
                        "function": [
                            "name": "get_weather",
                            "arguments": ["city": "Paris"],
                        ],
                    ]],
                ],
                ["role": "tool", "tool_call_id": "call_weather", "content": "{\"temp_c\":18}"],
                ["role": "user", "content": "Summarize."],
            ],
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])

        let canonical = DeepseekV4ChatEncoder().encode(
            messages: [
                .init(role: .user, content: "Check Paris weather."),
                .init(
                    role: .assistant,
                    content: "Checking.",
                    toolCalls: [
                        .init(
                            id: "call_weather",
                            name: "get_weather",
                            arguments: "{\"city\":\"Paris\"}")
                    ]),
                .init(role: .tool, content: "{\"temp_c\":18}", toolCallId: "call_weather"),
                .init(role: .user, content: "Summarize."),
            ],
            thinkingMode: .chat)

        XCTAssertEqual(rendered, canonical)
        XCTAssertTrue(
            rendered.contains(
                "<tool_result>{\"temp_c\":18}</tool_result>\n\nSummarize."),
            rendered.debugDescription)
        XCTAssertFalse(
            rendered.contains(
                "<tool_result>{\"temp_c\":18}</tool_result><\u{FF5C}User\u{FF5C}>Summarize."),
            rendered.debugDescription)
    }

}
