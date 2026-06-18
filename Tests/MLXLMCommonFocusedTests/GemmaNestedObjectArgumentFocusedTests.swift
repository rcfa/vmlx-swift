// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Testing
@testable import MLXLMCommon

/// Regression contracts for nested object arguments in the Gemma function-call
/// format (`call:name{key:{nested:value}}`).
///
/// Gemma emits a nested object argument (e.g. an agentic `target:{mark:1}`
/// field, or any tool whose schema declares an `object` parameter) in the same
/// `key:value` body syntax as the top-level call. Before the fix `parseValue`
/// had branches only for escaped strings and arrays, so a `{...}` value fell
/// through to the raw-scalar path and was captured as an undecodable string
/// (`"{mark:1}"`). A tool expecting an object then received a string and
/// rejected the call. These tests pin the object branch (and the matching
/// `}` raw-scalar terminator) so nested objects parse as real objects.
@Suite("Gemma nested object argument contracts")
struct GemmaNestedObjectArgumentFocusedTests {
    @Test("Gemma4 parses a nested object argument as an object, not a string")
    func gemma4ParsesNestedObjectArgument() throws {
        let output =
            #"<|tool_call>call:agent_action{note:<|"|>Type into Title<|"|>,target:{mark:1},text:<|"|>hello world<|"|>,verb:<|"|>type<|"|>}<tool_call|>"#
        let call = try #require(
            ToolCallFormat.gemma4.createParser().parse(
                content: output,
                tools: [agentActionToolSpec()]
            )
        )

        #expect(call.function.name == "agent_action")
        // The nested object survives as an object. Scalar typing (the `1`) is
        // intentionally left to the same downstream schema-aware coercion that
        // already types top-level scalars — the parser keeps bare scalars as
        // strings everywhere for consistency.
        #expect(call.function.arguments["target"] == .object(["mark": .string("1")]))
        #expect(call.function.arguments["text"] == .string("hello world"))
        #expect(call.function.arguments["verb"] == .string("type"))
        #expect(call.function.arguments["note"] == .string("Type into Title"))
    }

    @Test("Gemma4 parses an escaped string value inside a nested object")
    func gemma4ParsesEscapedStringInsideNestedObject() throws {
        let output =
            #"<|tool_call>call:agent_action{target:{describe:<|"|>the Send button<|"|>},verb:<|"|>click<|"|>}<tool_call|>"#
        let call = try #require(
            ToolCallFormat.gemma4.createParser().parse(
                content: output,
                tools: [agentActionToolSpec()]
            )
        )

        #expect(call.function.name == "agent_action")
        #expect(
            call.function.arguments["target"] == .object(["describe": .string("the Send button")]))
        #expect(call.function.arguments["verb"] == .string("click"))
    }

    @Test("Gemma4 parses a nested object with multiple keys")
    func gemma4ParsesNestedObjectWithMultipleKeys() throws {
        let output = #"<|tool_call>call:move{from:{x:1,y:2},to:{x:3,y:4}}<tool_call|>"#
        let call = try #require(
            ToolCallFormat.gemma4.createParser().parse(content: output, tools: nil))

        #expect(call.function.name == "move")
        #expect(call.function.arguments["from"] == .object(["x": .string("1"), "y": .string("2")]))
        #expect(call.function.arguments["to"] == .object(["x": .string("3"), "y": .string("4")]))
    }

    @Test("Gemma4 parses a nested object as the final argument")
    func gemma4ParsesNestedObjectAsFinalArgument() throws {
        let output = #"<|tool_call>call:agent_action{verb:<|"|>click<|"|>,target:{mark:2}}<tool_call|>"#
        let call = try #require(
            ToolCallFormat.gemma4.createParser().parse(
                content: output,
                tools: [agentActionToolSpec()]
            )
        )

        #expect(call.function.arguments["verb"] == .string("click"))
        #expect(call.function.arguments["target"] == .object(["mark": .string("2")]))
    }

    @Test("Gemma3 native format also parses nested object arguments")
    func gemma3ParsesNestedObjectArgument() throws {
        let output = "<start_function_call>call:agent_action{target:{mark:1}}<end_function_call>"
        let call = try #require(
            ToolCallFormat.gemma.createParser().parse(
                content: output,
                tools: [agentActionToolSpec()]
            )
        )

        #expect(call.function.arguments["target"] == .object(["mark": .string("1")]))
    }

    @Test("Gemma4 still parses flat scalar/array/string arguments unchanged")
    func gemma4StillParsesFlatArgumentsUnchanged() throws {
        let output =
            #"<|tool_call>call:line_count{text:<|"|>one\ntwo<|"|>,tags:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>"#
        let call = try #require(
            ToolCallFormat.gemma4.createParser().parse(
                content: output,
                tools: [lineCountToolSpec()]
            )
        )

        #expect(call.function.arguments["text"] == .string("one\ntwo"))
        #expect(call.function.arguments["tags"] == .array([.string("a"), .string("b")]))
    }

    private func agentActionToolSpec() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "agent_action",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "verb": ["type": "string"] as [String: any Sendable],
                        "text": ["type": "string"] as [String: any Sendable],
                        "note": ["type": "string"] as [String: any Sendable],
                        "target": [
                            "type": "object",
                            "properties": [
                                "mark": ["type": "integer"] as [String: any Sendable],
                                "describe": ["type": "string"] as [String: any Sendable],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["verb"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    private func lineCountToolSpec() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "line_count",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "text": ["type": "string"] as [String: any Sendable],
                        "tags": [
                            "type": "array",
                            "items": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }
}
