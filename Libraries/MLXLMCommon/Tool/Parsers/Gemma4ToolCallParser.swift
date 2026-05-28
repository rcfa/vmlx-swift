// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Gemma-4 tool-call parser.
///
/// Gemma-4 bundles normally use the native `<|tool_call>...<tool_call|>`
/// function-call envelope, but live JANG/CRACK rows can emit the same
/// Zyphra XML function envelope used by ZAYA:
///
/// `<zyphra_tool_call><function=name><parameter=key>...</parameter></function></zyphra_tool_call>`
///
/// Accept both transports as real tool-call protocol. This is parser
/// recognition only: it does not inject tools, rewrite prompts, change sampler
/// defaults, or normalize argument values beyond the XML parser's schema-aware
/// string handling.
public struct Gemma4ToolCallParser: ToolCallParser, Sendable {
    private let native = GemmaFunctionParser(
        startTag: "<|tool_call>",
        endTag: "<tool_call|>",
        escapeMarker: "<|\"|>")
    private let zyphra = XMLFunctionParser(
        startTag: "<zyphra_tool_call>",
        endTag: "</zyphra_tool_call>",
        decodesHTMLLineBreaks: true,
        unwrapJSONQuotedStringParameters: true)

    public init() {}

    public var startTag: String? { native.startTag }
    public var endTag: String? { native.endTag }

    public var startTagAliases: [String] {
        native.startTagAliases + zyphra.startTagAliases
    }

    public var endTagAliases: [String] {
        native.endTagAliases + zyphra.endTagAliases
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        native.parse(content: content, tools: tools)
            ?? zyphra.parse(content: content, tools: tools)
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        let nativeCalls = native.parseEOS(toolCallBuffer, tools: tools)
        if !nativeCalls.isEmpty { return nativeCalls }
        return zyphra.parseEOS(toolCallBuffer, tools: tools)
    }
}
