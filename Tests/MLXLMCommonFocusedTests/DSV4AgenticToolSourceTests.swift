// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@Suite("DSV4 agentic tool source contracts")
struct DSV4AgenticToolSourceTests {
    @Test("agentic row enforces DSML, think_xml, bundle defaults, and L2 cache proof")
    func agenticRowEnforcesDSMLThinkXMLBundleDefaultsAndL2CacheProof() throws {
        let bench = try String(contentsOfFile: "RunBench/Bench.swift", encoding: .utf8)

        #expect(bench.contains("BENCH_DSV4_AGENTIC_TOOL"))
        #expect(bench.contains("func runDSV4AgenticToolCheck("))
        #expect(bench.contains("context.configuration.toolCallFormat == .dsml"))
        #expect(bench.contains("context.configuration.reasoningParserName == \"think_xml\""))
        #expect(bench.contains("generationConfig: ctx.configuration.generationDefaults"))
        #expect(bench.contains(".assistant(toolText, toolCalls: toolCalls)"))
        #expect(bench.contains(".tool(toolResultJSON, toolCallId: toolCallId)"))
        #expect(bench.contains("markerLeaks(in: result.text)"))
        #expect(bench.contains("snapshot.isPagedIncompatible"))
        #expect(bench.contains("diskStats.stores > 0"))
        #expect(bench.contains("diskStats.hits > 0"))
        #expect(!bench.contains("BENCH_DSV4_AGENTIC_TEMP\"] ??"))
        #expect(!bench.contains("BENCH_DSV4_AGENTIC_REPETITION_PENALTY\"] ??"))
    }
}
