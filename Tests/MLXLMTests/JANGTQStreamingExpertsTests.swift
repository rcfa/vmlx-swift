import Foundation
import Testing
@testable import MLXLMCommon

@Suite("JANGTQ streaming expert index", .serialized)
struct JANGTQStreamingExpertsTests {
    @Test("Qwen35 MoE JANGTQ bundles auto-enable streaming for large stacked experts")
    func qwen35MoEJANGTQAutoEnablesLargeStackedExperts() throws {
        let bundle = try Self.makeBundle(name: "qwen35")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeQwen35Config(at: bundle, numExperts: 512)
        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: Self.qwen35SwitchMLPTensors(layer: 0, experts: 512))

        try Self.withStreamingEnv(value: nil) {
            #expect(JANGTQStreamingExperts.shouldAutoEnableQwen35MoE(modelDirectory: bundle))
        }
    }

    @Test("explicit streaming disable wins over Qwen35 auto-enable")
    func explicitStreamingDisableWinsOverQwen35AutoEnable() throws {
        let bundle = try Self.makeBundle(name: "qwen35-disabled")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeQwen35Config(at: bundle, numExperts: 512)
        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: Self.qwen35SwitchMLPTensors(layer: 0, experts: 512))

        try Self.withStreamingEnv(value: "0") {
            #expect(!JANGTQStreamingExperts.shouldAutoEnableQwen35MoE(modelDirectory: bundle))
        }
    }

    @Test("prestacked switch_mlp tensors are streamable")
    func prestackedSwitchMLPTensorsAreStreamable() throws {
        let bundle = try Self.makeBundle(name: "prestacked")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: [
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.up_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.up_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.down_proj.tq_packed",
                    dtype: "U32", shape: [2, 2, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.down_proj.tq_norms",
                    dtype: "F16", shape: [2, 2]),
            ])

        #expect(JANGTQStreamingExperts.hasStreamableExperts(in: bundle))
    }

    @Test("Zaya zaya_block switch_mlp tensors are streamable")
    func zayaSwitchMLPTensorsAreStreamable() throws {
        let bundle = try Self.makeBundle(name: "zaya")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: [
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.gate_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.gate_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.up_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.up_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.down_proj.tq_packed",
                    dtype: "U32", shape: [2, 2, 1]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.down_proj.tq_norms",
                    dtype: "F16", shape: [2, 2]),
            ])

        #expect(JANGTQStreamingExperts.hasStreamableExperts(in: bundle))
    }

    @Test("legacy per-expert tensors remain streamable")
    func perExpertTensorsRemainStreamable() throws {
        let bundle = try Self.makeBundle(name: "per-expert")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: [
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.experts.0.w1.tq_packed",
                    dtype: "U32", shape: [3, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.experts.0.w1.tq_norms",
                    dtype: "F16", shape: [3]),
            ])

        #expect(JANGTQStreamingExperts.hasStreamableExperts(in: bundle))
    }

    private struct TensorSpec {
        var name: String
        var dtype: String
        var shape: [Int]
    }

    private static func makeBundle(name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("jangtq-streaming-\(name)-\(UUID().uuidString)",
                isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func writeQwen35Config(at bundle: URL, numExperts: Int) throws {
        let config: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "text_config": [
                "model_type": "qwen3_5_moe_text",
                "num_experts": numExperts,
                "num_experts_per_tok": 10,
            ],
        ]
        let jangConfig: [String: Any] = [
            "format": "jangtq",
            "weight_format": "mxtq",
            "profile": "JANGTQ2",
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: bundle.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: jangConfig, options: [.sortedKeys])
            .write(to: bundle.appendingPathComponent("jang_config.json"))
    }

    private static func qwen35SwitchMLPTensors(layer: Int, experts: Int) -> [TensorSpec] {
        [
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.gate_proj.tq_packed",
                dtype: "U32", shape: [experts, 3, 1]),
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.gate_proj.tq_norms",
                dtype: "F16", shape: [experts, 3]),
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.up_proj.tq_packed",
                dtype: "U32", shape: [experts, 3, 1]),
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.up_proj.tq_norms",
                dtype: "F16", shape: [experts, 3]),
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.down_proj.tq_packed",
                dtype: "U32", shape: [experts, 2, 1]),
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.down_proj.tq_norms",
                dtype: "F16", shape: [experts, 2]),
        ]
    }

    private static func withStreamingEnv(value: String?, body: () throws -> Void) throws {
        try withEnv("MLXPRESS_STREAMING_EXPERTS", value: value) {
            try withEnv("JANGPRESS_STREAMING_EXPERTS", value: value, body: body)
        }
    }

    private static func withEnv(_ key: String, value: String?, body: () throws -> Void) throws {
        let previous = getenv(key).map { String(cString: $0) }
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let previous {
                setenv(key, previous, 1)
            } else {
                unsetenv(key)
            }
        }
        try body()
    }

    private static func writeSafetensors(at url: URL, tensors: [TensorSpec]) throws {
        var header: [String: Any] = [:]
        var offset = 0
        for tensor in tensors {
            let byteCount = tensor.shape.reduce(1, *) * byteWidth(dtype: tensor.dtype)
            header[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, offset + byteCount],
            ]
            offset += byteCount
        }
        let headerJSON = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var data = Data()
        var headerSize = UInt64(headerJSON.count)
        withUnsafeBytes(of: &headerSize) { data.append(contentsOf: $0) }
        data.append(headerJSON)
        data.append(Data(count: offset))
        try data.write(to: url)
    }

    private static func byteWidth(dtype: String) -> Int {
        switch dtype {
        case "F16", "BF16", "I16", "U16": return 2
        case "F32", "I32", "U32": return 4
        case "F64", "I64", "U64": return 8
        default: return 1
        }
    }
}
