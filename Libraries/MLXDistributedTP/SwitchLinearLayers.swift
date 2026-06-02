import Foundation
import MLX
import MLXLMCommon
import MLXNN

private enum TPSwitchDebug {
    static func enabled(_ key: String) -> Bool {
        switch ProcessInfo.processInfo.environment[key]?.lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

    static var evalProjections: Bool {
        enabled("TP_SWITCH_PROJ_EVAL")
    }

    static var logProjections: Bool {
        enabled("TP_SWITCH_DEBUG") || evalProjections
    }

    static func eval(_ value: MLXArray, name: String) {
        guard evalProjections else { return }
        let start = ProcessInfo.processInfo.systemUptime
        print("[TPSwitch] \(name) eval begin shape=\(value.shape)")
        MLX.eval(value)
        let ms = (ProcessInfo.processInfo.systemUptime - start) * 1000
        print(String(format: "[TPSwitch] %@ eval end %.1fms", name, ms))
    }
}

/// Tensor-parallel routed-expert projection. Each rank holds an output shard
/// for every expert, so the routed result is sharded on the feature axis.
open class AllToShardedSwitchLinear: SwitchLinear {
    public let group: Group
    public var debugName: String?

    public static func from(
        _ linear: SwitchLinear,
        group: Group? = nil,
        segments: Int = 1
    ) -> AllToShardedSwitchLinear {
        let g = group ?? Group(strict: false)
        let shape = linear.weight.shape
        precondition(shape.count == 3, "SwitchLinear weight must be [experts, out, in]")
        let experts = shape[0]
        let out = shape[1]
        let input = shape[2]
        precondition(out % segments == 0, "output dims must be divisible by segments")
        let perSegmentOut = out / segments
        precondition(
            perSegmentOut % g.size == 0,
            "(output dims / segments) must be divisible by group size")

        let perRankOut = perSegmentOut / g.size
        var chunks: [MLXArray] = []
        for segment in 0 ..< segments {
            let segmentStart = segment * perSegmentOut
            let rankStart = segmentStart + g.rank * perRankOut
            let rankEnd = rankStart + perRankOut
            chunks.append(linear.weight[0..., rankStart ..< rankEnd, 0...])
        }
        let shardedWeight = concatenated(chunks, axis: 1)

        var shardedBias: MLXArray? = nil
        if let bias = linear.bias {
            var biasChunks: [MLXArray] = []
            for segment in 0 ..< segments {
                let segmentStart = segment * perSegmentOut
                let rankStart = segmentStart + g.rank * perRankOut
                let rankEnd = rankStart + perRankOut
                biasChunks.append(bias[0..., rankStart ..< rankEnd])
            }
            shardedBias = concatenated(biasChunks, axis: 1)
        }

        return AllToShardedSwitchLinear(
            inputDims: input,
            outputDims: shardedWeight.dim(1),
            numExperts: experts,
            weight: shardedWeight,
            bias: shardedBias,
            group: g)
    }

    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        bias: MLXArray?,
        group: Group
    ) {
        self.group = group
        super.init(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            weight: weight,
            bias: bias)
    }

    public override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        if ProcessInfo.processInfo.environment["TP_LINEAR_DEBUG"] == "1" || TPSwitchDebug.logProjections {
            print("[TPLinear] AllToShardedSwitch \(debugName ?? "<unknown>") x=\(x.shape) weight=\(weight.shape) indices=\(indices.shape)")
        }
        let result = super.callAsFunction(x, indices, sortedIndices: sortedIndices)
        TPSwitchDebug.eval(result, name: debugName ?? "AllToShardedSwitch")
        return result
    }
}

open class AllToShardedQuantizedSwitchLinear: QuantizedSwitchLinear {
    public let group: Group
    public var debugName: String?

    public static func from(
        _ linear: QuantizedSwitchLinear,
        group: Group? = nil,
        segments: Int = 1
    ) -> AllToShardedQuantizedSwitchLinear {
        let g = group ?? Group(strict: false)
        let shape = linear.weight.shape
        precondition(shape.count == 3, "QuantizedSwitchLinear weight must be [experts, out, packed_in]")
        let experts = shape[0]
        let out = shape[1]
        precondition(out % segments == 0, "output dims must be divisible by segments")
        let perSegmentOut = out / segments
        precondition(perSegmentOut % g.size == 0, "(output dims / segments) must divide group size")
        let perRankOut = perSegmentOut / g.size

        var weightRows: [MLXArray] = []
        var scaleRows: [MLXArray] = []
        var quantBiasRows: [MLXArray] = []
        var biasRows: [MLXArray] = []
        for segment in 0 ..< segments {
            let segmentStart = segment * perSegmentOut
            let rankStart = segmentStart + g.rank * perRankOut
            let rankEnd = rankStart + perRankOut
            weightRows.append(linear.weight[0..., rankStart ..< rankEnd, 0...])
            scaleRows.append(linear.scales[0..., rankStart ..< rankEnd, 0...])
            if let biases = linear.biases {
                quantBiasRows.append(biases[0..., rankStart ..< rankEnd, 0...])
            }
            if let bias = linear.bias {
                biasRows.append(bias[0..., rankStart ..< rankEnd])
            }
        }

        let weight = concatenated(weightRows, axis: 1)
        return AllToShardedQuantizedSwitchLinear(
            inputDims: linear.inputDims,
            outputDims: weight.dim(1),
            numExperts: experts,
            weight: weight,
            bias: linear.bias == nil ? nil : concatenated(biasRows, axis: 1),
            scales: concatenated(scaleRows, axis: 1),
            biases: linear.biases == nil ? nil : concatenated(quantBiasRows, axis: 1),
            groupSize: linear.groupSize,
            bits: linear.bits,
            mode: linear.mode,
            group: g)
    }

    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        bias: MLXArray?,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        group: Group
    ) {
        self.group = group
        super.init(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            weight: weight,
            bias: bias,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode)
    }

    public override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        if ProcessInfo.processInfo.environment["TP_LINEAR_DEBUG"] == "1" || TPSwitchDebug.logProjections {
            print("[TPLinear] AllToShardedQuantizedSwitch \(debugName ?? "<unknown>") x=\(x.shape) weight=\(weight.shape) scales=\(scales.shape) indices=\(indices.shape)")
        }
        let result = super.callAsFunction(x, indices, sortedIndices: sortedIndices)
        TPSwitchDebug.eval(result, name: debugName ?? "AllToShardedQuantizedSwitch")
        return result
    }
}

/// Tensor-parallel routed-expert projection that consumes a sharded feature
/// axis and all-reduces partial expert outputs back to full hidden size.
open class ShardedToAllSwitchLinear: SwitchLinear {
    public let group: Group
    public var debugName: String?

    public static func from(
        _ linear: SwitchLinear,
        group: Group? = nil,
        segments: Int = 1
    ) -> ShardedToAllSwitchLinear {
        let g = group ?? Group(strict: false)
        let shape = linear.weight.shape
        precondition(shape.count == 3, "SwitchLinear weight must be [experts, out, in]")
        let experts = shape[0]
        let out = shape[1]
        let input = shape[2]
        precondition(input % segments == 0, "input dims must be divisible by segments")
        let perSegmentIn = input / segments
        precondition(
            perSegmentIn % g.size == 0,
            "(input dims / segments) must be divisible by group size")

        let perRankIn = perSegmentIn / g.size
        var chunks: [MLXArray] = []
        for segment in 0 ..< segments {
            let segmentStart = segment * perSegmentIn
            let rankStart = segmentStart + g.rank * perRankIn
            let rankEnd = rankStart + perRankIn
            chunks.append(linear.weight[0..., 0..., rankStart ..< rankEnd])
        }
        let shardedWeight = concatenated(chunks, axis: 2)

        return ShardedToAllSwitchLinear(
            inputDims: shardedWeight.dim(2),
            outputDims: out,
            numExperts: experts,
            weight: shardedWeight,
            bias: linear.bias,
            group: g)
    }

    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        bias: MLXArray?,
        group: Group
    ) {
        self.group = group
        super.init(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            weight: weight,
            bias: bias)
    }

    public override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        if ProcessInfo.processInfo.environment["TP_LINEAR_DEBUG"] == "1" || TPSwitchDebug.logProjections {
            print("[TPLinear] ShardedToAllSwitch \(debugName ?? "<unknown>") x=\(x.shape) weight=\(weight.shape) indices=\(indices.shape)")
        }
        let weightT = self.weight.swappedAxes(-1, -2)
        let partial = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)
        var result = Collectives.allSum(partial, group: group)
        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }
        TPSwitchDebug.eval(result, name: debugName ?? "ShardedToAllSwitch")
        return result
    }
}

open class ShardedToAllQuantizedSwitchLinear: QuantizedSwitchLinear {
    public let group: Group
    public var debugName: String?

    public static func from(
        _ linear: QuantizedSwitchLinear,
        group: Group? = nil,
        segments: Int = 1
    ) -> ShardedToAllQuantizedSwitchLinear {
        let g = group ?? Group(strict: false)
        let shape = linear.weight.shape
        precondition(shape.count == 3, "QuantizedSwitchLinear weight must be [experts, out, packed_in]")
        let experts = shape[0]
        let out = shape[1]
        let input = linear.inputDims
        precondition(input % segments == 0, "input dims must be divisible by segments")
        let valuesPerWord = 32 / linear.bits
        let perSegmentIn = input / segments
        precondition(perSegmentIn % g.size == 0, "(input dims / segments) must divide group size")
        let perRankIn = perSegmentIn / g.size

        var weightCols: [MLXArray] = []
        var scaleCols: [MLXArray] = []
        var quantBiasCols: [MLXArray] = []
        for segment in 0 ..< segments {
            let segmentStart = segment * perSegmentIn
            let rankStart = segmentStart + g.rank * perRankIn
            let rankEnd = rankStart + perRankIn
            precondition(
                rankStart % valuesPerWord == 0 && rankEnd % valuesPerWord == 0,
                "quantized input shard must align to packed word")
            precondition(
                rankStart % linear.groupSize == 0 && rankEnd % linear.groupSize == 0,
                "quantized input shard must align to quant group")
            let packedStart = rankStart / valuesPerWord
            let packedEnd = rankEnd / valuesPerWord
            let scaleStart = rankStart / linear.groupSize
            let scaleEnd = rankEnd / linear.groupSize
            weightCols.append(linear.weight[0..., 0..., packedStart ..< packedEnd])
            scaleCols.append(linear.scales[0..., 0..., scaleStart ..< scaleEnd])
            if let biases = linear.biases {
                quantBiasCols.append(biases[0..., 0..., scaleStart ..< scaleEnd])
            }
        }

        let weight = concatenated(weightCols, axis: 2)
        return ShardedToAllQuantizedSwitchLinear(
            inputDims: perRankIn * segments,
            outputDims: out,
            numExperts: experts,
            weight: weight,
            bias: linear.bias,
            scales: concatenated(scaleCols, axis: 2),
            biases: linear.biases == nil ? nil : concatenated(quantBiasCols, axis: 2),
            groupSize: linear.groupSize,
            bits: linear.bits,
            mode: linear.mode,
            group: g)
    }

    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        bias: MLXArray?,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        group: Group
    ) {
        self.group = group
        super.init(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            weight: weight,
            bias: bias,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode)
    }

    public override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        if ProcessInfo.processInfo.environment["TP_LINEAR_DEBUG"] == "1" || TPSwitchDebug.logProjections {
            print("[TPLinear] ShardedToAllQuantizedSwitch \(debugName ?? "<unknown>") x=\(x.shape) weight=\(weight.shape) scales=\(scales.shape) indices=\(indices.shape)")
        }
        var result = MLX.gatherQuantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            sortedIndices: sortedIndices
        )
        result = Collectives.allSum(result, group: group)
        if let bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }
        TPSwitchDebug.eval(result, name: debugName ?? "ShardedToAllQuantizedSwitch")
        return result
    }
}
