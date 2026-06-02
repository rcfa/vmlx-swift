import Foundation
import MLX
import MLXNN
import MLXRandom

/// Tensor-parallel linear layer: each rank holds a row-shard of the
/// weight, applies the local matmul, and the result is sharded across
/// the group along the output axis. The aggregate equivalent of the
/// original (output_dims) is the concatenation of every rank's local
/// output.
///
/// Mirror of Python's `mlx.nn.layers.distributed.AllToShardedLinear`.
/// In the inference path (no gradients), forward is just a standard
/// linear with the locally-shaped weight. The Python reference wraps
/// the input in `sum_gradients(group)` for backward-pass aggregation
/// — that's a no-op on the forward path so we omit it.
///
/// Subclasses `Linear` so existing model code that declares
/// `@ModuleInfo var x: Linear` accepts the TP variant via
/// `Module.update(modules:)` (the runtime cast required by `@ModuleInfo`
/// only succeeds when the new value is type-compatible with the
/// declared property type — see `ShardingPlan.apply`).
open class AllToShardedLinear: Linear {

    public let group: Group
    public var debugName: String?

    /// Initialise from scratch with random weights, sharded across `group`.
    public init(
        _ inputDimensions: Int,
        _ outputDimensions: Int,
        bias: Bool = true,
        group: Group? = nil
    ) {
        let g = group ?? Group(strict: false)
        precondition(
            outputDimensions % g.size == 0,
            "AllToShardedLinear: output_dims (\(outputDimensions)) must be divisible by group size (\(g.size))")

        let scale = sqrt(1.0 / Float(inputDimensions))
        let perRankOut = outputDimensions / g.size

        let w = MLXRandom.uniform(-scale ..< scale, [perRankOut, inputDimensions])
        let b: MLXArray?
        if bias {
            b = MLXRandom.uniform(-scale ..< scale, [perRankOut])
        } else {
            b = nil
        }
        self.group = g
        super.init(weight: w, bias: b)
    }

    /// Initialise from an existing dense Linear layer by sharding its
    /// weights along the output axis. Equivalent to Python's
    /// `from_linear` classmethod. `segments` allows splitting fused
    /// weights (e.g. fused QKV with 3 segments) so each segment is
    /// sharded independently rather than splitting through a logical
    /// boundary.
    public static func from(
        _ linear: Linear,
        group: Group? = nil,
        segments: Int = 1
    ) -> AllToShardedLinear {
        let g = group ?? Group(strict: false)
        let (out, _) = linear.shape
        precondition(
            out % g.size == 0,
            "AllToShardedLinear.from: output_dims (\(out)) must be divisible by group size (\(g.size))")
        precondition(segments >= 1, "segments must be >= 1")
        precondition(out % segments == 0,
                     "output_dims (\(out)) must be divisible by segments (\(segments))")

        // Per-rank slice plan: split the weight into `segments` chunks
        // along axis 0, take rank-r's slice from each chunk, concat.
        let perSegmentOut = out / segments
        let perSegmentPerRank = perSegmentOut / g.size
        precondition(perSegmentOut % g.size == 0,
                     "(output_dims/segments) must be divisible by group size")

        var rows: [MLXArray] = []
        for seg in 0 ..< segments {
            let segStart = seg * perSegmentOut
            let rankStart = segStart + g.rank * perSegmentPerRank
            let rankEnd = rankStart + perSegmentPerRank
            rows.append(linear.weight[rankStart ..< rankEnd, 0...])
        }
        let shardedWeight = concatenated(rows, axis: 0)

        var shardedBias: MLXArray? = nil
        if let b = linear.bias {
            var biasRows: [MLXArray] = []
            for seg in 0 ..< segments {
                let segStart = seg * perSegmentOut
                let rankStart = segStart + g.rank * perSegmentPerRank
                let rankEnd = rankStart + perSegmentPerRank
                biasRows.append(b[rankStart ..< rankEnd])
            }
            shardedBias = concatenated(biasRows, axis: 0)
        }

        return AllToShardedLinear(
            preShardedWeight: shardedWeight,
            preShardedBias: shardedBias,
            group: g)
    }

    /// Subclass-friendly init that takes already-sharded weights.
    /// Used by `from(_:group:segments:)`.
    public init(preShardedWeight: MLXArray, preShardedBias: MLXArray?, group: Group) {
        self.group = group
        super.init(weight: preShardedWeight, bias: preShardedBias)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if ProcessInfo.processInfo.environment["TP_LINEAR_DEBUG"] == "1" {
            print("[TPLinear] AllToSharded \(debugName ?? "<unknown>") x=\(x.shape) weight=\(weight.shape)")
        }
        return super.callAsFunction(x)
    }

    // Forward inherits from `Linear`: y = x @ weight.T (+ bias). No
    // collective fires here — gather is implicit at the next
    // `ShardedToAllLinear` whose all-reduce reassembles the full hidden
    // state.
}

/// Tensor-parallel output-sharded affine-quantized linear.
open class AllToShardedQuantizedLinear: QuantizedLinear {

    public let group: Group
    public var debugName: String?

    public static func from(
        _ linear: QuantizedLinear,
        group: Group? = nil,
        segments: Int = 1
    ) -> AllToShardedQuantizedLinear {
        let g = group ?? Group(strict: false)
        let (out, _) = linear.shape
        precondition(out % g.size == 0, "output dims must be divisible by group size")
        precondition(segments >= 1, "segments must be >= 1")
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
            weightRows.append(linear.weight[rankStart ..< rankEnd, 0...])
            scaleRows.append(linear.scales[rankStart ..< rankEnd, 0...])
            if let biases = linear.biases {
                quantBiasRows.append(biases[rankStart ..< rankEnd, 0...])
            }
            if let bias = linear.bias {
                biasRows.append(bias[rankStart ..< rankEnd])
            }
        }

        return AllToShardedQuantizedLinear(
            weight: concatenated(weightRows, axis: 0),
            bias: linear.bias == nil ? nil : concatenated(biasRows, axis: 0),
            scales: concatenated(scaleRows, axis: 0),
            biases: linear.biases == nil ? nil : concatenated(quantBiasRows, axis: 0),
            groupSize: linear.groupSize,
            bits: linear.bits,
            mode: linear.mode,
            group: g)
    }

    public init(
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
            weight: weight,
            bias: bias,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if ProcessInfo.processInfo.environment["TP_LINEAR_DEBUG"] == "1" {
            print("[TPLinear] AllToShardedQuantized \(debugName ?? "<unknown>") x=\(x.shape) weight=\(weight.shape) scales=\(scales.shape)")
        }
        return super.callAsFunction(x)
    }
}

/// Tensor-parallel linear layer: each rank holds a column-shard of
/// the weight, applies the local matmul on its slice of the input,
/// and the **partial** outputs are summed via all-reduce across the
/// group to produce the full output.
///
/// Mirror of Python's `mlx.nn.layers.distributed.ShardedToAllLinear`.
/// Subclasses `Linear` for the same reason as `AllToShardedLinear` —
/// see that type's docstring.
open class ShardedToAllLinear: Linear {

    public let group: Group
    public var debugName: String?

    public init(
        _ inputDimensions: Int,
        _ outputDimensions: Int,
        bias: Bool = true,
        group: Group? = nil
    ) {
        let g = group ?? Group(strict: false)
        precondition(
            inputDimensions % g.size == 0,
            "ShardedToAllLinear: input_dims (\(inputDimensions)) must be divisible by group size (\(g.size))")

        let scale = sqrt(1.0 / Float(inputDimensions))
        let perRankIn = inputDimensions / g.size

        let w = MLXRandom.uniform(-scale ..< scale, [outputDimensions, perRankIn])
        let b: MLXArray?
        if bias {
            // Bias is full-width and added once after the all-reduce.
            b = MLXRandom.uniform(-scale ..< scale, [outputDimensions])
        } else {
            b = nil
        }
        self.group = g
        super.init(weight: w, bias: b)
    }

    /// Build from a dense Linear by column-sharding its weight.
    /// `segments` mirrors the AllToSharded variant — useful for fused
    /// projections that need to be split before column-shard.
    public static func from(
        _ linear: Linear,
        group: Group? = nil,
        segments: Int = 1
    ) -> ShardedToAllLinear {
        let g = group ?? Group(strict: false)
        let (_, inDim) = linear.shape
        precondition(inDim % g.size == 0,
                     "ShardedToAllLinear.from: input_dims (\(inDim)) must be divisible by group size (\(g.size))")
        precondition(segments >= 1, "segments must be >= 1")

        let perSegmentIn = inDim / segments
        let perSegmentPerRank = perSegmentIn / g.size
        precondition(perSegmentIn % g.size == 0,
                     "(input_dims/segments) must be divisible by group size")

        var cols: [MLXArray] = []
        for seg in 0 ..< segments {
            let segStart = seg * perSegmentIn
            let rankStart = segStart + g.rank * perSegmentPerRank
            let rankEnd = rankStart + perSegmentPerRank
            cols.append(linear.weight[0..., rankStart ..< rankEnd])
        }
        let shardedWeight = concatenated(cols, axis: 1)

        return ShardedToAllLinear(
            preShardedWeight: shardedWeight,
            bias: linear.bias,
            group: g)
    }

    /// Subclass-friendly init for callers that already sharded weights.
    public init(preShardedWeight: MLXArray, bias: MLXArray?, group: Group) {
        self.group = group
        super.init(weight: preShardedWeight, bias: bias)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if ProcessInfo.processInfo.environment["TP_LINEAR_DEBUG"] == "1" {
            print("[TPLinear] ShardedToAll \(debugName ?? "<unknown>") x=\(x.shape) weight=\(weight.shape)")
        }
        let partial = matmul(x, weight.T)
        let summed = Collectives.allSum(partial, group: group)
        if let bias {
            return summed + bias
        }
        return summed
    }
}

/// Tensor-parallel input-sharded affine-quantized linear.
open class ShardedToAllQuantizedLinear: QuantizedLinear {

    public let group: Group
    public var debugName: String?

    public static func from(
        _ linear: QuantizedLinear,
        group: Group? = nil,
        segments: Int = 1
    ) -> ShardedToAllQuantizedLinear {
        let g = group ?? Group(strict: false)
        let (_, input) = linear.shape
        precondition(input % g.size == 0, "input dims must be divisible by group size")
        precondition(segments >= 1, "segments must be >= 1")
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
            weightCols.append(linear.weight[0..., packedStart ..< packedEnd])
            scaleCols.append(linear.scales[0..., scaleStart ..< scaleEnd])
            if let biases = linear.biases {
                quantBiasCols.append(biases[0..., scaleStart ..< scaleEnd])
            }
        }

        return ShardedToAllQuantizedLinear(
            weight: concatenated(weightCols, axis: 1),
            bias: linear.bias,
            scales: concatenated(scaleCols, axis: 1),
            biases: linear.biases == nil ? nil : concatenated(quantBiasCols, axis: 1),
            groupSize: linear.groupSize,
            bits: linear.bits,
            mode: linear.mode,
            group: g)
    }

    public init(
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
            weight: weight,
            bias: bias,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if ProcessInfo.processInfo.environment["TP_LINEAR_DEBUG"] == "1" {
            print("[TPLinear] ShardedToAllQuantized \(debugName ?? "<unknown>") x=\(x.shape) weight=\(weight.shape) scales=\(scales.shape)")
        }
        var partial = MLX.quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        partial = Collectives.allSum(partial, group: group)
        if let bias {
            partial = partial + bias
        }
        return partial
    }
}
