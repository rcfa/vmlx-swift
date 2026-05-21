import Foundation

public enum MLXPressActivityCompressionVerdict: String, Sendable, Equatable {
    case passed
    case failed
    case unavailable
}

public struct MLXPressActivityCompressionCheck: Sendable, Equatable {
    public let modelBytes: UInt64
    public let preLoadFootprintBytes: UInt64?
    public let postLoadFootprintBytes: UInt64?
    public let footprintIncreaseBytes: UInt64?
    public let maxFootprintPercent: Double
    public let maxAllowedFootprintIncreaseBytes: UInt64
    public let verdict: MLXPressActivityCompressionVerdict

    public var passed: Bool {
        verdict == .passed
    }

    public var footprintIncreasePercent: Double? {
        guard let footprintIncreaseBytes, modelBytes > 0 else { return nil }
        return Double(footprintIncreaseBytes) / Double(modelBytes) * 100.0
    }

    public init(
        bundleFacts: MLXPressBundleFacts,
        preLoad: MLXPressMemorySnapshot,
        postLoad: MLXPressMemorySnapshot,
        maxFootprintPercent: Double
    ) {
        let modelBytes = bundleFacts.totalSafetensorsBytes
        let boundedPercent = maxFootprintPercent.isFinite
            ? max(0, maxFootprintPercent)
            : 0
        let maxAllowed = UInt64(
            (Double(modelBytes) * boundedPercent / 100.0).rounded(.up))

        let increase: UInt64?
        if let pre = preLoad.physicalFootprintBytes,
            let post = postLoad.physicalFootprintBytes
        {
            increase = post > pre ? post - pre : 0
        } else {
            increase = nil
        }

        let verdict: MLXPressActivityCompressionVerdict
        if modelBytes == 0 {
            verdict = .unavailable
        } else if let increase {
            verdict = increase <= maxAllowed ? .passed : .failed
        } else {
            verdict = .unavailable
        }

        self.modelBytes = modelBytes
        self.preLoadFootprintBytes = preLoad.physicalFootprintBytes
        self.postLoadFootprintBytes = postLoad.physicalFootprintBytes
        self.footprintIncreaseBytes = increase
        self.maxFootprintPercent = boundedPercent
        self.maxAllowedFootprintIncreaseBytes = maxAllowed
        self.verdict = verdict
    }
}
