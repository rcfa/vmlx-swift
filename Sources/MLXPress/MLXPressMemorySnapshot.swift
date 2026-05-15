import Foundation
import MLX

#if canImport(Darwin)
import Darwin
#endif

public struct MLXPressMemorySnapshot: Sendable, Equatable {
    public let residentSizeBytes: UInt64
    public let physicalFootprintBytes: UInt64?
    public let physicalMemoryBytes: UInt64
    public let mlxActiveMemoryBytes: UInt64
    public let mlxCacheMemoryBytes: UInt64
    public let mlxPeakMemoryBytes: UInt64

    public init(
        residentSizeBytes: UInt64,
        physicalFootprintBytes: UInt64?,
        physicalMemoryBytes: UInt64,
        mlxActiveMemoryBytes: UInt64 = 0,
        mlxCacheMemoryBytes: UInt64 = 0,
        mlxPeakMemoryBytes: UInt64 = 0
    ) {
        self.residentSizeBytes = residentSizeBytes
        self.physicalFootprintBytes = physicalFootprintBytes
        self.physicalMemoryBytes = physicalMemoryBytes
        self.mlxActiveMemoryBytes = mlxActiveMemoryBytes
        self.mlxCacheMemoryBytes = mlxCacheMemoryBytes
        self.mlxPeakMemoryBytes = mlxPeakMemoryBytes
    }

    public static func current() -> MLXPressMemorySnapshot {
        let mlxMemory = MLX.Memory.snapshot()
        return MLXPressMemorySnapshot(
            residentSizeBytes: currentResidentSizeBytes(),
            physicalFootprintBytes: currentPhysicalFootprintBytes(),
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            mlxActiveMemoryBytes: UInt64(max(0, mlxMemory.activeMemory)),
            mlxCacheMemoryBytes: UInt64(max(0, mlxMemory.cacheMemory)),
            mlxPeakMemoryBytes: UInt64(max(0, mlxMemory.peakMemory)))
    }
}

public func MLXPressFormatBytes(_ bytes: UInt64) -> String {
    ByteCountFormatter.string(
        fromByteCount: Int64(clamping: bytes),
        countStyle: .memory)
}

#if canImport(Darwin)
private func currentResidentSizeBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(
                mach_task_self_,
                task_flavor_t(MACH_TASK_BASIC_INFO),
                $0,
                &count)
        }
    }
    guard result == KERN_SUCCESS else { return 0 }
    return UInt64(info.resident_size)
}

private func currentPhysicalFootprintBytes() -> UInt64? {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(
                mach_task_self_,
                task_flavor_t(TASK_VM_INFO),
                $0,
                &count)
        }
    }
    guard result == KERN_SUCCESS else { return nil }
    return UInt64(info.phys_footprint)
}
#else
private func currentResidentSizeBytes() -> UInt64 {
    0
}

private func currentPhysicalFootprintBytes() -> UInt64? {
    nil
}
#endif
