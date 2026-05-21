// Copyright (c) 2026 Jinho Jang. All rights reserved.

import Foundation
import Testing
@testable import MLXLMCommon

@Suite("JangPressMachCache")
struct JangPressMachCacheTests {
    private static func pattern(_ size: Int, seed: UInt8) -> [UInt8] {
        (0..<size).map { UInt8((Int(seed) + $0) & 0xff) }
    }

    private static func register(
        _ cache: JangPressMachCache,
        layer: Int,
        expert: Int,
        component: String = "",
        size: Int,
        seed: UInt8
    ) throws -> ([UInt8], JangPressTile) {
        let bytes = pattern(size, seed: seed)
        let tile = try bytes.withUnsafeBytes {
            try cache.register(layer: layer, expert: expert, component: component, bytes: $0)
        }
        return (bytes, tile)
    }

    @Test("register places bytes in stable purgeable region")
    func registerPlacesBytesInStableRegion() throws {
        let cache = JangPressMachCache()
        let (original, tile) = try Self.register(
            cache,
            layer: 0,
            expert: 0,
            size: 8192,
            seed: 0xab)

        let stored = UnsafeRawBufferPointer(
            start: tile.baseAddress,
            count: original.count)
        #expect(Array(stored) == original)

        let acquired1 = try cache.acquire(layer: 0, experts: [0])
        let acquired2 = try cache.acquire(layer: 0, experts: [0])
        #expect(acquired1[0].baseAddress == acquired2[0].baseAddress)
    }

    @Test("acquire and release toggle volatile accounting")
    func acquireReleaseTogglesVolatileAccounting() throws {
        let cache = JangPressMachCache()
        _ = try Self.register(cache, layer: 0, expert: 0, size: 4096, seed: 0x10)
        _ = try Self.register(cache, layer: 0, expert: 1, size: 4096, seed: 0x11)
        _ = try Self.register(cache, layer: 0, expert: 2, size: 4096, seed: 0x12)

        let initial = cache.snapshot()
        #expect(initial.totalTiles == 3)
        #expect(initial.currentlyNonVolatile == 3)

        _ = try cache.acquire(layer: 0, experts: [0, 2])
        cache.release(layer: 0, experts: [0, 2])

        let released = cache.snapshot()
        #expect(released.acquireCount == 2)
        #expect(released.releaseCount == 2)
        #expect(released.currentlyVolatile == 2)
        #expect(released.currentlyNonVolatile == 1)
    }

    @Test("pinHot prevents release")
    func pinHotPreventsRelease() throws {
        let cache = JangPressMachCache()
        _ = try Self.register(cache, layer: 5, expert: 7, size: 4096, seed: 0x42)

        cache.pinHot(layer: 5, experts: [7])
        cache.release(layer: 5, experts: [7])

        let stats = cache.snapshot()
        #expect(stats.hotPinned == 1)
        #expect(stats.releaseCount == 0)
        #expect(stats.currentlyVolatile == 0)
    }

    @Test("unknown expert throws")
    func unknownExpertThrows() throws {
        let cache = JangPressMachCache()
        _ = try Self.register(cache, layer: 0, expert: 0, size: 4096, seed: 0)

        do {
            _ = try cache.acquire(layer: 0, experts: [999])
            Issue.record("expected unknownExpert")
        } catch JangPressMachError.unknownExpert(let layer, let expert) {
            #expect(layer == 0)
            #expect(expert == 999)
        }
    }

    @Test("releaseColdTiles uses compression percentage")
    func releaseColdTilesUsesCompressionPercentage() throws {
        let cache = JangPressMachCache(config: .init(manualCompressPercent: 70))
        for expert in 0..<10 {
            _ = try Self.register(
                cache,
                layer: 0,
                expert: expert,
                size: 4096,
                seed: UInt8(expert))
        }
        _ = try cache.acquire(layer: 0, experts: [8, 9])

        let released = cache.releaseColdTiles()
        let stats = cache.snapshot()
        #expect(released == 7)
        #expect(stats.currentlyVolatile == 7)
        #expect(stats.currentlyNonVolatile == 3)
    }

    @Test("manual compression percent maps to hot fraction")
    func manualCompressionPercentMapsToHotFraction() {
        let config = JangPressMachConfig(manualCompressPercent: 70)
        #expect(config.alwaysHotFraction == 0.3)
        #expect(config.manualCompressPercent == 70)
    }

    @Test("component tiles share expert acquire release")
    func componentTilesShareExpertAcquireRelease() throws {
        let cache = JangPressMachCache()
        _ = try Self.register(
            cache,
            layer: 0,
            expert: 3,
            component: "gate_proj.tq_packed",
            size: 4096,
            seed: 0x31)
        _ = try Self.register(
            cache,
            layer: 0,
            expert: 3,
            component: "gate_proj.tq_norms",
            size: 4096,
            seed: 0x32)

        let acquired = try cache.acquire(layer: 0, experts: [3])
        #expect(acquired.count == 2)

        cache.release(layer: 0, experts: [3])
        let stats = cache.snapshot()
        #expect(stats.currentlyVolatile == 2)
        #expect(stats.releaseCount == 2)
    }

    @Test("component array view reads tile bytes")
    func componentArrayViewReadsTileBytes() throws {
        let cache = JangPressMachCache()
        let bytes = [UInt8](0..<16)
        _ = try bytes.withUnsafeBytes {
            try cache.register(
                layer: 0,
                expert: 1,
                component: "down_proj.tq_packed",
                bytes: $0)
        }

        let array = try cache.array(
            layer: 0,
            expert: 1,
            component: "down_proj.tq_packed",
            shape: [16],
            dtype: .uint8)
        #expect(array.asArray(UInt8.self) == bytes)
    }

    @Test("remove component tile updates stats")
    func removeComponentTileUpdatesStats() throws {
        let cache = JangPressMachCache()
        _ = try Self.register(
            cache,
            layer: 1,
            expert: 4,
            component: "gate_proj.tq_packed",
            size: 4096,
            seed: 0x41)
        _ = try Self.register(
            cache,
            layer: 1,
            expert: 4,
            component: "up_proj.tq_packed",
            size: 4096,
            seed: 0x42)

        #expect(cache.remove(layer: 1, expert: 4, component: "gate_proj.tq_packed"))
        let stats = cache.snapshot()
        #expect(stats.totalTiles == 1)
        #expect(stats.totalPayloadBytes == 4096)
        #expect(stats.currentlyNonVolatile == 1)

        do {
            _ = try cache.acquire(layer: 1, expert: 4, component: "gate_proj.tq_packed")
            Issue.record("expected removed component to throw")
        } catch JangPressMachError.unknownExpert(let layer, let expert) {
            #expect(layer == 1)
            #expect(expert == 4)
        }
    }
}
