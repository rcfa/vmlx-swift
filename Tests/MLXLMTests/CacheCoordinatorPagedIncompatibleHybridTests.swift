// Regression tests for the distinction between typed disk persistence and
// paged-RAM compatibility. Hybrid SSM/GLA models require an exact-boundary
// recurrent companion, but Mamba/Arrays + ordinary/TurboQuant attention can
// still use paged KV blocks when that companion exists. Rotating, CCA, affine,
// and hybrid-pool layouts remain disk-only.

import Foundation
@testable import MLXLMCommon
import Testing

@Test func mambaCacheTriggersPathDependent() {
    let cache: [any KVCache] = [MambaCache(), KVCacheSimple()]
    #expect(cacheContainsPathDependentState(cache))
    #expect(cacheRequiresDiskBackedCoordinatorRestore(cache))
    #expect(!cacheCannotUsePagedCoordinatorRestore(cache))
}

@Test func arraysCacheTriggersPathDependent() {
    let cache: [any KVCache] = [ArraysCache(size: 2), KVCacheSimple()]
    #expect(cacheContainsPathDependentState(cache))
    #expect(cacheRequiresDiskBackedCoordinatorRestore(cache))
    #expect(!cacheCannotUsePagedCoordinatorRestore(cache))
}

@Test func zayaCCACacheTriggersPathDependent() {
    let cache: [any KVCache] = [ZayaCCACache(), KVCacheSimple()]
    #expect(cacheContainsPathDependentState(cache))
    #expect(cacheRequiresDiskBackedCoordinatorRestore(cache))
    #expect(cacheCannotUsePagedCoordinatorRestore(cache))
}

@Test func cacheListWrappingMambaTriggersPathDependent() {
    let composite = CacheList(MambaCache(), KVCacheSimple())
    let cache: [any KVCache] = [composite]
    #expect(cacheContainsPathDependentState(cache))
    #expect(!cacheCannotUsePagedCoordinatorRestore(cache))
}

@Test func cacheListWrappingArraysTriggersPathDependent() {
    let composite = CacheList(ArraysCache(size: 2), KVCacheSimple())
    let cache: [any KVCache] = [composite]
    #expect(cacheContainsPathDependentState(cache))
    #expect(!cacheCannotUsePagedCoordinatorRestore(cache))
}

@Test func plainKVDoesNotTriggerPathDependent() {
    let cache: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
    #expect(!cacheContainsPathDependentState(cache))
    #expect(!cacheRequiresDiskBackedCoordinatorRestore(cache))
    #expect(!cacheCannotUsePagedCoordinatorRestore(cache))
}

@Test func emptyCacheDoesNotTriggerPathDependent() {
    let cache: [any KVCache] = []
    #expect(!cacheContainsPathDependentState(cache))
    #expect(!cacheRequiresDiskBackedCoordinatorRestore(cache))
    #expect(!cacheCannotUsePagedCoordinatorRestore(cache))
}

@Test func unknownCacheTypeFailsClosedForPagedRestore() {
    let cache: [any KVCache] = [BaseKVCache()]
    #expect(cacheCannotUsePagedCoordinatorRestore(cache))
}

@Test func rotatingCacheRequiresDiskBackedCoordinatorRestore() {
    let cache: [any KVCache] = [RotatingKVCache(maxSize: 32), KVCacheSimple()]
    #expect(!cacheContainsPathDependentState(cache))
    #expect(cacheRequiresDiskBackedCoordinatorRestore(cache))
    #expect(cacheCannotUsePagedCoordinatorRestore(cache))
}

@Test func mixedHybridLayerOrderingTriggersPathDependent() {
    // Real hybrid models (Nemotron-H, Bailing) interleave KV layers with
    // recurrent state layers. Path-dependent detection must fire regardless
    // of which position the recurrent layer occupies.
    let cacheStart: [any KVCache] = [MambaCache(), KVCacheSimple(), KVCacheSimple()]
    let cacheMiddle: [any KVCache] = [KVCacheSimple(), MambaCache(), KVCacheSimple()]
    let cacheEnd: [any KVCache] = [KVCacheSimple(), KVCacheSimple(), MambaCache()]
    #expect(cacheContainsPathDependentState(cacheStart))
    #expect(cacheContainsPathDependentState(cacheMiddle))
    #expect(cacheContainsPathDependentState(cacheEnd))
    #expect(!cacheCannotUsePagedCoordinatorRestore(cacheStart))
    #expect(!cacheCannotUsePagedCoordinatorRestore(cacheMiddle))
    #expect(!cacheCannotUsePagedCoordinatorRestore(cacheEnd))
}

@Test func turboQuantAttentionWithMambaCompanionCanUsePagedRestore() {
    let cache: [any KVCache] = [
        MambaCache(),
        TurboQuantKVCache(keyBits: 4, valueBits: 4),
    ]
    #expect(cacheRequiresDiskBackedCoordinatorRestore(cache))
    #expect(!cacheCannotUsePagedCoordinatorRestore(cache))
}

@Test func rotatingAttentionInsideHybridRemainsDiskOnly() {
    let cache: [any KVCache] = [MambaCache(), RotatingKVCache(maxSize: 32)]
    #expect(cacheRequiresDiskBackedCoordinatorRestore(cache))
    #expect(cacheCannotUsePagedCoordinatorRestore(cache))
}
