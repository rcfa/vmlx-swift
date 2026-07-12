// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// How much of the host's remaining memory a single prefix-cache store may take.
///
/// This is the engine-side projection of the user's memory-safety level. The
/// resolver in `ServerRuntimeSettings` turns that level into load-time caps
/// (`LoadConfiguration.memoryLimit` / `maxResidentBytes`); this carries the same
/// level down to the one runtime decision those caps cannot reach, because
/// `CacheStoreBudget` runs deep inside the decode loop with no settings handle.
/// The host sets `CacheStoreBudget.policy` before each load, exactly as it
/// already sets `TiedHeadQuantizationPolicy.current`.
///
/// A fraction of *free headroom*, deliberately — not a ceiling on total memory.
/// The load caps are a graph-scheduling guideline, not a hard resident limit, so
/// a large pack legitimately runs with `activeMemory` above them (Hy3 sits at
/// ~96 GB on a 128 GB Mac under the default 0.70 load cap). Treating a load cap
/// as a store ceiling would put `activeBytes` over budget before a single KV byte
/// and refuse *every* store — silently killing the prefix cache for precisely the
/// large models whose re-prefill hurts most. Scaling the *increment* instead lets
/// the safety level bite without that collapse.
public struct CacheStorePolicy: Sendable, Equatable {

    /// Share of current free headroom (`physical - active`) that one store may use.
    public var headroomFraction: Double

    public init(headroomFraction: Double) {
        self.headroomFraction = min(max(headroomFraction, 0), 1)
    }

    public static let performance = CacheStorePolicy(headroomFraction: 0.45)
    public static let balanced = CacheStorePolicy(headroomFraction: 0.35)
    public static let safeAuto = CacheStorePolicy(headroomFraction: 0.25)
    public static let strict = CacheStorePolicy(headroomFraction: 0.15)
    public static let diagnosticDangerous = CacheStorePolicy(headroomFraction: 0.55)
}

/// Whether a KV cache can be *saved* without pushing the host over a cliff.
///
/// The prefix cache is an optimisation: a stored entry only ever makes a later
/// request faster. Storing one must therefore never be able to take the machine
/// down — but until this guard existed, it could.
///
/// Reported live on an M5 Max / 128 GB running Llama-3.3-70B-8bit at a 64K
/// context: prefill tracked expectation (75 → 102 GiB), then +23 GiB landed
/// immediately after generation, inside the cache-store window — ~23 GiB being
/// exactly the size of the 62K-token KV cache. Free memory collapsed, page
/// reclaim stalled, and the kernel panicked. macOS will not jetsam a plain user
/// process out of the way, so nothing intervenes: the cap has to live here.
///
/// So: measure first, and if the store would not fit, skip it. A skipped store
/// costs one slower request. An unchecked store costs the machine.
public enum CacheStoreBudget {

    private static let lock = NSLock()
    nonisolated(unsafe) private static var _policy: CacheStorePolicy = .safeAuto

    /// The user's memory-safety level, projected onto this decision. Set by the
    /// host before each model load; defaults to the same Safe Auto the settings
    /// resolver defaults to, so an embedder that never sets it is unaffected.
    public static var policy: CacheStorePolicy {
        get { lock.withLock { _policy } }
        set { lock.withLock { _policy = newValue } }
    }

    /// Peak *additional* full-KV materialisation the store performs, in units of
    /// the live cache.
    ///
    /// One, not three. An earlier revision of this guard claimed three copies —
    /// a deep copy, a host `Data` extract, and the disk-store cache — and none of
    /// the three survive contact with the code: `KVCacheSimple.copy()` takes
    /// full-range slices that share the source buffer, `extractLayerData` returns
    /// references, `makeDiskStoreCache` returns the snapshot unchanged on the raw
    /// path, and `mlx_save_safetensors` streams to the file descriptor rather than
    /// building a host `Data`. What is left is the `contiguous` pass the writer
    /// makes before writing, which can allocate at most one compact copy of the
    /// logical KV. The reporter's own footprint series agrees: the excursion was
    /// +23 GiB against a 23 GiB cache — one copy, not three.
    ///
    /// Overcounting is not free: at 3x this refused stores that fit, quietly
    /// costing the cache hits it was never meant to cost.
    static let materializationFactor = 1

    /// Live bytes held by a KV cache, without copying its contents.
    ///
    /// `state` is the same array set the store would serialise, and `nbytes` is
    /// shape x item size, so nothing is evaluated and no tensor data is copied.
    /// Some implementations do build lazy views to answer it (slices in
    /// `KVCacheSimple` / `RotatingKVCache` / `QuantizedKVCache`, a flatten in
    /// `CacheList`), so this is cheap rather than literally free — orders of
    /// magnitude below the copy + serialize it is deciding whether to allow.
    public static func cacheBytes(_ cache: [KVCache]) -> Int {
        cache.reduce(0) { total, layer in
            total + layer.state.reduce(0) { $0 + $1.nbytes }
        }
    }

    /// Whether storing `cache` fits in what the host can still give us.
    ///
    /// Budgeted against **physical memory**, not the GPU working set. The working
    /// set is the wrong ceiling for this decision in both directions: exceeding it
    /// only means macOS pages the excess — slow, survivable — whereas exhausting
    /// physical memory is what actually kills the host, and that is the only thing
    /// this guard exists to prevent.
    public static func canStore(_ cache: [KVCache]) -> Bool {
        let liveBytes = cacheBytes(cache)
        guard liveBytes > 0 else { return true }
        return canStore(cacheBytes: liveBytes)
    }

    /// Testable core: no MLX state, just the arithmetic.
    ///
    /// `active + storeCost + margin <= budget`, where the margin is whatever share
    /// of the headroom the user's safety level says to leave alone. Equivalently,
    /// and more directly: the store may consume `headroomFraction` of the memory
    /// the host still has free.
    static func canStore(
        cacheBytes liveBytes: Int,
        activeBytes: Int = max(0, MLX.Memory.activeMemory),
        budgetBytes: Int? = Int(exactly: ProcessInfo.processInfo.physicalMemory),
        policy: CacheStorePolicy = CacheStoreBudget.policy
    ) -> Bool {
        guard liveBytes > 0 else { return true }
        guard let budgetBytes, budgetBytes > 0 else {
            // No budget to reason about: don't guess, don't block.
            return true
        }
        // Already over the wall: nothing to hand out.
        guard activeBytes < budgetBytes else { return false }

        let headroom = budgetBytes - activeBytes
        let allowance = Int(Double(headroom) * policy.headroomFraction)
        // Overflow-safe: liveBytes and the factor are both bounded well below
        // Int.max/2 in practice, but compare in the division direction anyway.
        guard materializationFactor > 0 else { return true }
        return liveBytes <= allowance / materializationFactor
    }
}
