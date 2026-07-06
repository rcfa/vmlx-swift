// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Serializes every CPU-side driver of the shared Metal command stream:
/// `eval`/`asyncEval`/`item` here, plus `Stream.synchronize()`,
/// `Memory.clearCache()`, `compile`, and stream lifecycle elsewhere.
///
/// This is REQUIRED for correctness, not just for the global-state ops.
/// MLX is not thread-safe: `mlx_eval`/`mlx_async_eval` run `gpu::eval`
/// *inline on the calling thread* (the per-stream `StreamThread` stays idle
/// for the default GPU stream), encoding into and committing the stream's
/// single `MTLCommandBuffer`. `Stream.synchronize()` ends+commits that same
/// buffer. The fork's C++ `stream_map_mtx_` only guards the stream-map lookup
/// (fix 9dabb6c4), NOT the encoder, so two Swift threads — e.g. a request's
/// prefill `async_eval` and a concurrent unload/`strictEvict`
/// `Stream.gpu.synchronize()` — otherwise mutate the same command buffer
/// concurrently. Live-reproduced SIGABRT: "Completed handler provided after
/// commit" (`addCompletedHandler` on a buffer synchronize just committed) and
/// AGX "command encoder is already encoding".
///
/// The lock is held ONLY across the brief CPU-side encode+commit
/// (`mlx_eval`/`mlx_async_eval` return once the buffer is committed; the GPU
/// executes asynchronously afterward with the lock already released), so the
/// CPU→GPU pipeline parallelism that motivated dropping the lock from this
/// path is preserved: in steady-state single-producer decode the lock is
/// uncontended, and it only serializes during teardown/cancel/cross-model —
/// exactly when a second thread would otherwise corrupt the encoder.
/// Recursive so a lock-holding path that re-enters eval does not self-deadlock.
let evalLock = NSRecursiveLock()

/// Evaluate one or more `MLXArray`
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func eval(_ arrays: MLXArray...) {
    let vector_array = newEvalVectorArray(arrays)
    evalLock.lock()
    mlx_eval(vector_array)
    evalLock.unlock()
    mlx_vector_array_free(vector_array)
}

/// Evaluate one or more `MLXArray`
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func eval(_ arrays: some Collection<MLXArray>) {
    let vector_array = newEvalVectorArray(arrays)
    evalLock.lock()
    mlx_eval(vector_array)
    evalLock.unlock()
    mlx_vector_array_free(vector_array)
}

/// Evaluate one or more `MLXArray` asynchronously.
///
/// ### See Also
/// - <doc:lazy-evaluation>
/// - ``asyncEval(_:)-(Collection<MLXArray>)``
public func asyncEval(_ arrays: some Collection<MLXArray>) {
    let vector_array = newEvalVectorArray(arrays)
    evalLock.lock()
    mlx_async_eval(vector_array)
    evalLock.unlock()
    mlx_vector_array_free(vector_array)
}

/// Evaluate one or more `MLXArray`.
///
/// This variant allows several structured types:
///
/// ```swift
/// let a: MLXArray
/// let b: [MLXArray]
/// let c: [String:MLXArray]
/// let d: [String:[MLXArray]]
/// let e: (MLXArray, MLXArray)
/// let f: [(String, MLXArray)]
/// let nested: [(MLXArray, [MLXArray])]
///
/// eval(a, b, c, d, e, f)
/// ```
///
/// Other structured types may be supported -- check the implementation.
///
/// ### See Also
/// - <doc:lazy-evaluation>
/// - ``asyncEval(_:)-(Collection<MLXArray>)``
public func eval(_ values: Any...) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    eval(arrays)
}

/// Evaluate one or more `MLXArray`.
///
/// See ``eval(_:)``
public func eval(_ values: some Sequence<Any>) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    eval(arrays)
}

/// Variant of ``eval(_:)-(Collection<MLXArray>)`` that checks for errors in MLX and throws.
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func checkedEval(_ values: Any...) throws {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    try withError {
        eval(arrays)
    }
}

/// Variant of ``eval(_:)-(MLXArray...)`` that checks for errors in MLX and throws.
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func checkedEval(_ values: some Sequence<Any>) throws {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    try withError {
        eval(arrays)
    }
}

/// Evaluate one or more `MLXArray` asynchronously.
///
/// This variant allows several structured types:
///
/// ```swift
/// let a: MLXArray
/// let b: [MLXArray]
/// let c: [String:MLXArray]
/// let d: [String:[MLXArray]]
/// let e: (MLXArray, MLXArray)
/// let f: [(String, MLXArray)]
/// let nested: [(MLXArray, [MLXArray])]
///
/// asyncEval(a, b, c, d, e, f)
/// ```
///
/// Other structured types may be supported -- check the implementation.
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func asyncEval(_ values: Any...) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    asyncEval(arrays)
}

/// Evaluate one or more `MLXArray` asynchronously.
///
/// See ``asyncEval(_:)-(Collection<MLXArray>)``
public func asyncEval(_ values: some Sequence<Any>) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    asyncEval(arrays)
}

private func collect(_ item: Any, into arrays: inout [MLXArray]) {
    switch item {
    case let v as Evaluatable:
        arrays.append(contentsOf: v.innerState())

    case let v as NestedDictionary<String, MLXArray>:
        arrays.append(contentsOf: v.flattened().map { $0.1 })

    case let v as MLXArray:
        arrays.append(v)
    case let v as [MLXArray]:
        arrays.append(contentsOf: v)
    case let v as [Any]:
        for item in v {
            collect(item, into: &arrays)
        }
    case let v as [AnyHashable: Any]:
        for item in v.values {
            collect(item, into: &arrays)
        }
    case let v as (Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
    case let v as (Any, Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
        collect(v.2, into: &arrays)
    case let v as (Any, Any, Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
        collect(v.2, into: &arrays)
        collect(v.3, into: &arrays)
    case let v as (Any, Any, Any, Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
        collect(v.2, into: &arrays)
        collect(v.3, into: &arrays)
        collect(v.4, into: &arrays)
    case is String, is any BinaryInteger, is any BinaryFloatingPoint:
        // ignore, e.g. (String, MLXArray)
        break
    default:
        fatalError("Unable to extract MLXArray from \(item)")
    }
}

private extension MLXArray {
    var hasBackingArray: Bool {
        ctx.ctx != nil
    }
}

@inline(__always)
private func newEvalVectorArray(_ arrays: some Collection<MLXArray>) -> mlx_vector_array {
    if arrays.contains(where: { !$0.hasBackingArray }) {
        return new_mlx_vector_array(arrays.filter(\.hasBackingArray))
    }
    return new_mlx_vector_array(arrays)
}
