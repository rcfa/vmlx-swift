import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public struct MLXPressRuntimeCapabilities: Sendable, Equatable {
    public let hasCanonicalMmapRoutedAdvisor: Bool
    public let hasCanonicalMmapExpertAdvisor: Bool
    public let hasCanonicalMmapLayerAdvisor: Bool

    public var hasCanonicalMmapSafetensorsSupport: Bool {
        hasCanonicalMmapRoutedAdvisor && hasCanonicalMmapExpertAdvisor
            && hasCanonicalMmapLayerAdvisor
    }

    public var activityCompressionReady: Bool {
        hasCanonicalMmapSafetensorsSupport
    }

    public static func current() -> MLXPressRuntimeCapabilities {
        MLXPressRuntimeCapabilities(
            hasCanonicalMmapRoutedAdvisor: symbolAvailable(
                "mlx_safetensors_mmap_advise_routed"),
            hasCanonicalMmapExpertAdvisor: symbolAvailable(
                "mlx_safetensors_mmap_advise_experts"),
            hasCanonicalMmapLayerAdvisor: symbolAvailable(
                "mlx_safetensors_mmap_advise_layer"))
    }
}

#if canImport(Darwin) || canImport(Glibc)
private func symbolAvailable(_ name: String) -> Bool {
    guard let handle = dlopen(nil, RTLD_LAZY) else { return false }
    return dlsym(handle, name) != nil
}
#else
private func symbolAvailable(_ name: String) -> Bool {
    false
}
#endif
