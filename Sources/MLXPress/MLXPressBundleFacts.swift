import Foundation

public enum MLXPressBundleFormat: String, Sendable, Equatable {
    case mlx = "mlx"
    case jang = "jang"
    case jangTQ = "jangtq"
}

public struct MLXPressBundleFacts: Sendable, Equatable {
    public let directory: URL
    public let format: MLXPressBundleFormat
    public let modelType: String?
    public let architecture: MLXPressArchitectureFacts
    public let totalSafetensorsBytes: UInt64
    public let isRouted: Bool
    public let numRoutedExperts: Int?
    public let topK: Int?
    public let hasTokenizerJSON: Bool
    public let hasSafetensorsIndex: Bool
    public let physicalMemoryBytes: UInt64

    public var autoCompressionEligible: Bool {
        isRouted
    }

    public static func inspect(at directory: URL) -> MLXPressBundleFacts {
        let fm = FileManager.default
        let entries = (try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles])) ?? []

        var totalSafetensorsBytes: UInt64 = 0
        for entry in entries where entry.pathExtension == "safetensors" {
            if let values = try? entry.resourceValues(forKeys: [.fileSizeKey]),
                let size = values.fileSize
            {
                totalSafetensorsBytes &+= UInt64(size)
            }
        }

        let config = loadJSONDictionary(
            directory.appendingPathComponent("config.json"))
        var configs = nestedConfigDictionaries(config)
        let jangConfig = loadJSONDictionary(
            directory.appendingPathComponent("jang_config.json"))
        if !jangConfig.isEmpty {
            configs.append(contentsOf: nestedConfigDictionaries(jangConfig))
        }

        let hasJangConfig = fm.fileExists(
            atPath: directory.appendingPathComponent("jang_config.json").path)
        let hasJangTQRuntime = fm.fileExists(
            atPath: directory.appendingPathComponent("jangtq_runtime.safetensors").path)
        let weightFormat = firstString(in: configs, keys: ["weight_format"])?.lowercased()
        let profile = firstString(in: configs, keys: ["profile"])?.lowercased() ?? ""
        let declaresJangTQ = weightFormat == "mxtq" || profile.contains("jangtq")
        let format: MLXPressBundleFormat = if hasJangTQRuntime || declaresJangTQ {
            .jangTQ
        } else if hasJangConfig {
            .jang
        } else {
            .mlx
        }
        let modelType = firstString(in: configs, keys: ["model_type"])
        let numRoutedExperts = firstPositiveInt(
            in: configs,
            keys: ["n_routed_experts", "num_local_experts", "num_experts"])
        let topK = firstPositiveInt(
            in: configs,
            keys: ["num_experts_per_tok", "top_k_experts", "experts_per_token"])
        let isRouted = detectsRoutedMoE(configs: configs)
        let hasTokenizerJSON = fm.fileExists(
            atPath: directory.appendingPathComponent("tokenizer.json").path)
        let hasSafetensorsIndex = fm.fileExists(
            atPath: directory.appendingPathComponent("model.safetensors.index.json").path)

        return MLXPressBundleFacts(
            directory: directory.standardizedFileURL,
            format: format,
            modelType: modelType,
            architecture: MLXPressArchitectureFacts.derive(
                format: format,
                modelType: modelType,
                configs: configs),
            totalSafetensorsBytes: totalSafetensorsBytes,
            isRouted: isRouted,
            numRoutedExperts: numRoutedExperts,
            topK: topK,
            hasTokenizerJSON: hasTokenizerJSON,
            hasSafetensorsIndex: hasSafetensorsIndex,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
    }
}

public struct MLXPressArchitectureFacts: Sendable, Codable, Equatable {
    public let architectureNames: [String]
    public let attentionKinds: [String]
    public let matmulKinds: [String]
    public let positionEncodings: [String]
    public let positionVectorKinds: [String]
    public let cacheStorageKinds: [String]
    public let cacheEncodingKinds: [String]
    public let companionStateKinds: [String]
    public let hybridSplitKinds: [String]
    public let mediaCacheKinds: [String]

    public var attentionSummary: String {
        attentionKinds.isEmpty ? "unknown" : attentionKinds.joined(separator: ", ")
    }

    public var matmulSummary: String {
        matmulKinds.isEmpty ? "unknown" : matmulKinds.joined(separator: ", ")
    }

    public var positionSummary: String {
        positionEncodings.isEmpty ? "unknown" : positionEncodings.joined(separator: ", ")
    }

    public var positionVectorSummary: String {
        positionVectorKinds.isEmpty ? "unknown" : positionVectorKinds.joined(separator: ", ")
    }

    public var cacheStorageSummary: String {
        cacheStorageKinds.isEmpty ? "unknown" : cacheStorageKinds.joined(separator: ", ")
    }

    public var cacheEncodingSummary: String {
        cacheEncodingKinds.isEmpty ? "unknown" : cacheEncodingKinds.joined(separator: ", ")
    }

    public var companionStateSummary: String {
        companionStateKinds.isEmpty ? "none detected" : companionStateKinds.joined(separator: ", ")
    }

    public var hybridSplitSummary: String {
        hybridSplitKinds.isEmpty ? "none detected" : hybridSplitKinds.joined(separator: ", ")
    }

    public var mediaCacheSummary: String {
        mediaCacheKinds.isEmpty ? "not detected" : mediaCacheKinds.joined(separator: ", ")
    }

    static func derive(
        format: MLXPressBundleFormat,
        modelType: String?,
        configs: [[String: Any]]
    ) -> MLXPressArchitectureFacts {
        var probeParts = firstStringArray(in: configs, keys: ["architectures"])
        if let modelType {
            probeParts.append(modelType)
        }
        let modelProbe = probeParts
            .map { $0.lowercased() }
            .joined(separator: " ")
        let routed = detectsRoutedMoE(configs: configs)
        let visionFlags = configs.compactMap { boolValue($0["has_vision"]) }
        let explicitlyVisionEnabled = visionFlags.contains(true)
        let explicitlyVisionDisabled = !explicitlyVisionEnabled && visionFlags.contains(false)
        let hasExplicitVisionModel = explicitlyVisionEnabled || configs.contains { config in
            if let visionConfig = config["vision_config"] as? [String: Any],
                !visionConfig.isEmpty
            {
                return true
            }
            guard let nestedType = config["model_type"] as? String else { return false }
            return containsVisionMarker(nestedType.lowercased())
        }
        let isVL = !explicitlyVisionDisabled
            && (containsVisionMarker(modelProbe) || hasExplicitVisionModel)
        let isQwenVLM = isVL && modelProbe.contains("qwen")
        let isZaya = modelProbe.contains("zaya")
        let isDeepseekV4 = modelProbe.contains("deepseek_v4") || modelProbe.contains("dsv4")
        let hasMLAPartialRoPE = hasAnyKey(
            in: configs,
            keys: ["q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim"])

        var attentionKinds: [String] = []
        if routed {
            appendUnique("routed-moe", to: &attentionKinds)
        }
        if hasMLAPartialRoPE {
            appendUnique("mla-partial-rope-attention", to: &attentionKinds)
        }
        if hasAnyKey(
            in: configs,
            keys: ["sliding_window", "max_window_layers", "use_sliding_window", "window_size"])
        {
            appendUnique("sliding-window-attention", to: &attentionKinds)
        }
        if hasAnyKey(in: configs, keys: ["full_attention_interval", "hybrid_override_pattern"])
            || hasAnyKey(in: configs, keys: ["linear_num_key_heads", "linear_conv_kernel_dim"])
        {
            appendUnique("hybrid-linear-or-ssm-attention", to: &attentionKinds)
        }
        if isDeepseekV4 {
            appendUnique("compressor-indexer-attention", to: &attentionKinds)
        }
        if isZaya {
            appendUnique("zaya-cca-attention", to: &attentionKinds)
        }
        if isVL {
            appendUnique("vl-encoder-plus-text-attention", to: &attentionKinds)
        }
        if attentionKinds.isEmpty {
            appendUnique("dense-full-attention", to: &attentionKinds)
        }

        var matmulKinds: [String] = []
        if format == .jangTQ {
            appendUnique("jangtq-hadamard-rotation", to: &matmulKinds)
            appendUnique("randomized-hadamard-pow2-blocks", to: &matmulKinds)
            appendUnique("turboquant-codebook-gather(tq_packed+tq_norms)", to: &matmulKinds)
            appendUnique("jangtq-runtime-signs-and-codebooks", to: &matmulKinds)
            if routed {
                appendUnique("routed-active-expert-slice-matmul", to: &matmulKinds)
                appendUnique("fused-gate-up-swiglu-tq", to: &matmulKinds)
            }
        } else if routed {
            appendUnique("routed-affine-or-mxfp4-matmul", to: &matmulKinds)
        } else {
            appendUnique("dense-mlx-matmul", to: &matmulKinds)
        }

        var positionEncodings: [String] = []
        let ropeParameterDictionaries = firstDictionaries(
            in: configs,
            keys: ["rope_scaling", "rope_parameters"])
        let qkRopeDim = firstPositiveInt(in: configs, keys: ["qk_rope_head_dim"])
        if let qkRope = qkRopeDim {
            appendUnique("partial-rope(qk_rope_head_dim=\(qkRope))", to: &positionEncodings)
        }
        if let ropeTheta = firstNumberString(in: configs, keys: ["rope_theta", "rotary_base"])
            ?? firstNumberString(in: ropeParameterDictionaries, keys: ["rope_theta", "rotary_base"])
        {
            appendUnique("rope(theta=\(ropeTheta))", to: &positionEncodings)
        }
        if let ropePct = firstNumberString(in: configs, keys: ["rope_pct", "partial_rotary_factor"])
            ?? firstNumberString(in: ropeParameterDictionaries, keys: ["partial_rotary_factor"])
        {
            appendUnique("partial-rotary-factor(\(ropePct))", to: &positionEncodings)
        }
        var hasMropeSection = false
        var hasMropeInterleaved = false
        for scaling in ropeParameterDictionaries {
            let type = (scaling["type"] as? String) ?? (scaling["rope_type"] as? String) ?? "configured"
            appendUnique("rope-scaling(\(type))", to: &positionEncodings)
            if let section = intArrayValue(scaling["mrope_section"]), !section.isEmpty {
                hasMropeSection = true
                appendUnique("mrope-section(\(section.map(String.init).joined(separator: "/")))", to: &positionEncodings)
            } else if scaling.keys.contains("mrope_section") {
                hasMropeSection = true
                appendUnique("mrope-section", to: &positionEncodings)
            }
            if let interleaved = scaling["mrope_interleaved"] as? Bool, interleaved {
                hasMropeInterleaved = true
            } else if scaling.keys.contains("mrope_interleaved") {
                hasMropeInterleaved = true
            }
        }
        if hasAnyKey(in: configs, keys: ["mrope_section", "mrope_interleaved"]) {
            hasMropeSection = hasMropeSection || hasAnyKey(in: configs, keys: ["mrope_section"])
            hasMropeInterleaved = hasMropeInterleaved || hasAnyKey(in: configs, keys: ["mrope_interleaved"])
            appendUnique("mrope-section", to: &positionEncodings)
        }
        if hasMropeInterleaved {
            appendUnique("mrope-interleaved", to: &positionEncodings)
        }
        if isQwenVLM {
            appendUnique("2d-3d-mrope-grid", to: &positionEncodings)
            appendUnique("qwen-vl-2d-image-3d-video-mrope-grid", to: &positionEncodings)
        }

        var positionVectorKinds: [String] = []
        appendUnique("scalar-token-position-offset", to: &positionVectorKinds)
        appendUnique("text-position-ids[batch,seq]", to: &positionVectorKinds)
        if let qkRope = qkRopeDim {
            appendUnique("partial-rope-head-vector(qk=\(qkRope))", to: &positionVectorKinds)
        }
        if attentionKinds.contains("sliding-window-attention") {
            appendUnique("sliding-window-offset-vector", to: &positionVectorKinds)
        }
        if hasMropeSection {
            appendUnique("mrope-section-vector", to: &positionVectorKinds)
            appendUnique("mrope-three-axis-head-split", to: &positionVectorKinds)
        }
        if isQwenVLM {
            appendUnique("qwen-vl-position-ids[3,batch,seq]", to: &positionVectorKinds)
            appendUnique("qwen-vl-image-grid-thw(2d-spatial)", to: &positionVectorKinds)
            appendUnique("qwen-vl-video-grid-thw(3d-temporal-spatial)", to: &positionVectorKinds)
            appendUnique("qwen-vl-mrope-delta-vector", to: &positionVectorKinds)
        }
        if isVL {
            appendUnique("media-token-position-vector", to: &positionVectorKinds)
        }
        if isZaya {
            appendUnique("zaya-cca-offset-array", to: &positionVectorKinds)
        }

        var cacheStorageKinds: [String] = []
        appendUnique("paged-cacheblock-chain", to: &cacheStorageKinds)
        appendUnique("paged-cacheblock-chain(parent-hash)", to: &cacheStorageKinds)
        appendUnique("disk-l2-safetensors", to: &cacheStorageKinds)
        appendUnique("disk-l2-safetensors(token-hash)", to: &cacheStorageKinds)
        appendUnique("prompt-boundary-raw-kv-snapshot", to: &cacheStorageKinds)
        if format == .jangTQ {
            appendUnique("jangtq-runtime-sidecar", to: &cacheStorageKinds)
            appendUnique("turboquant-kv-eligible", to: &cacheStorageKinds)
        }
        if attentionKinds.contains("sliding-window-attention") {
            appendUnique("rotating-kv-metadata", to: &cacheStorageKinds)
        }
        if isDeepseekV4 {
            appendUnique("hybrid-pool-cache-blocks", to: &cacheStorageKinds)
        }
        if isZaya {
            appendUnique("zaya-cca-cache-blocks", to: &cacheStorageKinds)
        }
        if isVL {
            appendUnique("media-salted-cache-keys", to: &cacheStorageKinds)
        }

        var cacheEncodingKinds: [String] = []
        appendUnique("cacheblock-sha256(modelKey,mediaSalt,parentHash,tokenIdsRaw)", to: &cacheEncodingKinds)
        appendUnique("disk-l2-sha256(modelKey,mediaSalt,tokenJson)", to: &cacheEncodingKinds)
        appendUnique("cache-policy-salt(kvMode,kvBits,kvGroup,maxKV,promptBoundaryRawKV)", to: &cacheEncodingKinds)
        appendUnique("tqdiskserializer-v2-layer-kind-tags", to: &cacheEncodingKinds)
        appendUnique("layer-kinds(kv,tq,qkv,mamba,rotating,deepseekV4,cacheList,zayaCCA)", to: &cacheEncodingKinds)
        if format == .jangTQ {
            appendUnique("turboquant-kv-eligible", to: &cacheEncodingKinds)
            appendUnique("prompt-boundary-disk-raw-kv-before-tq-generated-boundary", to: &cacheEncodingKinds)
        }
        if attentionKinds.contains("sliding-window-attention") {
            appendUnique("rotating-kv-meta(keep,maxSize,step,offset,idx)", to: &cacheEncodingKinds)
        }
        if isDeepseekV4 {
            appendUnique("dsv4-compressor-indexer-pool-encode", to: &cacheEncodingKinds)
        }
        if isZaya {
            appendUnique("zaya-cca-four-array-state(keys,values,conv_state,prev_hs)", to: &cacheEncodingKinds)
        }
        if isVL {
            appendUnique("media-scope-salt(bytes,shape,dtype,processor,reasoning)", to: &cacheEncodingKinds)
            appendUnique("vl-partial-hit-rejected-for-media-token-region", to: &cacheEncodingKinds)
        }

        var companionStateKinds: [String] = []
        if isZaya {
            appendUnique("zaya-cca(conv_state,prev_hs)", to: &companionStateKinds)
        }
        if isDeepseekV4 {
            appendUnique("dsv4(compressor,indexer,pool,buffers)", to: &companionStateKinds)
        }
        if attentionKinds.contains("hybrid-linear-or-ssm-attention") {
            appendUnique("ssm-or-linear-recurrence-state", to: &companionStateKinds)
        }
        if !companionStateKinds.isEmpty {
            appendUnique("path-dependent-hit-reject-or-serialize-companion-state", to: &cacheEncodingKinds)
        }

        var hybridSplitKinds: [String] = []
        if format == .jangTQ && routed {
            appendUnique("routed-expert-weight-split(active-slice-vs-prestack-overlay)", to: &hybridSplitKinds)
        }
        if isZaya {
            appendUnique("zaya-cca-kv-plus-conv_state-prev_hs", to: &hybridSplitKinds)
        }
        if isDeepseekV4 {
            appendUnique("dsv4-kv-plus-compressor-indexer-pools", to: &hybridSplitKinds)
            appendUnique("dsv4-rotating-kv-plus-incomplete-window-buffers", to: &hybridSplitKinds)
        }
        if attentionKinds.contains("hybrid-linear-or-ssm-attention") {
            appendUnique("ssm-linear-kv-plus-recurrence-state", to: &hybridSplitKinds)
        }
        if isVL {
            appendUnique("vl-media-embeddings-plus-text-kv-no-partial-media-hit", to: &hybridSplitKinds)
        }

        var mediaCacheKinds: [String] = []
        if isVL {
            appendUnique("image-video-audio-bytes-shape-dtype-salt", to: &mediaCacheKinds)
            appendUnique("processor-grid-scope", to: &mediaCacheKinds)
        }
        if isQwenVLM {
            appendUnique("mrope-grid-position-ids", to: &mediaCacheKinds)
            appendUnique("qwen-vl-grid-thw-and-delta", to: &mediaCacheKinds)
        }

        return MLXPressArchitectureFacts(
            architectureNames: firstStringArray(in: configs, keys: ["architectures"]),
            attentionKinds: attentionKinds,
            matmulKinds: matmulKinds,
            positionEncodings: positionEncodings,
            positionVectorKinds: positionVectorKinds,
            cacheStorageKinds: cacheStorageKinds,
            cacheEncodingKinds: cacheEncodingKinds,
            companionStateKinds: companionStateKinds,
            hybridSplitKinds: hybridSplitKinds,
            mediaCacheKinds: mediaCacheKinds)
    }
}

private func loadJSONDictionary(_ url: URL) -> [String: Any] {
    guard let data = try? Data(contentsOf: url),
        let object = try? JSONSerialization.jsonObject(with: data),
        let dictionary = object as? [String: Any]
    else {
        return [:]
    }
    return dictionary
}

private func boolValue(_ value: Any?) -> Bool? {
    if let value = value as? Bool {
        return value
    }
    if let value = value as? String {
        switch value.lowercased() {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            return nil
        }
    }
    if let value = value as? Int {
        return value != 0
    }
    return nil
}

private func nestedConfigDictionaries(_ root: [String: Any]) -> [[String: Any]] {
    var dictionaries = [root]
    for key in ["text_config", "language_config", "llm_config", "vision_config"] {
        if let nested = root[key] as? [String: Any] {
            dictionaries.append(nested)
        }
    }
    return dictionaries
}

private func detectsRoutedMoE(configs: [[String: Any]]) -> Bool {
    let routedKeys = [
        "num_local_experts",
        "num_experts",
        "moe_intermediate_size",
        "n_routed_experts",
    ]
    for config in configs {
        if firstPositiveInt(in: [config], keys: routedKeys) ?? 0 > 1 {
            return true
        }
        if let experts = config["experts"] as? [String: Any], !experts.isEmpty {
            return true
        }
    }
    return false
}

private func firstString(in configs: [[String: Any]], keys: [String]) -> String? {
    for config in configs {
        for key in keys {
            if let value = config[key] as? String, !value.isEmpty {
                return value
            }
        }
    }
    return nil
}

private func firstStringArray(in configs: [[String: Any]], keys: [String]) -> [String] {
    var values: [String] = []
    for config in configs {
        for key in keys {
            if let string = config[key] as? String, !string.isEmpty {
                appendUnique(string, to: &values)
            }
            if let array = config[key] as? [String] {
                for string in array where !string.isEmpty {
                    appendUnique(string, to: &values)
                }
            }
            if let array = config[key] as? [Any] {
                for value in array {
                    guard let string = value as? String, !string.isEmpty else { continue }
                    appendUnique(string, to: &values)
                }
            }
        }
    }
    return values
}

private func firstPositiveInt(in configs: [[String: Any]], keys: [String]) -> Int? {
    for config in configs {
        for key in keys {
            if let value = config[key] as? Int, value > 0 {
                return value
            }
            if let value = config[key] as? NSNumber, value.intValue > 0 {
                return value.intValue
            }
        }
    }
    return nil
}

private func firstNumberString(in configs: [[String: Any]], keys: [String]) -> String? {
    for config in configs {
        for key in keys {
            if let value = config[key] as? Int {
                return String(value)
            }
            if let value = config[key] as? Double {
                return formatConfigNumber(value)
            }
            if let value = config[key] as? Float {
                return formatConfigNumber(Double(value))
            }
            if let value = config[key] as? NSNumber {
                return formatConfigNumber(value.doubleValue)
            }
        }
    }
    return nil
}

private func firstDictionary(in configs: [[String: Any]], keys: [String]) -> [String: Any]? {
    for config in configs {
        for key in keys {
            if let dictionary = config[key] as? [String: Any] {
                return dictionary
            }
        }
    }
    return nil
}

private func firstDictionaries(in configs: [[String: Any]], keys: [String]) -> [[String: Any]] {
    var dictionaries: [[String: Any]] = []
    for config in configs {
        for key in keys {
            if let dictionary = config[key] as? [String: Any] {
                dictionaries.append(dictionary)
            }
        }
    }
    return dictionaries
}

private func intArrayValue(_ value: Any?) -> [Int]? {
    if let values = value as? [Int] {
        return values
    }
    if let values = value as? [NSNumber] {
        return values.map(\.intValue)
    }
    if let values = value as? [Any] {
        let ints = values.compactMap { value -> Int? in
            if let int = value as? Int { return int }
            if let number = value as? NSNumber { return number.intValue }
            return nil
        }
        return ints.count == values.count ? ints : nil
    }
    return nil
}

private func hasAnyKey(in configs: [[String: Any]], keys: [String]) -> Bool {
    for config in configs {
        for key in keys where config.keys.contains(key) {
            return true
        }
    }
    return false
}

private func containsVisionMarker(_ value: String) -> Bool {
    value.contains("_vl")
        || value.contains("-vl")
        || value.contains(" vl")
        || value.contains("vl_")
        || value.contains("vl-")
        || value.contains("vlfor")
        || value.contains("vision")
}

private func appendUnique(_ value: String, to values: inout [String]) {
    guard !values.contains(value) else { return }
    values.append(value)
}

private func formatConfigNumber(_ value: Double) -> String {
    if value.rounded() == value {
        return String(Int64(value))
    }
    return String(value)
}
