@_exported import Generation
@_exported import Hub
@_exported import Jinja
@_exported import MLX
@_exported import MLXEmbedders
@_exported import MLXFFT
@_exported import MLXFast
@_exported import MLXHuggingFace
@_exported import MLXLLM
@_exported import MLXLMCommon
@_exported import MLXLinalg
@_exported import MLXNN
@_exported import MLXOptimizers
@_exported import MLXRandom
@_exported import MLXVLM
@_exported import Models
@_exported import Tokenizers

/// Marker type for consumers that need to verify they are linked against the
/// unified Osaurus vmlx Swift facade package.
public enum VMLXSwift {
    public static let packageName = "vmlx-swift"
}
