// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "vmlx-swift",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "VMLXSwift",
            targets: ["VMLXSwift"]
        ),
        .executable(
            name: "vmlx-swift",
            targets: ["vmlx-swift"]
        )
    ],
    dependencies: [
        .package(
            url: "https://github.com/osaurus-ai/mlx-swift.git",
            revision: "0a56f9041d56b4b8161f67a6cbd540ae66efc9fd"
        ),
        .package(
            url: "https://github.com/osaurus-ai/vmlx-swift-lm.git",
            revision: "b166896353b9c95d773de993990c20a0b5ba6905"
        ),
        .package(
            url: "https://github.com/osaurus-ai/swift-transformers.git",
            revision: "087a66b17e482220b94909c5cf98688383ae481a"
        ),
        .package(
            url: "https://github.com/osaurus-ai/Jinja.git",
            revision: "58d21aa5b69fdd9eb7e23ce2c3730f47db8e0c9d"
        ),
    ],
    targets: [
        .target(
            name: "VMLXSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),

                .product(name: "MLXLMCommon", package: "vmlx-swift-lm"),
                .product(name: "MLXLLM", package: "vmlx-swift-lm"),
                .product(name: "MLXVLM", package: "vmlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "vmlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "vmlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Transformers", package: "swift-transformers"),

                .product(name: "Jinja", package: "jinja"),
            ]
        ),
        .executableTarget(
            name: "vmlx-swift",
            dependencies: ["VMLXSwift"]
        )
    ]
)
