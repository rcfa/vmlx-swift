import VMLXSwift

let args = CommandLine.arguments.dropFirst()

if args.contains("--version") || args.contains("version") {
    print("\(VMLXSwift.packageName) facade")
} else {
    print("""
    vmlx-swift facade

    Commands:
      version       Print package identity
      check         Verify the executable links against the facade

    This initial CLI is intentionally lightweight. Model loading and runtime
    smoke commands will be added after the source-import phase has a stable
    command surface.
    """)
}
