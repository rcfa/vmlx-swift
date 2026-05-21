# Vendored Runtime Libraries

This directory contains source copies of the Swift language-runtime libraries
that vmlx-swift exposes directly, so downstream apps can depend on one package
instead of pinning the old split repos.

Vendored into package targets:

- `Jinja`: from `osaurus-ai/Jinja`, revision `0aeefadec459ce8e11a333769950fb86183aca43`.
- `Hub`, `Tokenizers`, `Generation`, `Models`: from `osaurus-ai/swift-transformers`, revision `087a66b17e482220b94909c5cf98688383ae481a`.
- `HuggingFace`: from `huggingface/swift-huggingface`, revision `b721959445b617d0bf03910b2b4aced345fd93bf`.
- `EventSource`: from `mattt/EventSource`, revision `a3a85a85214caf642abaa96ae664e4c772a59f6e`.
- `yyjson`: from `ibireme/yyjson`, revision `8b4a38dc994a110abaec8a400615567bd996105f`.

The package still keeps external SwiftPM dependencies for Apple/NIO/Crypto
infrastructure (`swift-numerics`, `swift-syntax`, `swift-nio`,
`swift-nio-ssl`, `swift-nio-http2`, `swift-certificates`, and
`swift-crypto`) plus Apple `swift-collections` for the `OrderedCollections`
module used by Jinja. Those are not model/runtime template or tokenizer repos
and remain upstream dependencies intentionally.

When updating these vendors, copy source plus license/readme files together,
then run `swift package describe`, focused builds for `Jinja`, `Tokenizers`,
`Hub`, `MLXPress`, `RunBench`, and the Osaurus-facing tests before moving the
new pin into Osaurus.
