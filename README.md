# vmlx-swift

Unified Osaurus Swift runtime facade for MLX, vmlx-swift-lm, swift-transformers,
and Jinja.

This repository starts as a pinned compatibility package. It gives Osaurus one
package identity to import while the underlying libraries continue to live in
their current repositories. The migration target is a true monorepo where the
same products are vendored and built from one source tree.

## Current package shape

The `VMLXSwift` product re-exports:

- MLX products from `osaurus-ai/mlx-swift`
- language, vision-language, embedding, and runtime helper products from
  `osaurus-ai/vmlx-swift-lm`
- tokenizer, generation, model, and Hub modules from
  `osaurus-ai/swift-transformers`
- Jinja from `osaurus-ai/Jinja`

Current pinned source SHAs:

| Dependency | Revision |
|---|---|
| `osaurus-ai/mlx-swift` | `0a56f9041d56b4b8161f67a6cbd540ae66efc9fd` |
| `osaurus-ai/vmlx-swift-lm` | `b166896353b9c95d773de993990c20a0b5ba6905` |
| `osaurus-ai/swift-transformers` | `087a66b17e482220b94909c5cf98688383ae481a` |
| `osaurus-ai/Jinja` | `58d21aa5b69fdd9eb7e23ce2c3730f47db8e0c9d` |

## Build

```sh
swift package resolve
swift build --target VMLXSwift
swift run vmlx-swift version
```

## Migration rule

Do not move Osaurus onto this package until:

1. This package resolves and builds from clean checkout.
2. Osaurus runtime policy tests pass against this package.
3. The runtime coverage matrix in `docs/RUNTIME_COVERAGE_MATRIX.md` has at
   least one real-model row for each architecture bucket that Osaurus ships.
4. Package pins are remote SHAs, not local paths.
5. No local-only fork or dirty working tree is used as a hidden dependency.

## Future phases

1. **Facade**: current state. One import surface, pinned upstream repos.
2. **Vendored package sources**: move MLX, vmlx-swift-lm, swift-transformers,
   and Jinja sources under one tree while preserving product names.
3. **Osaurus repin**: Osaurus consumes only `osaurus-ai/vmlx-swift`.
4. **Legacy repo deprecation**: old repos stay as mirrors or upstream-sync
   sources until all app and engine checks pass from this repo alone.

Distributed/JACCL products are intentionally not re-exported in the first
facade commit. At the current pins they require MLX C distributed headers that
are not present in the pinned `osaurus-ai/mlx-swift` package. They should be
added only with a buildable MLX distributed C surface and a dedicated runtime
validation row.
