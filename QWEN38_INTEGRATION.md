# Qwen3.8 runtime integration

This branch integrates upstream runtime/test changes through
`e3d4a20e9e20e7b8ab39aded7bbfad4ae22c9438`, followed by:

- [PR545](https://github.com/ml-explore/mlx-swift-lm/pull/545), head
  `1a562aa00bb66d611a086174e14951f41c43e100`: Qwen3.8 language/vision model
  configuration, layer schedules, cache anchors and MTP registration.
- [PR607](https://github.com/ml-explore/mlx-swift-lm/pull/607), head
  `65c7f45eb5c9c58cafe1300f0ed356d742155221`: DFlash2 text speculative decoding.

PR607 does not include PR545. The runtime prerequisite update also includes
Qwen checkpoint norm sanitization and processor-loading fixes. Imported paths are
`Libraries`, `Tests` and `Package.swift`; upstream CI/workflows are not imported.

The fork retains the local `../mlx-swift` dependency, Swift tools 6.2, Gemma4
audio support and assistant registrations, generation-config propagation, and
the explicit `generateTask(stopStrings:)` override used by grammar-constrained
generation. Passing an empty stop set leaves termination with the grammar.
Projection stacking refuses a linear with a per-tensor global scale, because
the sibling fork's NVFP4 representation cannot safely discard that scale.

## Consumer compatibility

Use the accompanying mlx-swift-structured compatibility change. Its constrained
iterator handles throwing cache creation, the current prefill policy and public
iterator state; examples handle rejected tool calls. These API adaptations do
not add DFlash2 support to the constrained iterator.

The current DFlash2 target is `MLXLLM.Qwen35`. It rejects visual input and
unsupported caches; the production `MLXVLM` target is not supported. SightRoll's
grammar processor also lacks an independent state-copy contract for speculative
branches. DFlash2 remains disabled in the four-model SightRoll benchmark.

Qwen3.8 checkpoints use the `qwen3_5` architecture. SightRoll's baseline model
comparison disables thinking at its structured/visual prompt boundary, pins
checkpoint revisions, and measures standard decoding. The Director's explicit
reasoning controls are a separate configuration. Benchmark outcomes do not
automatically change production model defaults.

## Validation

On the M5 Max host, a fresh scratch build and serial full-precision tests passed:

```sh
swift build --build-tests --scratch-path /private/tmp/qwen38-lm-clean
# Copy a freshly built sibling mlx.metallib into the XCTest bundle before running.
MLX_ENABLE_TF32=0 swift test --scratch-path /private/tmp/qwen38-lm-clean \
  --skip-build --no-parallel
```

Result: 611 XCTest passes, one skip, and 833 Swift Testing passes across 77 suites.
The build used the sibling MLX fork at
`de8a9179a1cd7f68b6322d09771e9db39cb73cff`.

Full precision is needed by existing tight numerical parity fixtures on this
host: default TF32 changes GEMM/GEMV rounding. Test tolerances were not loosened.
An old scratch build mixed stale SwiftSyntax modules with newer prebuilt archives
and crashed the macro test; a fresh scratch build fixed it. Parallel model tests
stalled on shared MLX evaluation/compiled-function locks, so the full gate ran
serially. The SightRoll performance benchmark separately records and uses the
production TF32 default.

Structured's full suite passed 30 tests. CatalogKit's deterministic core passed
927 tests against this runtime before the final benchmark-runner review. Actual
four-model output, failure, timing and memory evidence is retained in private
SightRoll benchmark reports, outside these repositories.
