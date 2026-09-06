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

## Standalone MTP weight compatibility

The `qwen3_5_mtp` and `qwen3_8_mtp` standalone drafter registrations accept
converted checkpoints with either bare predictor keys (`fc.*`, `layers.*`,
`norm.*`, `pre_fc_norm_*`) or `mtp.`-prefixed keys. Select the text registry
through `Qwen35TextMTPRegistration.register()` or the vision-language registry
through `Qwen35VLMMTPRegistration.register()` for the corresponding target.
The existing weight loader handles both layouts without editing checkpoint files.

Bare keys receive the `mtp.` namespace only for standalone registrations when
no prefixed keys exist. Already prefixed checkpoints keep their existing
filtering; full target checkpoints never reinterpret bare target weights as MTP
parameters. Converted norm values and quantized weights, scales and biases are
preserved. This compatibility fix does not enable production MTP or change
model defaults, sampling, block size or speculative decoding algorithms.

`Qwen35MTPRegistrationTests` writes tiny synthetic quantized safetensors into
unique temporary directories and loads them through the production weight
loader. Its four text/vision and bare/prefixed combinations verify exact
parameter equality; both bare variants reproduced missing-parameter failures
before the fix and pass afterward. Two further cases verify that full text and
vision checkpoints reject bare target weights. Fixtures require no downloaded
checkpoint or user data and remove their temporary directories afterward.

## Validation

On the M5 Max host, a fresh scratch build and serial full-precision tests passed:

```sh
swift build --build-tests --scratch-path /private/tmp/qwen38-lm-clean
# Copy a freshly built sibling mlx.metallib into the XCTest bundle before running.
MLX_ENABLE_TF32=0 swift test --scratch-path /private/tmp/qwen38-lm-clean \
  --skip-build --no-parallel
```

For the standalone loading fix `600821e`, the 2026-09-06 full run reported
612 XCTest executions: 611 passes, one skipped real-checkpoint benchmark and
zero failures. Swift Testing reported 835 tests across 77 suites with zero
failures; the log also explicitly lists the existing disabled three-test
`ConstraintCachingTests` suite, whose required XGrammar fork API is unavailable
in the vendored version. These counts are dated evidence, not a claim that every
optional integration test ran.
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
