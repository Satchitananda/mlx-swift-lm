# Qwen3.8 runtime integration

This fork integrates merged upstream runtime/test changes through
`604fae710a4e3324346fc59e3845952350acd4b7`, preserving:

- [PR545](https://github.com/ml-explore/mlx-swift-lm/pull/545), head
  `1a562aa00bb66d611a086174e14951f41c43e100`: Qwen3.8 language/vision model
  configuration, layer schedules, cache anchors and MTP registration.
- [PR607](https://github.com/ml-explore/mlx-swift-lm/pull/607), head
  `65c7f45eb5c9c58cafe1300f0ed356d742155221`: DFlash2 text speculative decoding.

PR607 does not include PR545. The runtime prerequisite update also includes
Qwen checkpoint norm sanitization and processor-loading fixes. The September 14
sync includes upstream reference documentation alongside the runtime and unit
tests. Fork CI/workflows, the separate integration project, and contributor policy
are retained.

The sync includes generation worker isolation (#611), Unicode streaming fixes
(#613), Gemma VLM softcapping fusion (#615), differentiable gated-delta recurrence
(#616), configuration-based LoRA metadata (#597), and wrapped rotating-cache
trimming (#584). Pair it with the sibling MLX fork synced through
`ea8a179690170ca891a97bc0473198ab1ecda5f4`, including wired-memory cancellation
handling (#471) and the core update from upstream #450. No additional open PRs
are included, and speculative decoding remains opt-in.

The fork retains the local `../mlx-swift` dependency, Swift tools 6.2, Gemma4
audio support and assistant registrations, generation-config propagation, and
the explicit `generateTask(stopStrings:)` override used by grammar-constrained
generation. Passing an empty stop set leaves termination with the grammar.
Projection stacking refuses a linear with a per-tensor global scale, because
the sibling fork's NVFP4 representation cannot safely discard that scale.

## Consumer compatibility

The synced sibling MLX package requires Swift 6.3 or newer (including its
experimental C generation support). Validation uses Xcode 26.6 / Swift 6.3.3.


Use the accompanying mlx-swift-structured compatibility change. Its constrained
iterator handles throwing cache creation, the current prefill policy and public
iterator state; examples handle rejected tool calls. These API adaptations do
not add DFlash2 support to the constrained iterator.

The current DFlash2 targets are `MLXLLM.Qwen35TextModel` and `MLXLLM.Qwen35Model`. It rejects visual input and
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

In the earlier 2026-09-06 validation, Structured's full suite passed 30 tests.
CatalogKit's deterministic core passed
927 tests against this runtime before the final benchmark-runner review. Actual
four-model output, failure, timing and memory evidence is retained in private
SightRoll benchmark reports, outside these repositories.

## Upstream sync validation — 2026-09-14

The merged upstream runtime at `604fae710a4e3324346fc59e3845952350acd4b7`,
paired with the sibling MLX integration at
`76bb2f25e1708867b50feeec84bdea6f67a5e9fe`, passed a fresh host build and the
serial full-precision suite: 638 XCTest executions (two skipped, zero failures)
and 853 Swift Testing tests across 77 suites. The existing disabled
`ConstraintCachingTests` suite remains unavailable for the reason documented
above. No optional real-checkpoint benchmark is claimed by these totals.

Six cache-validation fixtures now materialize loaded tensors before rewriting
the file they came from. This avoids reading lazy arrays from a file while it
is being replaced; typed cache validation assertions remain unchanged.

The paired MLX core passed 584 MLX tests and one C bridge test. Structured
passed all 35 tests across nine suites, and CatalogKit built against these
updated dependencies. Model-quality and performance measurements run separately
with production TF32 settings and frozen benchmark inputs.
