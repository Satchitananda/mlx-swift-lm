# Upstream main sync — 2026-10-23

Baseline: upstream `main` at `d02948b` ("Generate JSON properties in the schema's declared order
(#718)"), 49 commits after 3.32.3. This sync deliberately takes unreleased upstream commits; the
owner chose that scope on 2026-10-23. Pair with sibling mlx-swift `sync/upstream-main-20261023`
(fork PR #6), which carries the core missing-stream fixes and upstream mlx-swift main.

Sixteen files conflicted. Rulings, by area:

## Fork patch disposition

| Patch | Disposition |
| --- | --- |
| Local sibling MLX dependency | Retained (`.package(path: "../mlx-swift")`). |
| Qwen 3.8 model support | Replaced by upstream #545 (text, VLM, MTP speculative decoding). Upstream's `Qwen35CheckpointPolicy` / `CheckpointComponent` cover the fork's bare-key standalone MTP loading, vision-key stripping and `model.*` remaps, so the fork's `qwenMTPSanitizeWeights` is gone. `Qwen35.swift`, `Qwen35MTP.swift` (LLM and VLM), `QwenMTP.swift` and `Qwen35VLMMTPRegistration.swift` now equal upstream. |
| DFlash2 drafter runtime and same-input projection stacking | Retained; upstream has no equivalent. The fork's `normalizedQK(q:k:)` helper, shared with `verifyForward`, now carries upstream's GDN eps fix (#633). NVFP4 global-scale rejection stays in `SameInputProjectionStacking`. Speculation stays opt-in. |
| MTP drafter registry entries (Gemma 4 E2B/E4B/12B assistants) | Retained as a superset of upstream's `all()`. |
| Gemma 4 audio and assistant support | Retained. Upstream moved the Gemma 4 vision stack and an audio encoder into MLXLMCommon as `package` types; MLXVLM now reaches the vision types through upstream's typealiases and keeps the fork's `public` `Gemma4AudioConfiguration` / `Gemma4AudioModel` (upstream's are `package`, lack the padded `audioTower(_:audioMelMask:)` the fork's `getAudioFeatures` needs, and `Gemma4Configuration.audioConfiguration` is public). `Gemma4ClippableLinear` stays MLXVLM-internal because `Gemma4Audio.swift` uses it. Tests qualify the two audio type names with `MLXVLM.` where MLXLMTests also sees upstream's `package` types. |
| Native multimodal EmbeddingGemma 2 encoder (fork PR #10) | Retained next to upstream #684. The catalog kit consumes the fork's `EmbeddingGemma2Container` API (`load`, `embedBatch`, `EmbeddingGemma2Part/Image/Frame`); upstream's `EmbeddingGemma2Embedding` has a different API and no batch entry point. Only upstream registers in `MLXEmbedders`. The fork's configuration type is renamed `EmbeddingGemma2ContainerConfiguration` to make room for upstream's `EmbeddingGemma2Configuration` typealias; the fork's test suite is renamed `EmbeddingGemma2ContainerTests`. Follow-up: migrate the catalog kit to upstream's embedder, then retire `MLXVLM/Embeddings/*`. |
| `generation_config.json` sampling propagation | Retained alongside upstream's `suppress_tokens` (#412): `GenerationConfigFile` carries both. |
| Safetensors validation (fail-closed offsets, group arithmetic, payload bounds) | Retained inside upstream's component-aware loader (#679, #580). Upstream still parses headers with `try?` and falls through, so the fork's `safetensorSpansInFileOrder` stays and upstream's read-filter is applied after it. Upstream's loader-thread error propagation (#698) is in. |
| `generateTask(stopStrings:)` override, Markdown fence handling | Retained (no conflicts). |
| Tests | Both sides' tests kept. One fork negative test changed shape: a drafter given only bare target weights now throws `CheckpointComponent.SelectionError` instead of returning an empty dictionary, matching upstream's `sanitize` that now throws. |
| `@Sendable` stateless `compile` shadows (`CompileOverloads.swift`, top of `CompiledTrace.swift`) | Removed. Sibling mlx-swift main (#469) now provides the same 15 overloads, so the shadows made every unqualified `compile(shapeless:_:)` call ambiguous. `CompiledTrace` stores a plain `([MLXArray]) -> [MLXArray]`, because mlx-swift's stateful `compile(inputs:)` now returns a non-Sendable closure. The fork's `testPlainCompileFreezesCapturedWeights` reproduces the capture hazard through `MLX.compile(inputs: [])` now that the stateless overload rejects the capture at build time. Both factories pass the fork's `parsedGenerationConfig` to upstream's `mergeGenerationConfigSuppressedTokens`. This is the first sync against a sibling that is itself ahead of its release. |
| Upstream `.github` changes | None in this range. |

Public API changes inherited from upstream: the Qwen 3.5/3.8 `sanitize(weights:)` methods now throw, and `prepareCheckpoint(_:)` / `excludedCheckpointNamespaces` are new. Downstream callers of those `sanitize` methods need `try`.

## Validation

See the PR for the build and test record on this baseline.

# Published release sync — 2026-10-05

Baseline: [`mlx-swift-lm` 3.32.3](https://github.com/ml-explore/mlx-swift-lm/releases/tag/3.32.3), commit `3b339ad6e3b3f44c8121ecff5131c7fd55e075e6`. Pair with sibling mlx-swift 0.32.3 plus the retained fork fixes. No post-tag upstream commits are included.

## Fork patch disposition

| Patch | Disposition |
| --- | --- |
| Local sibling MLX dependency | Retained to preserve one package identity throughout SightRoll and release trees. |
| Qwen3.8 / DFlash2 runtime and same-input projection stacking | Retained. Upstream does not supply these implementations. NVFP4 global-scale rejection remains. Speculation stays opt-in. |
| Standalone Qwen MTP registrations and bare-key checkpoint compatibility | Retained, with text/VLM registry separation and full-target negative cases. |
| Gemma4 audio and assistant support | Retained. Gemma message construction now uses upstream's label-aware media helper with images → videos → audios → text, without duplicating audio markers. |
| `generation_config.json` sampling propagation | Retained. |
| `generateTask(stopStrings:)` override | Retained alongside upstream's new tools and `toolCallPolicy`. An empty override keeps termination with the grammar. |
| MXFP8 custom layer placeholders and relaxed parameter verification | Removed. Release MLX correctly initializes non-affine quantization; the ordinary loader now uses full `.all` verification. A synthetic checkpoint test covers valid loading and malformed scale-shape rejection. |
| Six eager-evaluation additions in KV-cache fixtures | Removed as separate fork edits; release tests already evaluate their arrays and serialize the relevant suites. `KVCacheTests.swift` equals the release. |
| Uncommitted safetensor validation in the primary checkout | Ported into this worktree, preserving the primary files. Reject negative, fractional, boolean, overflowing and out-of-file offsets before native loading; checked group arithmetic and payload boundaries remain fail-closed. |
| Markdown closing fences in tool-call recovery | Added during shipping review on 2026-10-06. A closing fence must start a line with at most three spaces of indentation and have only spaces or tabs after the fence run. Streaming preserves that boundary across chunks and waits for the complete line, keeping tool-shaped examples inside invalidly closed code blocks inert. |
| Markdown LF, CRLF and CR boundaries | Added during the release-update engineering review. Opening, closing and retained partial fence lines recognize all three CommonMark line endings, including a stream split between the CR and LF scalars. This prevents fenced examples after CRLF/CR prose from being dispatched and preserves valid calls after real closers. The existing regression covers native/alternate calls, both fence characters, invalid closers and scalar chunk boundaries. |
| Upstream `.github` changes | Deferred to a dedicated CI review. Existing fork workflows and templates are unchanged. |

## Validation

The 2026-10-05 baseline native build with complete strict concurrency passed on Xcode 27.0 / Swift 6.4. Focused regressions passed: 41 XCTest tests plus 34 Swift Testing tests for safetensors, MXFP8, stop strings, Gemma audio/media and MTP registration. The full suite passed 673 XCTest executions (2 skipped) and 1,195 Swift Testing tests in 118 suites. These totals predate the shipping-review fence regression; later verification and consumer results are recorded separately in the umbrella update plan and artifacts.

Use freshly compiled sibling Metal kernels and explicit `--no-parallel` for the GPU suites. The host's default Swift Build Metal-wrapper issue is documented in the sibling MLX sync notes; no build or CI checks were weakened to work around it. The installed swift-format reports one existing upstream trailing-closure lint finding in `Gemma4.processVideos`; formatting of the changed code was applied without rewriting that unrelated call.

## Guided generation

The release includes `MLXGuidedGeneration`; its implementation matches the referenced upstream-main README's implementation at inspection time. It can eventually replace the XGrammar backend, but the bundled generation loop cannot preserve all current SightRoll contracts without adaptation. The umbrella assessment documents sampling, reasoning composition, special-token and multi-stop masks, compact JSON, cache reuse and content-free logging requirements. Structured remains in the dependency graph for this update.
