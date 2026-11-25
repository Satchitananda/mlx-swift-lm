# Upstream main sync — 2026-11-25

Baseline: upstream `main` at `6759555` ("Gemma 4: apply bidirectional image attention on a warm
KV cache (#813)"), 69 commits after the 2026-10-23 baseline `d02948b`. Pair with sibling
mlx-swift `sync/upstream-main-20261125` (upstream main `00afc1dd`, core v0.32.4 plus #4746 and
#4818). Plan: `plans/wip/MLX_FORKS_UPSTREAM_SYNC_20261125_PLAN.md`, lane B, task M3.

Branch history: `a060cbd` drops DFlash2 on fork main before the merge, then `d22adec` merges
upstream. The last fork commit carrying DFlash2 is tagged `fork/dflash2-last-20261125`
(`5b9d525`, local tag).

## Fork patch disposition

| Patch | Disposition |
| --- | --- |
| Local sibling MLX dependency | Retained (`.package(path: "../mlx-swift")`). Upstream now requires mlx-swift 0.33.0 or newer, which the sibling's upstream-main sync provides. |
| DFlash2 drafter runtime, `SameInputProjectionStacking`, `DFlash2Registration` | **Dropped** (owner decision D3). Nothing in SightRoll used it, and upstream's Qwen 3.5/3.8 MTP drafts for both text and VLM targets. Removed: six DFlash2/stacking sources, `DFlash2Tests`, `ProjectionStackingCompatibilityTests`, the `generate`/`generateTokens` overloads taking `dflash2Drafter:`, and the verify-pass, rope-in-trace and stacked-projection hooks in `Qwen35.swift` / `Qwen3Next.swift`, which now equal upstream. `QWEN38_INTEGRATION.md` is gone; README and CONTRIBUTING point here. |
| `Linux/Logger.swift` shims (MLXLMCommon, MLXGuidedGeneration) | Dropped with upstream #702 (`MLXLogger`). The fork had not changed them since the merge base, and no fork-only code uses `os.Logger`. |
| MTP drafter registry: Gemma 4 E2B/E4B/12B assistants | Retained as a superset of upstream's `all()`, which now also lists the LFM2 DSpark drafters. |
| Gemma 4 audio (`Gemma4Audio*.swift`, processor audio branches, `Gemma4AudioConfiguration`) | Retained. Upstream #730/#813 moved E-series multimodal embedding into `multimodalInputs(_:)`, shared by `prepare` and `hiddenStates(_:)`; it now passes the audio features and mask, and both entry points treat audio-only input as multimodal. Early-return audit below. |
| Safetensors fail-closed validation | Retained. It merged cleanly with upstream #711 (skip absent checkpoint components): headers are still parsed with `try`, never `try?`, and #711's read filter applies after parsing. |
| `generateTask(stopStrings:)` override | Retained, before upstream's new `reasoningPrimedInside:` argument. |
| Markdown fence LF/CRLF/CR tool-call recovery | Retained. The merge did not touch it. |
| `embed(sharingVideo:variants:task:dimensions:)` (fork PR #17) | Retained on upstream's rewritten embedder. Upstream made `EmbeddingGemma2Embedding.Input/Task/Modality` typealiases of the shared `EmbeddingInput/EmbeddingTask/EmbeddingModality` (`MultimodalEmbedding.swift`), so `.sharedVideo` is now a case of `EmbeddingInput.Part`. The sharing path validates every variant, encodes the video once, then runs upstream's `sequence(_:task:sharedVideo:)` and `unitRows` one variant at a time. Upstream's batched `embed(_:[Input])` rejects `.sharedVideo` with `misplacedSharedVideo`. |
| `generation_config.json` sampling (`temperature`, `topK`, `topP`) | Replaced by upstream's `GenerationConfigFile` (it adds `minP`, `repetitionPenalty` and `applyingSamplingDefaults`) and `ModelConfiguration.generationConfig`, which the factories set from the file unless the registry entry supplies one. The fork's public `ModelContext.generationConfig` stays as a get/set alias of `configuration.generationConfig`, so the catalog kit's `context.generationConfig?.temperature/topK/topP` reads still compile. The fork's `parsedGenerationConfig` factory edits are gone. |
| Gemma 4 assistant: backbone sliding window wider than the drafter's (`a986d9b`) | Retained (`Gemma4Assistant.swift`). This is the file's only difference from upstream. |
| TF32 test tolerance adaptations (Gemma3nSharedKV, VLMTextOnlyContinuation, Mellum, EmbeddingGemma2) | Superseded by upstream #802 (`expectClose` / `assertClose` with `reducedAtol`), which covers the same tests. The fork's hand-written `MatmulPrecision.tolerance` blocks and the `logitsAtol`/`attentionAtol` statics are removed. Both TF32 regimes still have to pass (Validation). |
| `CompiledTrace` plain closure and the `testPlainCompileFreezesCapturedWeights` hazard test | Upstream adopted the same plain-closure `CompiledTrace` and moved the capture hazard test into mlx-swift (`TransformTests.testCompileCapturesModuleParameterWithoutState`). Both files now equal upstream. |
| `UserInput.Audio` big-endian pin, `imagesThenVideosThenAudiosThenText` message layout | Retained. |
| Upstream `.github` changes (issue templates, ci-state scripts and workflows) | Deferred. The fork's `.github` is unchanged. |
| `IntegrationTesting/` | Takes upstream's files, plus the fork's `Gemma4UnifiedAudioIntegrationTests.swift`. The 2026-10-23 merge had kept stale pre-merge versions of the upstream integration tests, which no longer compiled against the library (for example a missing `.rejectedToolCall` case and synchronous `loadWeights`). The six upstream test files that merge had dropped are restored. |

## Conflicts by area

29 files conflicted (23 content, 6 modify/delete):

- CI and repo files: `.github/workflows/{integration_tests,pull_request}.yml` kept the fork's. `.gitignore` keeps both `.gstack/` and `.worktrees`.
- Integration tests: 8 content and 6 modify/delete conflicts under `IntegrationTesting/`, all resolved to upstream (see the table).
- Package: `Package.swift` keeps the sibling path.
- Generation config: `GenerationConfigFile.swift`, `LLMModelFactory.swift` and `VLMModelFactory.swift` take upstream's. `ModelFactory.swift` (auto-merged) has `ModelContext.generationConfig` rebuilt as an alias.
- Evaluate: `Evaluate.swift` is upstream's plus `stopStrings:` (doc reference, parameter, handler).
- MTP registry: `MTPDrafterModelFactory.swift` is the union of both lists.
- Gemma 4: `Gemma4.swift` keeps `getAudioFeatures` beside the new `multimodalInputs`. The audio arguments are added and the `prepare` guard includes audio.
- EmbeddingGemma 2: `EmbeddingGemma2.swift` is upstream's plus the sharing path, and `MultimodalEmbedding.swift` gains `.sharedVideo`. `Libraries/MLXEmbedders/README.md` keeps both sections.
- Compile: `CompiledTrace.swift` and `CompiledTraceTests.swift` are upstream's.
- Tests: `EmbeddingGemma2Tests`, `Gemma3nSharedKVTests`, `MellumTests` and `VLMTextOnlyContinuationTests` take upstream's #802 tolerance calls. Fork-only tests are kept.
- Semantic, without a conflict: `StopStringTests` handles the new `Generation.reasoning` case.

Gemma 4 early-return audit: in `Libraries/MLXVLM/Models/Gemma4*.swift`, both processors' fast
paths include audio (`if input.images.isEmpty, input.videos.isEmpty, input.audios.isEmpty` and
`if input.images.isEmpty, input.audios.isEmpty`). The E-series `prepare` and `hiddenStates`
checks include `input.audio?.features`, and the unified model's `prepare` and `hiddenStates`
already include `input.audio`. #813's `gemma4SplitImagePrompt` refuses to split any input
carrying audio, so the prompt prefills in full. MLXLMCommon's Gemma 4 code has no
image-only return.

## Public API changes for downstream (catalog kit)

Removed:
- `DFlash2DrafterModel`, `DFlash2DrafterModelFactory` / registry types, `DFlash2SpeculativeTokenIterator`, `DFlash2TargetModel` (and the `dflash2*` members on `Qwen35TextModel` / `Qwen35Model`), DFlash2 verify request/result types, `DFlash2Registration`.
- `generate(input:cache:parameters:context:dflash2Drafter:blockSize:components:wiredMemoryTicket:)` and the matching `generateTokens`.
- The kit used none of these.

Changed:
- `ModelContext.init(configuration:model:processor:tokenizer:generationConfig:)` loses `generationConfig:`. `ModelContext.generationConfig` is now computed over `configuration.generationConfig`, so a registry-supplied config wins over the file.
- `GenerationConfigFile` is `Equatable`. Its memberwise init is now `init(eosTokenIds:stopStrings:temperature:topP:topK:minP:repetitionPenalty:suppressTokens:)`.
- `EmbeddingGemma2Embedding.Modality/Input/Task` are typealiases of top-level `EmbeddingModality/EmbeddingInput/EmbeddingTask`, and the actor conforms to `MultimodalEmbedding`.
- `EmbeddingInput.Part` gains `.sharedVideo`, so exhaustive switches over it must handle it.
- `embed(_:[Input],task:dimensions:)` now batches text-only inputs (≤1,024 tokens per batch). `init` reads the tokenizer from the resolved `tokenizerDirectory`.
- `generateTask(...)` gains a trailing `reasoningPrimedInside:` argument.
- `Generation` gains `.reasoning(String)` (#563). With a resolved `ModelConfiguration.reasoningConfig` (Gemma 4 channels), thought text now arrives as `.reasoning` instead of `.chunk`. The kit's own reasoning split and `default: break` switches must be checked.
- Load functions gain `progress: LoadProgressHandlers` (#551); the trailing `progressHandler` overloads remain. `ChatSession` adds steering and interrupts (#608, #798).

## Validation

TODO (next phase): `swift build --build-tests -Xswiftc -strict-concurrency=complete`, then stage
`mlx.metallib` into every `*.xctest` bundle and run `swift test --skip-build --no-parallel` under
the default TF32 setting and again with `MLX_ENABLE_TF32=0`. Also run the Gemma 4 audio-only
suites (`Gemma4UnifiedAudioTests|Gemma4AudioFeatureStreamTests|Gemma4AudioTests|UserInputAudioTests`)
explicitly.

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
| Native multimodal EmbeddingGemma 2 encoder (fork PR #10) | Retired after the catalog kit moved to upstream's `EmbeddingGemma2Embedding` (kit branch `feat/embeddinggemma2-upstream-api`). `MLXVLM/Embeddings/*`, the container tests, the pinned fixture, its generator and the runtime benchmark are gone; only upstream's embedder remains. |
| `generation_config.json` sampling propagation | Retained alongside upstream's `suppress_tokens` (#412): `GenerationConfigFile` carries both. |
| Safetensors validation (fail-closed offsets, group arithmetic, payload bounds) | Retained inside upstream's component-aware loader (#679, #580). Upstream still parses headers with `try?` and falls through, so the fork's `safetensorSpansInFileOrder` stays and upstream's read-filter is applied after it. Upstream's loader-thread error propagation (#698) is in. |
| `generateTask(stopStrings:)` override, Markdown fence handling | Retained (no conflicts). |
| Tests | Both sides' tests kept. One fork negative test changed shape: a drafter given only bare target weights now throws `CheckpointComponent.SelectionError` instead of returning an empty dictionary, matching upstream's `sanitize` that now throws. |
| `@Sendable` stateless `compile` shadows (`CompileOverloads.swift`, top of `CompiledTrace.swift`) | Removed. Sibling mlx-swift main (#469) now provides the same 15 overloads, so the shadows made every unqualified `compile(shapeless:_:)` call ambiguous. `CompiledTrace` stores a plain `([MLXArray]) -> [MLXArray]`, because mlx-swift's stateful `compile(inputs:)` now returns a non-Sendable closure. The fork's `testPlainCompileFreezesCapturedWeights` reproduces the capture hazard through `MLX.compile(inputs: [])` now that the stateless overload rejects the capture at build time. Both factories pass the fork's `parsedGenerationConfig` to upstream's `mergeGenerationConfigSuppressedTokens`. This is the first sync against a sibling that is itself ahead of its release. |
| Upstream `.github` changes | None in this range. |

Public API changes inherited from upstream: the Qwen 3.5/3.8 `sanitize(weights:)` methods now throw, and `prepareCheckpoint(_:)` / `excludedCheckpointNamespaces` are new. Downstream callers of those `sanitize` methods need `try`.

## Validation

Xcode 27.0 / Swift 6.4. `swift build --build-tests`, then `mlx.metallib` compiled from the sibling's
`Source/Cmlx/mlx-generated/metal/*.metal` with MLX's kernel flags (`-Wall -Wextra -fno-fast-math
-Wno-c++17-extensions -Wno-c++20-extensions`) and copied into every `*.xctest` bundle, then
`swift test --skip-build --no-parallel` under the default TF32 setting and once more with
`MLX_ENABLE_TF32=0`. Both must pass.

TF32: this host (M5 class) runs float32 matmuls as TF32 by default. Upstream's
`Tests/MLXLMTests/MatmulPrecision.swift` (#649) measures that and adapts tolerances, and its policy
is not to pin `MLX_ENABLE_TF32=0`. Upstream CI runs on pre-M5 hardware, so eleven upstream tests
written after #649 kept fixed IEEE bounds and failed here with TF32 on (Gemma3nSharedKV,
VLMTextOnlyContinuation, Mellum, and the upstream EmbeddingGemma2 reference tests). They now use
`MatmulPrecision` with bounds measured on this host; the IEEE bounds are unchanged. The fork's
EmbeddingGemma 2 fixture was pinned from Python mlx-vlm with TF32 on, so its float32 pinned test
keeps 1e-6 under TF32 and uses 2e-3 under IEEE float32; regenerating the fixture under both
regimes would tighten it. Results are in the PR.

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
