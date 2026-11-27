# Upstream release sync — 2026-11-26

Owner decision 2026-11-26: the fork tracks the latest upstream releases, not `main`.

| Item | Pin |
| --- | --- |
| mlx-swift-lm | release [`3.33.0`](https://github.com/ml-explore/mlx-swift-lm/releases/tag/3.33.0), `ca661dc` ("Size five float32 test comparisons for TF32 matmuls (#802)"). Of the six later commits on upstream `main`, #716 and #813 are carried (below); #772, #812, #809 and #817 are not included. |
| Sibling mlx-swift | `sync/upstream-release-20261126`, `e31ba6d9`: release 0.33.0 plus core v0.32.5 and its fork patches. `.package(path: "../mlx-swift")` resolves to that worktree (`.build/workspace-state.json`). |

This replaces the unmerged main-based sync (`sync/upstream-main-20261125`, PR #18, recorded on
that branch). Branch history:

- `c728e8c` drops DFlash2 on fork `main` (decision D3; same tree as the main-based `a060cbd`).
- `64c6dfa` merges `3.33.0`. The 29 conflicts are the same files as on 2026-11-25. They are
  resolved from that merge (`d22adec`) with the diff of the six post-release commits reversed;
  the reverse diff applied cleanly. The fork's `.github` is kept.
- The fork commits after that merge are re-applied without their sync-doc edits: test
  adaptations, multi-stop `GrammarTokenizer`, `JSONSchemaFormat`, XGrammar 0.2.3.
- `git cherry-pick -x` of #716 and #813. Both apply cleanly onto the fork tree.
- After the picks, the tree equals the main-based branch tip `0d343f8` minus #772, #812, #809
  and #817 (checked by diff). Then come the new commits: the covering tests, and the drops of
  the assistant mask and the GatedDelta bound.

## Fork patch disposition

Each retained patch was taken out (file restored to `3.33.0`, or its behaviour switched off with
a one-line edit when removing it would only stop the tests compiling). Its covering tests were
run, and the patch was restored. Default TF32 unless stated.

| Patch | Disposition | Evidence |
| --- | --- | --- |
| Local sibling MLX dependency | Retained. | `workspace-state.json` resolves `mlx-swift` to the release-sync worktree (`fileSystem`). |
| DFlash2 drafter runtime, projection stacking | **Dropped** (D3). | `c728e8c`; nothing in SightRoll used it. |
| `Linux/Logger.swift` shims | Dropped: upstream #702 (`MLXLogger`) is in 3.33.0. | `Linux/` equals 3.33.0. |
| TF32 tolerance adaptations (Gemma3nSharedKV, VLMTextOnlyContinuation, Mellum, EmbeddingGemma2) | Dropped: upstream #802 is the 3.33.0 commit itself. | Those tests equal 3.33.0 (EmbeddingGemma2Tests differs only by the fork's sharing tests) and pass in both regimes. |
| `CompiledTrace` plain closure and hazard test | Dropped: upstream has the same code. | `CompiledTrace.swift` and `CompiledTraceTests.swift` equal 3.33.0. |
| `generation_config.json` sampling | Replaced by upstream's `GenerationConfigFile` / `ModelConfiguration.generationConfig` (in 3.33.0). The fork keeps only the `ModelContext.generationConfig` get/set alias, which the catalog kit reads. | `GenerationConfigFile.swift` and both factories equal 3.33.0. New `ForkCompatibilityTests.modelContextGenerationConfigAliasesConfiguration`: with 3.33.0's `ModelFactory.swift` it does not compile (`value of type 'ModelContext' has no member 'generationConfig'`). |
| MTP drafter registry: Gemma 4 E2B/E4B/12B assistants | Retained. | New `ForkCompatibilityTests.registersSmallGemma4AssistantDrafters`: with 3.33.0's `MTPDrafterModelFactory.swift`, all 4 ids are missing (4 issues). |
| Gemma 4 assistant: backbone sliding window wider than the drafter's (`a986d9b`) | **Dropped.** Upstream #581 (in 3.33.0) makes `validateCompatibility` refuse a target whose effective sliding window exceeds the drafter's. `MTPSpeculativeTokenIterator` and `ChatSession` validate before drafting. So the fork's branch for a sliding KV wider than the drafter window could not run. The 2026-11-25 sync kept it without noticing this. A wider-window pairing is now refused with `MTPDrafterCompatibilityError.incompatibleTarget`, as upstream intends. | Upstream `rejectsLargerTargetWindow` / `usesEffectiveTargetWindowFallback` pass. With `Gemma4Assistant.swift` at 3.33.0, the `Gemma4Assistant*`, `MTPDrafter*`, `MTPSpeculative*`, `Gemma4EmitDrafterState` and `ChatSessionMTP*` suites (85 tests) pass. |
| Gemma 4 audio: E-series audio tower | Retained. | With the tower not built for an `audio_config`: `UnreadCheckpointTowersTests.gemma4VLMWithAudioConfigReadsAudio` fails (2 issues). |
| Gemma 4 audio: processor fast paths include audio | Retained. | With `input.audios.isEmpty` taken out of both processors' fast paths, 7 audio tests fail with 15 issues, among them "Processor output feeds a unified model with an audio_config end to end", "Placeholders that do not pair with clips fail typed" and "Several clips batch right-padded to the longest". |
| `imagesThenVideosThenAudiosThenText` message layout | Retained. | With the audio parts not emitted: "Gemma message generation preserves image labels, video, and audio" fails, along with 6 processor and unified-audio tests (9 issues). |
| `UserInput.Audio` big-endian pin | Retained. | With 3.33.0's `UserInput+Audio.swift`: "A big-endian AIFF decodes to the same samples as a little-endian WAV" fails (2 issues). |
| Safetensors fail-closed validation | Retained. | With 3.33.0's `Load.swift`, three `LoadWeightsTests` crash the test process (signal 5), each found by skipping the one before: `testLoadGroupsRejectOverflowAndBalanceLargeValidTotals`, `testPublicWeightLoaderRejectsOverflowingHeader` and `testSafetensorSpansRejectsInvalidOffsets`. |
| `generateTask(stopStrings:)` override | Retained. | With the override ignored: `StopStringTests.testGenerateTaskExplicitEmptyStopStringsKeepsStopTextInsideJSONString` fails (3 assertions). |
| Markdown fence LF/CRLF/CR tool-call recovery | Retained. | With 3.33.0's `TextToolCallRecoveryScanner.swift`: "A Markdown closing fence needs a complete whitespace-only line" fails for every fence character, line ending and native/alternate combination. |
| `embed(sharingVideo:variants:task:dimensions:)` | Retained. | With the shared video dropped from each variant: `sharedVideoVariantsMatchIndependentEmbeddings` fails (2 issues). With its placeholder check removed: `sharedVideoPlaceholderIsValidatedBeforeDecoding` fails. |
| MLXGuidedGeneration multi-stop `GrammarTokenizer` | Retained; upstreamable (branch `guided/multi-stop-grammar-tokenizer`). | With only the first stop id registered and the `prematureEOS` guard off, 4 `MultiStopGrammarTokenizerTests` fail (7 issues): "every registered stop token is masked until the grammar can terminate", "a fresh matcher keeps the full stop set", "the loop cannot stop on a registered extra stop before the grammar completes" and "an unregistered stop inside the value is premature, not success". |
| MLXGuidedGeneration `JSONSchemaFormat` | Retained; upstreamable (branch `guided/json-schema-format-options`). | With the format ignored in the shim and the bridge, 6 tests fail (13 issues) across `JSONSchemaFormatTests` and `JSONSchemaFormatShimTests`. |
| Vendored XGrammar 0.2.3 | Retained; upstreamable (branch `guided/xgrammar-0.2.3`). | With `Libraries/MLXCXGrammar` and `Package.swift` at the 0.1.30 vendor (`7431beb`), 16 issues: `JSONSchemaNumberRangeTests` "minimum 0 accepts fractions below 1" (6) and "a bounded maximum rejects fractions past it" (4); five `ConstraintCachingTests` clone cases; `VersionTests` (`v0.1.30`). |
| Test adaptation: `StopStringTests` handles `.reasoning` / `.steering` | Retained (compile adaptation only). | The suite's exhaustive switch needs both cases. |
| Test adaptation: `UnreadCheckpointTowersTests.gemma4VLMWithAudioConfigReadsAudio` | Retained. | It is the audio-tower evidence above. |
| Test adaptation: GatedDelta near-one decay bound | **Dropped**; `GatedDeltaTests.swift` equals 3.33.0. With core v0.32.5 the chunked NAX kernel's drift no longer appears under `MLX_ENABLE_TF32=0`. | Measured drift: 1.436e-3 with TF32 on (upstream bound 1e-2), 9.2e-7 with `MLX_ENABLE_TF32=0` (bound 1e-3). With upstream's bound and `MLX_ENABLE_TF32=0`, `GatedDeltaTests` passes 8 of 8. |
| Fork-only Qwen 3.5 MTP loading tests (`Qwen35MTPTests` additions) | Retained (tests only; they cover upstream code). | Pass. |
| `IntegrationTesting/` | Upstream's files plus the fork's `Gemma4UnifiedAudioIntegrationTests.swift`, and #813's new integration test. | Not run (needs model downloads). |
| CI/workflow delta | Deferred, as before. | `.github` equals fork `main`. |

## Carried upstream patches

| Patch | Commit on this branch | Drop when |
| --- | --- | --- |
| #716 `4daac23`: restore a prompt checkpoint when the cache cannot trim back | `47afe1f` (`cherry-pick -x`) | a release contains it |
| #813 `6759555`: Gemma 4 bidirectional image attention on a warm KV cache | `31e3276` (`cherry-pick -x`) | a release contains it |

#813 also brings `gemma4SplitImagePrompt`, which refuses to split input that carries audio, so the
2026-11-25 early-return audit still holds. Its tests (`Gemma4WarmImagePrefillTests`,
`ChatSessionCheckpointTests`, `KVCacheCheckpointTests`, `PromptCheckpointNumericsTests`) pass.

## Public API changes for downstream (catalog kit)

They match the 2026-11-25 list (on PR #18), with these exceptions:

- No #772, #812, #809 or #817 behaviour: Gemma 2 batching masks, Qwen tool-argument
  rejections, DiffusionGemma prompt chunking, Idefics3 prepare validation.
- A Gemma 4 assistant paired with a backbone whose sliding window is wider than its own is
  refused with `MTPDrafterCompatibilityError.incompatibleTarget` instead of drafting.

## Validation

Xcode 27.1 / Swift 6.4, M5 Max. `swift build --build-system native --build-tests
-Xswiftc -strict-concurrency=complete` passes. No warnings in fork-owned code. The remaining
warnings are upstream test deprecations and unused results, plus two in fork-touched test files
on upstream-authored lines. A fresh `mlx.metallib` was compiled from the sibling's 14
`Source/Cmlx/mlx-generated/metal/**/*.metal` files with MLX's kernel flags. It was staged beside
the test binary and in the `.xctest` bundle's `Contents/MacOS` and `Contents/Resources`, and
cleared before each build. Then `swift test --build-system native --skip-build --no-parallel`:

| Regime | XCTest | Swift Testing |
| --- | --- | --- |
| default (TF32 on) | 1,081 executions, 9 skipped, 0 failures | 1,893 tests in 185 suites passed (2 known issues) |
| `MLX_ENABLE_TF32=0` | 1,081 executions, 9 skipped, 0 failures | 1,893 tests in 185 suites passed (2 known issues) |

The native build system links every test target into one `mlx-swift-lmPackageTests.xctest`, so
the totals are per framework, not per bundle. The 2 known issues are upstream's own
`withKnownIssue` in the MiniCPM5 CDATA tool-call test. The skips are checkpoint- and
environment-gated tests. Counts are below the main-based branch's because 3.33.0 lacks #772,
#812, #809 and #817 and their tests.

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
