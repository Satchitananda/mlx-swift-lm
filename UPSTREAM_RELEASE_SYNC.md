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
