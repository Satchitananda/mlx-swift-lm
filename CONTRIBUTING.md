# Contributing to MLX Swift LM

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

1. Fork and submit pull requests to the repo. 
2. If you've added code that should be tested, add tests.
3. Every PR should have passing tests (if any) and at least one review. 
4. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   If needed you may need to `brew install swift-format`.
 
   You can also run the formatters manually as follows:
 
     ```
     swift-format format --in-place --recursive Libraries Tools Applications IntegrationTesting
     ```
 
   or run `pre-commit run --all-files` to check all files in the repo.
 
## Running Tests

Unit tests do not download models; tests evaluating MLX arrays need Metal.
For this fork's verified SwiftPM setup, including the matching Metal library
and serial full-precision numerical gate, see [Qwen runtime validation](QWEN38_INTEGRATION.md#validation).
Xcode also assembles the test bundle:

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -skipPackagePluginValidation
```

Integration tests verify end-to-end model loading and generation. They require
macOS with Metal and download models from Hugging Face Hub on first run. These
tests do not run in CI.

Open `IntegrationTesting/IntegrationTesting.xcodeproj` in Xcode and run the
test target (`Cmd+U` or via the Test Navigator), or use `xcodebuild`:

```bash
# Run all integration tests
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation

# Run a single test
xcodebuild test \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:IntegrationTestingTests/ToolCallIntegrationTests/qwen35FormatAutoDetection\(\)
```

See [Libraries/IntegrationTestHelpers/README.md](Libraries/IntegrationTestHelpers/README.md) for more details.

The upstream CI verifies DocC documentation for library targets. Its workflow
jobs are restricted to `ml-explore/mlx-swift-lm`, so they are skipped in this
fork regardless of PR draft status. Run the documentation check locally with:

```bash
scripts/verify-docs.sh
```

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX Swift LM, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
