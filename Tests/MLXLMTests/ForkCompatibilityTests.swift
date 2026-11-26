// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

/// Fork patches that SightRoll's catalog kit relies on and upstream does not provide.
struct ForkCompatibilityTests {

    /// The Gemma 4 E2B/E4B/12B assistant drafters are registered next to upstream's 26B/31B.
    @Test(arguments: [
        "mlx-community/gemma-4-E2B-it-qat-assistant-mxfp4",
        "mlx-community/gemma-4-E2B-it-assistant-bf16",
        "mlx-community/gemma-4-E4B-it-qat-assistant-mxfp4",
        "mlx-community/gemma-4-12B-it-assistant-4bit",
    ])
    func registersSmallGemma4AssistantDrafters(id: String) {
        #expect(MTPDrafterRegistry.shared.contains(id: id))
    }

    /// `ModelContext.generationConfig` reads and writes `configuration.generationConfig`, so a
    /// registry-supplied config and the model's `generation_config.json` reach callers the same way.
    @Test func modelContextGenerationConfigAliasesConfiguration() {
        var configuration = ModelConfiguration(id: "fork-generation-config")
        configuration.generationConfig = GenerationConfigFile(
            temperature: 0.7, topP: 0.9, topK: 40)
        let processor = TestInputProcessor()
        var context = ModelContext(
            configuration: configuration, model: StubLanguageModel(), processor: processor,
            tokenizer: processor.tokenizer)

        #expect(context.generationConfig?.temperature == 0.7)
        #expect(context.generationConfig?.topP == 0.9)
        #expect(context.generationConfig?.topK == 40)

        context.generationConfig = GenerationConfigFile(temperature: 0.2)
        #expect(context.configuration.generationConfig?.temperature == 0.2)
        #expect(context.configuration.generationConfig?.topK == nil)

        context.generationConfig = nil
        #expect(context.configuration.generationConfig == nil)
    }
}

private final class StubLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [] }

    func prepare(
        _ input: LMInput,
        cache _: [KVCache],
        state _: LMOutput.State?,
        prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache _: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, inputs.dim(-1), 1])
    }
}
