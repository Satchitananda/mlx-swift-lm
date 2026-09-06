// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Registry of `model_type` strings (e.g. `"gemma4_assistant"`) to creator
/// closures that instantiate ``MTPDrafterModel`` instances from `config.json`
/// data.
///
/// Empty at bootstrap because drafter implementations live in MLXVLM —
/// importing them here would create a circular dependency. Downstream
/// modules call ``ModelTypeRegistry/registerModelType(_:creator:)`` on the
/// shared registry at app/import time. See `Gemma4AssistantRegistration`
/// in MLXVLM.
///
/// Note: ``ModelTypeRegistry`` is an `actor`; registration is `async`.
public enum MTPDrafterTypeRegistry {
    /// Shared registry. Empty until a downstream module registers a drafter
    /// type via `await MTPDrafterTypeRegistry.shared.registerModelType(...)`.
    public static let shared: ModelTypeRegistry<any MTPDrafterModel> = .init()

    /// Registry for drafters whose forward pass consumes multimodal position
    /// state. It is separate from ``shared`` because standalone Qwen MTP
    /// checkpoints intentionally ship an empty `vision_config`, so their
    /// target architecture cannot be inferred from drafter metadata alone.
    public static let visionLanguage: ModelTypeRegistry<any MTPDrafterModel> = .init()
}

/// Registry of model id (e.g. `"mlx-community/gemma-4-31B-it-assistant-bf16"`)
/// to ``ModelConfiguration``. Drafters don't have prompts, so the
/// configurations omit `defaultPrompt`.
public class MTPDrafterRegistry: AbstractModelRegistry, @unchecked Sendable {
    public static let shared = MTPDrafterRegistry(modelConfigurations: all())

    // E2B (mxfp4 QAT — use_ordered_embeddings=true, sparse MaskedEmbedder LM head)
    public static let gemma4_E2B_qat_assistant_mxfp4 = ModelConfiguration(
        id: "mlx-community/gemma-4-E2B-it-qat-assistant-mxfp4"
    )
    // E2B (bf16 — use_ordered_embeddings=false, dense asLinear LM head)
    public static let gemma4_E2B_assistant_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-4-E2B-it-assistant-bf16"
    )
    // E4B
    public static let gemma4_E4B_qat_assistant_mxfp4 = ModelConfiguration(
        id: "mlx-community/gemma-4-E4B-it-qat-assistant-mxfp4"
    )
    // 12B
    public static let gemma4_12B_assistant_4bit = ModelConfiguration(
        id: "mlx-community/gemma-4-12B-it-assistant-4bit"
    )
    // 26B / 31B (large model experiments)
    public static let gemma4_26B_assistant_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-4-26B-A4B-it-assistant-bf16"
    )
    public static let gemma4_31B_assistant_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-4-31B-it-assistant-bf16"
    )
    public static let qwen3_8_27b_mtp_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3.8-27B-MTP-4bit"
    )

    private static func all() -> [ModelConfiguration] {
        [
            gemma4_E2B_qat_assistant_mxfp4,
            gemma4_E2B_assistant_bf16,
            gemma4_E4B_qat_assistant_mxfp4,
            gemma4_12B_assistant_4bit,
            gemma4_26B_assistant_bf16,
            gemma4_31B_assistant_bf16,
            qwen3_8_27b_mtp_4bit,
        ]
    }
}

/// Loader for ``MTPDrafterModel`` checkpoints. Mirrors `LLMModelFactory`
/// in shape, but produces an ``MTPDrafterContext`` (no tokenizer, no user
/// input processor, no message generator — drafters borrow their target's
/// tokenizer).
public final class MTPDrafterModelFactory: GenericModelFactory {
    public typealias ContextType = MTPDrafterContext
    public typealias ContainerType = MTPDrafterContainer

    public static let shared = MTPDrafterModelFactory(
        typeRegistry: MTPDrafterTypeRegistry.shared,
        modelRegistry: MTPDrafterRegistry.shared
    )

    /// Loader for multimodal drafters. Use this factory when the verifier was
    /// loaded by `VLMModelFactory` and can emit multimodal RoPE deltas.
    public static let visionLanguage = MTPDrafterModelFactory(
        typeRegistry: MTPDrafterTypeRegistry.visionLanguage,
        modelRegistry: MTPDrafterRegistry.shared
    )

    public let typeRegistry: ModelTypeRegistry<any MTPDrafterModel>
    public let modelRegistry: AbstractModelRegistry

    public init(
        typeRegistry: ModelTypeRegistry<any MTPDrafterModel>,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> MTPDrafterContext {
        let modelDirectory = configuration.modelDirectory
        let configurationURL = modelDirectory.appending(component: "config.json")
        let configData: Data
        do {
            configData = try Data(contentsOf: configurationURL)
        } catch {
            throw ModelFactoryError.configurationFileError(
                configurationURL.lastPathComponent, configuration.name, error)
        }
        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let model: any MTPDrafterModel
        do {
            model = try await typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        try await loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization,
            weightFileSelection: configuration.weightFileSelection
        )

        let modelConfig = ModelConfiguration(
            directory: modelDirectory,
            tokenizerSource: nil,
            defaultPrompt: ""
        )
        return MTPDrafterContext(configuration: modelConfig, model: model)
    }

    public func _wrap(_ context: MTPDrafterContext) -> MTPDrafterContainer {
        .init(context: context)
    }
}
