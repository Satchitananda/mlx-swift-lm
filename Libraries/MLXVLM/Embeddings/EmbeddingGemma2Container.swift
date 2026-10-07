import Foundation
import MLX
import MLXLMCommon

/// A dedicated serialized embedding runtime. All arrays, modules, and feature
/// extractors remain under exclusive access until evaluated primitive values return.
public final class EmbeddingGemma2Container: Sendable {
    private struct Context {
        let model: EmbeddingGemma2Model
        let processor: EmbeddingGemma2Processor
    }
    private let context: SerialAccessContainer<Context>
    public let dimensions: Int
    public let loadedTowers: Set<EmbeddingGemma2Tower>
    public enum Precision: Sendable { case bfloat16, float32 }
    public let precision: Precision

    private init(model: consuming EmbeddingGemma2Model, tokenizer: any Tokenizer) {
        dimensions = model.configuration.text.embeddingDimensions
        loadedTowers = model.towers
        precision = model.text.tokens.weight.dtype == .bfloat16 ? .bfloat16 : .float32
        let processor = EmbeddingGemma2Processor(
            configuration: model.configuration, tokenizer: tokenizer)
        context = SerialAccessContainer(Context(model: model, processor: processor))
    }

    /// Loads an explicit local checkpoint. BF16/FP32 only; affine and FP16
    /// checkpoints fail closed. No implicit downloads, global cache, or LLM factory.
    public static func load(
        directory: URL, tokenizerLoader: any TokenizerLoader,
        towers: Set<EmbeddingGemma2Tower> = [.vision, .audio]
    ) async throws -> EmbeddingGemma2Container {
        let tokenizer = try await tokenizerLoader.load(from: directory)
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(
                    with: Result {
                        let data = try Data(contentsOf: directory.appending(path: "config.json"))
                        let dictionary =
                            try JSONSerialization.jsonObject(with: data) as? [String: Any]
                        guard dictionary?["quantization"] == nil,
                            dictionary?["quantization_config"] == nil
                        else {
                            throw EmbeddingGemma2Error.unsupportedPrecision
                        }
                        let c = try JSONDecoder().decode(
                            EmbeddingGemma2Configuration.self, from: data)
                        let required = c.controlTokenIDs
                        guard
                            required.allSatisfy({ id in
                                guard let token = tokenizer.convertIdToToken(id) else {
                                    return false
                                }
                                return tokenizer.convertTokenToId(token) == id
                            })
                        else { throw EmbeddingGemma2Error.invalidConfiguration }
                        let model = try EmbeddingGemma2Model(c, towers: towers)
                        try loadWeights(modelDirectory: directory, model: model)
                        guard
                            model.parameters().flattened().allSatisfy({
                                $0.1.dtype == .bfloat16 || $0.1.dtype == .float32
                            })
                        else {
                            throw EmbeddingGemma2Error.unsupportedPrecision
                        }
                        return EmbeddingGemma2Container(model: model, tokenizer: tokenizer)
                    })
            }
        }
    }

    public struct Output: Sendable {
        public let vector: [Float]
        public let expandedTokens: Int
    }

    public func embed(_ parts: [EmbeddingGemma2Part]) async throws -> Output {
        try Task.checkCancellation()
        return try await context.read { context in
            try Task.checkCancellation()
            let prepared = try context.processor.prepare(parts)
            let vector = try context.model(prepared)
            eval(vector)
            try Task.checkCancellation()
            let values = vector[0].asArray(Float.self)
            let norm = values.reduce(0.0) { $0 + Double($1) * Double($1) }
            guard values.count == context.model.configuration.text.embeddingDimensions,
                values.allSatisfy(\.isFinite), norm.isFinite, abs(norm - 1) <= 0.002
            else { throw EmbeddingGemma2Error.invalidOutput }
            return Output(vector: values, expandedTokens: prepared.tokens.dim(1))
        }
    }
}
