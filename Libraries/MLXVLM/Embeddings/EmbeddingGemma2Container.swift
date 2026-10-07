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
        try await embedBatch([parts], maximumBatchSize: 1, visionBatchSize: 1)[0]
    }

    /// Independent samples, not interleaved files. One container and one
    /// exclusive model operation own all arrays. The padded token cap splits
    /// oversized batches without truncating any sample or modality. Different
    /// token lengths use separate batches to bound padding work and preserve
    /// the individual request's attention shape.
    public func embedBatch(
        _ inputs: [[EmbeddingGemma2Part]], maximumBatchSize: Int = 2,
        maximumPaddedTokens: Int = 8192, visionBatchSize: Int = 2
    ) async throws -> [Output] {
        guard !inputs.isEmpty, inputs.count <= 4, (1 ... 4).contains(maximumBatchSize),
            (8192 ... 32768).contains(maximumPaddedTokens), (1 ... 8).contains(visionBatchSize)
        else { throw EmbeddingGemma2Error.invalidInput }
        try Task.checkCancellation()
        return try await context.read { context in
            try Task.checkCancellation()
            let rows = try inputs.map { try context.processor.prepare($0) }
            var outputs: [Output] = []
            var start = 0
            while start < rows.count {
                try Task.checkCancellation()
                var end = start + 1
                var length = rows[start].tokens.dim(1)
                while end < rows.count, end - start < maximumBatchSize {
                    if rows[end].tokens.dim(1) != length { break }
                    let nextLength = max(length, rows[end].tokens.dim(1))
                    if nextLength * (end - start + 1) > maximumPaddedTokens { break }
                    length = nextLength
                    end += 1
                }
                let prepared = try EmbeddingGemma2Processor.batch(
                    Array(rows[start ..< end]),
                    padToken: context.model.configuration.text.padToken)
                let vectors = try context.model(prepared, visionBatchSize: visionBatchSize)
                eval(vectors)
                try Task.checkCancellation()
                for row in start ..< end {
                    let values = vectors[row - start].asArray(Float.self)
                    let norm = values.reduce(0.0) { $0 + Double($1) * Double($1) }
                    guard values.count == context.model.configuration.text.embeddingDimensions,
                        values.allSatisfy(\.isFinite), norm.isFinite, abs(norm - 1) <= 0.002
                    else { throw EmbeddingGemma2Error.invalidOutput }
                    outputs.append(Output(vector: values, expandedTokens: rows[row].tokens.dim(1)))
                }
                start = end
            }
            return outputs
        }
    }
}
