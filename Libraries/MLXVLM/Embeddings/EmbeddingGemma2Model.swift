import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct EmbeddingGemma2Prepared {
    let tokens: MLXArray
    let valid: MLXArray
    let images: [MLXArray]
    let videoFrames: [MLXArray]
    let audio: [(features: MLXArray, valid: MLXArray)]
}

/// Only the weight-loading protocol is shared with language models. This encoder
/// has no causal cache, generation interface, or chat conventions.
final class EmbeddingGemma2Model: Module, BaseLanguageModel {
    @ModuleInfo(key: "language_model") var text: EmbeddingGemma2Text
    @ModuleInfo(key: "vision_tower") var vision: Gemma4VisionModel?
    @ModuleInfo(key: "embed_vision") var visionProjection: Gemma4MultimodalEmbedder?
    @ModuleInfo(key: "audio_tower") var audio: Gemma4AudioModel?
    @ModuleInfo(key: "embed_audio") var audioProjection: Gemma4MultimodalEmbedder?
    let configuration: EmbeddingGemma2Configuration
    let towers: Set<EmbeddingGemma2Tower>

    init(_ c: EmbeddingGemma2Configuration, towers: Set<EmbeddingGemma2Tower>) throws {
        configuration = c
        self.towers = towers
        _text.wrappedValue = EmbeddingGemma2Text(c.text)
        if towers.contains(.vision) {
            guard let v = c.vision else { throw EmbeddingGemma2Error.missingTower }
            _vision.wrappedValue = Gemma4VisionModel(config: v)
            _visionProjection.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: v.hiddenSize, textHiddenSize: c.text.hiddenSize, eps: v.rmsNormEps)
        }
        if towers.contains(.audio) {
            guard let a = c.audio else { throw EmbeddingGemma2Error.missingTower }
            _audio.wrappedValue = Gemma4AudioModel(config: a)
            _audioProjection.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: a.outputProjectionDimensions ?? a.hiddenSize,
                textHiddenSize: c.text.hiddenSize, eps: a.rmsNormEps)
        }
        super.init()
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            // Selective loads intentionally omit every parameter belonging to an
            // unused tower. Strict Module.update still verifies all remaining keys.
            if !towers.contains(.vision),
                key.hasPrefix("vision_tower.") || key.hasPrefix("embed_vision.")
            {
                continue
            }
            if !towers.contains(.audio),
                key.hasPrefix("audio_tower.") || key.hasPrefix("embed_audio.")
            {
                continue
            }
            var value = value
            let convInputChannels =
                key.contains(".layer0.")
                ? 1 : (configuration.audio?.subsamplingConvChannels.first ?? 128)
            if key.contains("subsample_conv_projection"), key.hasSuffix("conv.weight"),
                value.ndim == 4, value.dim(-1) != convInputChannels
            {
                value = value.transposed(0, 2, 3, 1)
            }
            if key.contains("depthwise_conv1d"), key.hasSuffix("weight"),
                value.ndim == 3, value.dim(-1) != 1
            {
                value = value.transposed(0, 2, 1)
            }
            result[key] = value
        }
        return result
    }

    static func scatter(
        _ embeddings: MLXArray, tokens: MLXArray,
        mediaToken: Int, features: MLXArray?
    ) throws -> MLXArray {
        let mask = tokens .== mediaToken
        let expected = sum(mask.asType(.int32)).item(Int.self)
        let actual = features?.dim(0) ?? 0
        guard expected == actual else {
            throw EmbeddingGemma2Error.mediaTokenMismatch(expected: expected, actual: actual)
        }
        guard let features, actual > 0 else { return embeddings }
        let indices = maximum(cumsum(mask.flattened().asType(.int32)) - 1, 0)
        let aligned = take(features, indices, axis: 0).reshaped(embeddings.shape)
        return which(mask[.ellipsis, .newAxis], aligned.asType(embeddings.dtype), embeddings)
    }

    static func pool(_ hidden: MLXArray, valid: MLXArray) -> MLXArray {
        let mask = valid.asType(.float32)[.ellipsis, .newAxis]
        // The pinned pooling helper promotes the mask and sum to FP32 even
        // when token projections use BF16. Padding never contributes.
        let selected = which(
            valid.asType(.bool)[.ellipsis, .newAxis], hidden, MLXArray(0, dtype: hidden.dtype))
        let pooled = sum(selected * mask, axis: 1) / maximum(sum(mask, axis: 1), 1e-9)
        let fp32 = pooled.asType(.float32)
        return fp32 / maximum(sqrt(sum(fp32 * fp32, axis: -1, keepDims: true)), 1e-12)
    }

    func callAsFunction(_ input: EmbeddingGemma2Prepared, visionBatchSize: Int = 1) throws
        -> MLXArray
    {
        guard (1 ... 8).contains(visionBatchSize) else { throw EmbeddingGemma2Error.invalidInput }
        guard input.tokens.ndim == 2, input.tokens.shape == input.valid.shape,
            input.tokens.dim(0) > 0, input.tokens.dim(1) > 0,
            input.tokens.dim(1) <= 8192,
            all(sum(input.valid.asType(.int32), axis: 1) .> 0).item(Bool.self),
            all((input.tokens .>= 0) .&& (input.tokens .< configuration.text.vocabularySize)).item(
                Bool.self)
        else { throw EmbeddingGemma2Error.invalidInput }
        let mediaMask =
            (input.tokens .== configuration.imageToken)
            .|| (input.tokens .== configuration.videoToken)
            .|| (input.tokens .== configuration.audioToken)
        let textTokens = which(mediaMask, MLXArray(configuration.text.padToken), input.tokens)
        let unscaled = text.tokens(textTokens)
        var embeddings =
            unscaled * MLXArray(sqrt(Float(configuration.text.hiddenSize)), dtype: unscaled.dtype)
        for (pixels, token) in [
            (input.images, configuration.imageToken), (input.videoFrames, configuration.videoToken),
        ] {
            var features: [MLXArray] = []
            if !pixels.isEmpty {
                guard let vision, let visionProjection else {
                    throw EmbeddingGemma2Error.missingTower
                }
                // Only equal shapes share a tower batch. Restore original media
                // order before scatter; aspect ratios never force resampling.
                var start = 0
                while start < pixels.count {
                    var end = start + 1
                    while end < pixels.count, end - start < visionBatchSize,
                        pixels[end].shape == pixels[start].shape
                    { end += 1 }
                    let encoded = visionProjection(
                        vision(concatenated(Array(pixels[start ..< end]), axis: 0)))
                    for row in 0 ..< (end - start) { features.append(encoded[row]) }
                    start = end
                }
            }
            embeddings = try Self.scatter(
                embeddings, tokens: input.tokens, mediaToken: token,
                features: features.isEmpty ? nil : concatenated(features, axis: 0))
        }
        var features: [MLXArray] = []
        if !input.audio.isEmpty {
            guard let audio, let audioProjection else { throw EmbeddingGemma2Error.missingTower }
            for item in input.audio {
                let (encoded, invalid) = audio(item.features, audioMelMask: .!item.valid)
                let count = sum((.!invalid).asType(.int32)).item(Int.self)
                features.append(audioProjection(encoded)[0, ..<count])
            }
        }
        embeddings = try Self.scatter(
            embeddings, tokens: input.tokens, mediaToken: configuration.audioToken,
            features: features.isEmpty ? nil : concatenated(features, axis: 0))
        return Self.pool(text(embeddings, valid: input.valid), valid: input.valid)
    }
}
