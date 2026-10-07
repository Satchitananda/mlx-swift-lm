import Foundation
import MLX
import MLXNN

/// FP32 angles followed by activation-dtype trig rounding, as in the pinned reference.
final class EmbeddingGemma2Rotary {
    private let frequencies: MLXArray
    init(dimensions: Int, theta: Float) {
        frequencies =
            1 / pow(theta, MLXArray.arange(0, dimensions, step: 2, dtype: .float32) / dimensions)
    }
    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let angles = positions.asType(.float32)[.ellipsis, .newAxis] * frequencies
        let both = expandedDimensions(concatenated([angles, angles], axis: -1), axis: 1)
        let half = x.dim(-1) / 2
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return x * cos(both).asType(x.dtype) + rotated * sin(both).asType(x.dtype)
    }
}

private final class EmbeddingGemma2Attention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNormNoScale
    private let heads: Int
    private let kvHeads: Int
    private let dimension: Int
    private let rotary: EmbeddingGemma2Rotary

    init(_ c: EmbeddingGemma2Configuration.Text, layer: Int) {
        let override = c.layerOverrides[String(format: "%02d", layer)]
        heads = override?.attentionHeads ?? c.attentionHeads
        kvHeads = override?.keyValueHeads ?? c.keyValueHeads
        dimension = override?.headDim ?? c.headDim
        rotary = EmbeddingGemma2Rotary(
            dimensions: dimension, theta: c.rope[c.layerTypes[layer]]!.theta)
        _q.wrappedValue = Linear(c.hiddenSize, heads * dimension, bias: c.attentionBias)
        _k.wrappedValue = Linear(c.hiddenSize, kvHeads * dimension, bias: c.attentionBias)
        _v.wrappedValue = Linear(c.hiddenSize, kvHeads * dimension, bias: c.attentionBias)
        _output.wrappedValue = Linear(heads * dimension, c.hiddenSize, bias: c.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: dimension, eps: c.eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: dimension, eps: c.eps)
        _vNorm.wrappedValue = Gemma4RMSNormNoScale(eps: c.eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, positions: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let l = x.dim(1)
        let q = rotary(
            qNorm(q(x).reshaped(b, l, heads, dimension)).transposed(0, 2, 1, 3),
            positions: positions)
        let k = rotary(
            kNorm(k(x).reshaped(b, l, kvHeads, dimension)).transposed(0, 2, 1, 3),
            positions: positions)
        let v = vNorm(v(x).reshaped(b, l, kvHeads, dimension)).transposed(0, 2, 1, 3)
        let h = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1, mask: .array(mask))
        return output(h.transposed(0, 2, 1, 3).reshaped(b, l, heads * dimension))
    }
}

private final class EmbeddingGemma2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: EmbeddingGemma2Configuration.Text) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(geluApproximate(gate(x)) * up(x)) }
}

private final class EmbeddingGemma2PerLayerInputs: Module, UnaryLayer {
    @ModuleInfo(key: "per_layer_model_projection") var projection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var norm: RMSNorm
    private let layers: Int, width: Int
    private let scale: Float
    init(_ c: EmbeddingGemma2Configuration.Text) {
        layers = c.layers
        width = c.perLayerWidth
        scale = pow(Float(c.hiddenSize), -0.5)
        _projection.wrappedValue = Linear(c.hiddenSize, layers * width, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: width, eps: c.eps)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm((projection(x) * scale).reshaped(x.dim(0), x.dim(1), layers, width))
    }
}

private final class EmbeddingGemma2PerLayerBlock: Module {
    @ModuleInfo(key: "per_layer_input_gate") var gate: Linear
    @ModuleInfo(key: "per_layer_projection") var projection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var norm: RMSNorm
    init(_ c: EmbeddingGemma2Configuration.Text) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.perLayerWidth, bias: false)
        _projection.wrappedValue = Linear(c.perLayerWidth, c.hiddenSize, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.eps)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, input: MLXArray) -> MLXArray {
        x + norm(projection(geluApproximate(gate(x)) * input))
    }
}

private final class EmbeddingGemma2Layer: Module {
    @ModuleInfo(key: "self_attn") var attention: EmbeddingGemma2Attention
    @ModuleInfo(key: "mlp") var mlp: EmbeddingGemma2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var feedforwardNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var outputNorm: RMSNorm
    @ModuleInfo(key: "ple_block") var perLayer: EmbeddingGemma2PerLayerBlock?
    @ParameterInfo(key: "layer_scalar") var scalar: MLXArray
    init(_ c: EmbeddingGemma2Configuration.Text, layer: Int) {
        _attention.wrappedValue = EmbeddingGemma2Attention(c, layer: layer)
        _mlp.wrappedValue = EmbeddingGemma2MLP(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.eps)
        _attentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.eps)
        _feedforwardNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.eps)
        _outputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.eps)
        _perLayer.wrappedValue = c.perLayerWidth > 0 ? EmbeddingGemma2PerLayerBlock(c) : nil
        _scalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray, positions: MLXArray, input: MLXArray?)
        -> MLXArray
    {
        var h = x + attentionNorm(attention(inputNorm(x), mask: mask, positions: positions))
        h = h + outputNorm(mlp(feedforwardNorm(h)))
        if let perLayer, let input { h = perLayer(h, input: input) }
        return h * scalar
    }
}

final class EmbeddingGemma2Text: Module {
    @ModuleInfo(key: "embed_tokens") var tokens: Embedding
    @ModuleInfo(key: "ple") private var perLayer: EmbeddingGemma2PerLayerInputs?
    @ModuleInfo(key: "layers") private var layers: [EmbeddingGemma2Layer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "embedding_projection") var projection: Linear
    let configuration: EmbeddingGemma2Configuration.Text
    init(_ c: EmbeddingGemma2Configuration.Text) {
        configuration = c
        _tokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _perLayer.wrappedValue = c.perLayerWidth > 0 ? EmbeddingGemma2PerLayerInputs(c) : nil
        _layers.wrappedValue = (0 ..< c.layers).map { EmbeddingGemma2Layer(c, layer: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.eps)
        _projection.wrappedValue = Linear(c.hiddenSize, c.embeddingDimensions, bias: false)
        super.init()
    }

    static func masks(_ valid: MLXArray, window: Int) -> (full: MLXArray, local: MLXArray) {
        let length = valid.dim(1)
        let full = valid.asType(.bool).reshaped(valid.dim(0), 1, 1, length)
        let positions = MLXArray.arange(length)
        return (
            full,
            full .&& (abs(positions.reshaped(length, 1) - positions.reshaped(1, length)) .<= window)
        )
    }

    func callAsFunction(_ embeddings: MLXArray, valid: MLXArray, positions: MLXArray? = nil)
        -> MLXArray
    {
        let masks = Self.masks(valid, window: configuration.slidingWindow)
        let positions = positions ?? MLXArray.arange(embeddings.dim(1))[.newAxis]
        let inputs = perLayer?(embeddings)
        var h = embeddings
        for (i, layer) in layers.enumerated() {
            h = layer(
                h,
                mask: configuration.layerTypes[i] == "sliding_attention" ? masks.local : masks.full,
                positions: positions, input: inputs?[0..., 0..., i])
        }
        return projection(norm(h))
    }
}
