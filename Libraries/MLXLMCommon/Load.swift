// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let (w, m) = try loadArraysAndMetadata(url: url)
            for (key, value) in w {
                weights[key] = value
            }
            if metadata.isEmpty {
                metadata = m
            }
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(
            model: model,
            filter: { path, module in
                if weights["\(path).scales"] != nil {
                    if let perLayerQuantization {
                        return perLayerQuantization.quantization(layer: path)?.asTuple
                    } else {
                        return quantization?.asTuple
                    }
                } else {
                    return nil
                }
            },
            apply: { layer, groupSize, bits, mode in
                // mxfp8 matmul kernel requires biases == nil (zero-points don't exist
                // in this format). QuantizedLinear's default init always produces non-nil
                // biases via affine quantization, which crashes at inference.
                // Use the explicit initializer with biases:nil; weight/scales are
                // overwritten by model.update(parameters:) with the real checkpoint data.
                if mode == .mxfp8, let linear = layer as? Linear {
                    let w = linear.weight
                    let dummyScales = MLXArray.zeros([w.dim(0), max(1, w.dim(1) / groupSize)])
                    return QuantizedLinear(
                        weight: w, bias: linear.bias,
                        scales: dummyScales, biases: nil,
                        groupSize: groupSize, bits: bits, mode: mode)
                }
                return quantizeSingle(layer: layer, groupSize: groupSize, bits: bits, mode: mode)
            }
        )
    }

    // apply the loaded weights
    //
    // Use .noUnusedKeys only (not .all) because pre-quantized mxfp8 models have a
    // different weight layout than the affine QuantizedLinear created by quantize(model:)
    // above (shapes are reconciled via _updateInternal once the real data is applied).
    // .noUnusedKeys still catches any unexpected extra keys in the checkpoint.
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    eval(model)
}
