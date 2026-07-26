// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

package func safetensorWeightURLs(in modelDirectory: URL) throws -> [URL] {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    if FileManager.default.fileExists(atPath: indexURL.path) {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        return Set(index.weightMap.values)
            .sorted()
            .map { modelDirectory.appendingPathComponent($0) }
    }

    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    return enumerator.compactMap { item -> URL? in
        guard let url = item as? URL, url.pathExtension == "safetensors" else {
            return nil
        }
        return url
    }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads model weight `safetensor` files in the given `modelDirectory`,
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
    for url in try safetensorWeightURLs(in: modelDirectory) {
        let (w, m) = try loadArraysAndMetadata(url: url)
        for (key, value) in w {
            weights[key] = value
        }
        if metadata.isEmpty {
            metadata = m
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // quantize if needed
    var builtMXFP8Layer = false
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
                    builtMXFP8Layer = true
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
    // Strict verification (.all) by default — a checkpoint whose packed shapes don't
    // match the quantized modules must fail loudly, not decode as garbage (upstream
    // regression #395). Relax to .noUnusedKeys ONLY when an mxfp8 layer was built:
    // pre-quantized mxfp8 models carry a different weight layout than the affine
    // QuantizedLinear placeholder above, and their shapes are reconciled via
    // _updateInternal once the real data is applied.
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(
        parameters: parameters, verify: builtMXFP8Layer ? [.noUnusedKeys] : [.all])

    eval(model)
}
