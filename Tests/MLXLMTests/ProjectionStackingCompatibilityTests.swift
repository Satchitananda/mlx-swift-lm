import MLX
import MLXNN
import Testing
@testable import MLXLLM

@Suite("SightRoll projection stacking compatibility")
struct ProjectionStackingCompatibilityTests {
    @Test func retainsGloballyScaledProjections() {
        let layer = QuantizedLinear(
            weight: MLXArray.zeros([8, 8], dtype: .uint32),
            scales: MLXArray.ones([8, 1]), biases: nil,
            groupSize: 64, bits: 4, globalScale: MLXArray(Float(2)))
        #expect(stackedQuantizedLinear([layer, layer]) == nil)
        #expect(layer.globalScale?.item(Float.self) == 2)
    }

    @Test func affineProjectionsKeepTheirOutputs() throws {
        let a = QuantizedLinear(64, 32, bias: false, groupSize: 64, bits: 4)
        let b = QuantizedLinear(64, 32, bias: false, groupSize: 64, bits: 4)
        let stacked = try #require(stackedQuantizedLinear([a, b]))
        let input = MLXArray.ones([1, 64])
        let expected = concatenated([a(input), b(input)], axis: -1)
        #expect(allClose(stacked(input), expected).item(Bool.self))
    }
}
