// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXVLM

/// The audio features are built on one thread and consumed on another: the processor prepares
/// them on its own executor, the token iterator evaluates them wherever it runs. Every op in the
/// pipeline must stay on the caller's (thread-unsafe, process-wide) stream. The core's `rfft`
/// padding path once recorded its `slice_update` on the calling thread's default stream, which
/// made the graph unevaluable from any other thread ("There is no Stream(gpu, N) in current
/// thread"). The core fix (ml-explore/mlx#4746) is carried by this fork's `mlx-swift`; these tests
/// keep the production path on that contract.
private final class ArrayBox: @unchecked Sendable {
    let values: [MLXArray]
    init(_ values: [MLXArray]) { self.values = values }
}

@Suite(.serialized)
struct Gemma4AudioFeatureStreamTests {
    /// Evaluates on a real thread, never a task that may land on the creating thread.
    private static func evaluateOnAnotherThread(_ arrays: [MLXArray]) {
        let box = ArrayBox(arrays)
        let done = DispatchSemaphore(value: 0)
        let thread = Thread {
            eval(box.values)
            done.signal()
        }
        thread.start()
        done.wait()
    }

    @Test("Audio features built on one thread evaluate on another")
    func featuresEvaluateAcrossThreads() {
        let extractor = Gemma4AudioFeatureExtractor(sampleRate: 16_000)
        // 8 s at 16 kHz: 320-sample frames padded to the 512-point FFT, the production shape.
        let samples = (0 ..< 128_000).map { Float(sin(Double($0) * 0.05)) }
        let (features, mask) = extractor(MLXArray(samples))
        Self.evaluateOnAnotherThread([features, mask])
        #expect(features.dim(0) == 1)
        #expect(features.dim(2) == 128)
        #expect(mask.dim(1) == features.dim(1))
    }

    @Test("Explicit frame padding matches the FFT's own padding")
    func explicitPaddingMatchesRequestedLength() {
        let frames = MLXArray.ones([4, 320]) * MLXArray(0.5)
        let viaN = rfft(frames, n: 512, axis: -1)
        let viaPad = rfft(padded(frames, widths: [IntOrPair((0, 0)), IntOrPair((0, 192))]), axis: -1)
        eval(viaN, viaPad)
        #expect(viaN.shape == viaPad.shape)
        let difference = abs(viaN.realPart() - viaPad.realPart()).max().item(Float.self)
        #expect(difference < 1e-5)
    }
}
