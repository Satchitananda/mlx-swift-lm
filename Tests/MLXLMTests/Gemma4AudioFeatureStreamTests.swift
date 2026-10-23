// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
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

/// Emits one audio placeholder (30) per audio content part so the unified processor
/// expands and batches real clips.
private struct StreamTestTokenizer: Tokenizer {
    let vocabularySize: Int = 64
    let bosToken: String? = nil
    let eosToken: String? = nil
    let eosTokenId: Int? = 1
    let unknownToken: String? = nil
    let unknownTokenId: Int? = 0

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [2, 5] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var tokens = [2]
        for message in messages {
            guard let parts = message["content"] as? [[String: String]] else { continue }
            tokens.append(contentsOf: parts.filter { $0["type"] == "audio" }.map { _ in 30 })
        }
        tokens.append(5)
        return tokens
    }
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

    /// The unified (12B) path: raw-frame features plus the shared batch padding, built by the
    /// processor on this task and evaluated on another thread, as the token iterator does.
    @Test("Unified raw-frame audio features built on one thread evaluate on another")
    func unifiedFeaturesEvaluateAcrossThreads() async throws {
        let config = try JSONDecoder.json5().decode(
            Gemma4UnifiedProcessorConfiguration.self,
            from: Data(
                """
                {
                  "processor_class": "Gemma4UnifiedProcessor",
                  "audio_token_id": 30, "boa_token_id": 28, "eoa_token_id": 29,
                  "feature_extractor": { "sampling_rate": 16000 },
                  "audio_ms_per_token": 40
                }
                """.utf8))
        let processor = Gemma4UnifiedProcessor(config, tokenizer: StreamTestTokenizer())
        // 8 s plus a partial frame (zero-padding on the graph) and a shorter second clip
        // (batch padding on the graph).
        let long = MLXArray((0 ..< 128_100).map { Float(sin(Double($0) * 0.05)) })
        let short = MLXArray((0 ..< 16_000).map { Float(sin(Double($0) * 0.02)) })
        let input = try await processor.prepare(
            input: UserInput(prompt: "listen", audios: [.array(long), .array(short)]))
        let audio = try #require(input.audio)
        let mask = try #require(audio.mask)
        Self.evaluateOnAnotherThread([audio.features, mask, input.text.tokens])
        #expect(audio.features.shape == [2, 201, 640])
        #expect(mask.shape == [2, 201])
    }

    @Test("Explicit frame padding matches the FFT's own padding")
    func explicitPaddingMatchesRequestedLength() {
        let frames = MLXArray.ones([4, 320]) * MLXArray(0.5)
        let viaN = rfft(frames, n: 512, axis: -1)
        let viaPad = rfft(
            padded(frames, widths: [IntOrPair((0, 0)), IntOrPair((0, 192))]), axis: -1)
        eval(viaN, viaPad)
        #expect(viaN.shape == viaPad.shape)
        let difference = abs(viaN.realPart() - viaPad.realPart()).max().item(Float.self)
        #expect(difference < 1e-5)
    }
}
