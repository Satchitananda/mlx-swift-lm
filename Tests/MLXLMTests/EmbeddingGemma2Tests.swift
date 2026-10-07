import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite(.serialized) struct EmbeddingGemma2Tests {
    private struct ProcessorTokenizer: Tokenizer {
        var bosToken: String? { "s2" }
        var eosToken: String? { "s1" }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [2, 4, 1] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertIdToToken(_ id: Int) -> String? { "s\(id)" }
        func convertTokenToId(_ token: String) -> Int? { Int(token.dropFirst()) }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
    }
    private func fixture(_ name: String) -> URL {
        Bundle.module.resourceURL!.appending(path: "EmbeddingGemma2/\(name)")
    }
    private func configuration() throws -> EmbeddingGemma2Configuration {
        try JSONDecoder().decode(
            EmbeddingGemma2Configuration.self, from: Data(contentsOf: fixture("config.json")))
    }

    @Test(arguments: [DType.float32, .bfloat16]) func matchesPinnedEncoder(_ dtype: DType) throws {
        let model = try EmbeddingGemma2Model(configuration(), towers: [])
        let weights = try loadArrays(url: fixture("tiny-float32.safetensors")).mapValues {
            $0.asType(dtype)
        }
        try model.update(parameters: .unflattened(weights), verify: [.all])
        let expected = try loadArrays(url: fixture("expected.safetensors"))
        let output = try model(
            .init(
                tokens: expected["tokens"]!, valid: expected["valid"]!,
                images: [], videoFrames: [], audio: []))
        eval(output)
        #expect(output.dtype == .float32)
        let key = dtype == .float32 ? "float32" : "bf16"
        #expect(
            allClose(output, expected[key]!, atol: dtype == .float32 ? 1e-6 : 2e-3).all().item(
                Bool.self))
        #expect(
            allClose(sum(output * output, axis: 1), MLXArray.ones([2]), atol: 1e-6).all().item(
                Bool.self))
        // Removing batch padding leaves the first result unchanged.
        let individual = try model(
            .init(
                tokens: expected["tokens"]![0, ..<4].reshaped(1, 4),
                valid: MLXArray.ones([1, 4]).asType(.bool), images: [], videoFrames: [], audio: []))
        #expect(
            allClose(output[0], individual[0], atol: dtype == .float32 ? 1e-6 : 2e-3).all().item(
                Bool.self))
    }

    @Test(arguments: [DType.float32, .bfloat16]) func rotaryRoundsTrigToActivationDtype(
        _ dtype: DType
    ) throws {
        let expected = try loadArrays(url: fixture("expected.safetensors"))
        let x = MLXArray.arange(96, dtype: .float32).reshaped(2, 2, 3, 8).asType(dtype) / 31
        let positions = MLXArray([511, 512, 513, 2047, 2048, 2049], [2, 3])
        let result = EmbeddingGemma2Rotary(dimensions: 8, theta: 1_000_000)(x, positions: positions)
        let key = dtype == .float32 ? "rotary_float32" : "rotary_bf16"
        #expect(allClose(result, expected[key]!, atol: 1e-6).all().item(Bool.self))
    }

    @Test func localMaskIsBidirectionalAndIncludesWindowEdge() {
        let valid = MLXArray([1, 1, 1, 1, 0], [1, 5])
        let masks = EmbeddingGemma2Text.masks(valid, window: 2)
        #expect(masks.full[0, 0, 0, 3].item(Bool.self))
        #expect(masks.local[0, 0, 0, 2].item(Bool.self))
        #expect(masks.local[0, 0, 2, 0].item(Bool.self))
        #expect(!masks.local[0, 0, 0, 3].item(Bool.self))
        #expect(!masks.local[0, 0, 3, 4].item(Bool.self))
    }

    @Test(arguments: [1, 2, 4]) func preprocessedRowsAreReleasedBetweenBoundedChunks(
        _ batchSize: Int
    ) throws {
        final class Tracker {
            var active = 0
            var peak = 0
        }
        final class Row {
            let value: Int
            let tracker: Tracker
            init(_ value: Int, tracker: Tracker) {
                self.value = value
                self.tracker = tracker
                tracker.active += 1
                tracker.peak = max(tracker.peak, tracker.active)
            }
            deinit { tracker.active -= 1 }
        }
        let tracker = Tracker()
        var consumed: [Int] = []
        let outputs = try mapEmbeddingGemma2Batches(
            [0, 1, 2, 3], maximumBatchSize: batchSize,
            prepare: { Row($0, tracker: tracker) },
            consume: { rows in
                #expect(tracker.active <= batchSize)
                consumed += rows.map(\.value)
                return rows.map(\.value)
            })
        #expect(outputs == [0, 1, 2, 3])
        #expect(consumed == outputs)
        #expect(tracker.peak == batchSize)
        #expect(tracker.active == 0)
    }

    @Test func longPaddingKeepsAttentionAndPoolingFinite() throws {
        let model = try EmbeddingGemma2Model(configuration(), towers: [])
        try model.update(
            parameters: .unflattened(loadArrays(url: fixture("tiny-float32.safetensors"))),
            verify: [.all])
        let rows: [EmbeddingGemma2Prepared] = [
            .init(
                tokens: MLXArray([2, 8, 7, 1], [1, 4]), valid: MLXArray.ones([1, 4]).asType(.bool),
                images: [], videoFrames: [], audio: []),
            .init(
                tokens: MLXArray(Array(repeating: 6, count: 40), [1, 40]),
                valid: MLXArray.ones([1, 40]).asType(.bool),
                images: [], videoFrames: [], audio: []),
        ]
        let batch = try EmbeddingGemma2Processor.batch(rows, padToken: 0)
        let masks = EmbeddingGemma2Text.masks(batch.valid, window: 2)
        #expect(all(sum(masks.local.asType(.int32), axis: -1) .> 0).item(Bool.self))
        let together = try model(batch)
        let individual = try model(rows[0])
        #expect(allClose(together[0], individual[0], atol: 1e-6).all().item(Bool.self))
        let hidden = MLXArray([Float(1), 1, .nan, .nan], [1, 2, 2])
        let pooled = EmbeddingGemma2Model.pool(hidden, valid: MLXArray([1, 0], [1, 2]))
        #expect(pooled.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    @Test(arguments: [DType.float32, .bfloat16]) func processorBatchPreservesIndependentRows(
        _ dtype: DType
    ) throws {
        let model = try EmbeddingGemma2Model(configuration(), towers: [])
        let weights = try loadArrays(url: fixture("tiny-float32.safetensors")).mapValues {
            $0.asType(dtype)
        }
        try model.update(parameters: .unflattened(weights), verify: [.all])
        let rows: [EmbeddingGemma2Prepared] = [
            .init(
                tokens: MLXArray([2, 8, 7, 1], [1, 4]), valid: MLXArray.ones([1, 4]).asType(.bool),
                images: [], videoFrames: [], audio: []),
            .init(
                tokens: MLXArray([2, 6, 1], [1, 3]), valid: MLXArray.ones([1, 3]).asType(.bool),
                images: [], videoFrames: [], audio: []),
        ]
        let batch = try EmbeddingGemma2Processor.batch(rows, padToken: 0)
        #expect(batch.tokens.shape == [2, 4])
        #expect(
            batch.valid.asArray(Bool.self) == [true, true, true, true, true, true, true, false])
        let together = try model(batch)
        eval(together)
        for index in rows.indices {
            let separate = try model(rows[index])
            #expect(
                allClose(together[index], separate[0], atol: dtype == .float32 ? 1e-6 : 2e-3).all()
                    .item(Bool.self))
        }
        #expect(throws: EmbeddingGemma2Error.self) {
            try EmbeddingGemma2Processor.batch([], padToken: 0)
        }
    }

    @Test func processorBatchKeepsMediaOrderAcrossSamples() throws {
        let rows = (1 ... 2).map { index in
            EmbeddingGemma2Prepared(
                tokens: MLXArray([2, 28, 1], [1, 3]),
                valid: MLXArray.ones([1, 3]).asType(.bool), images: [MLXArray(index)],
                videoFrames: [MLXArray(index + 10)], audio: [(MLXArray(index + 20), MLXArray(true))]
            )
        }
        let batch = try EmbeddingGemma2Processor.batch(rows, padToken: 0)
        #expect(batch.images.map { $0.item(Int.self) } == [1, 2])
        #expect(batch.videoFrames.map { $0.item(Int.self) } == [11, 12])
        #expect(batch.audio.map { $0.features.item(Int.self) } == [21, 22])
    }

    @Test func scatterPreservesOrderedMediaSlotsAndRejectsMismatch() throws {
        let base = MLXArray.zeros([2, 3, 2])
        let tokens = MLXArray([1, 28, 28, 28, 2, 0], [2, 3])
        let features = MLXArray([1, 2, 3, 4, 5, 6], [3, 2])
        let output = try EmbeddingGemma2Model.scatter(
            base, tokens: tokens, mediaToken: 28, features: features)
        #expect(output.asArray(Int.self) == [0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0])
        #expect(throws: EmbeddingGemma2Error.self) {
            try EmbeddingGemma2Model.scatter(base, tokens: tokens, mediaToken: 28, features: nil)
        }
    }

    @Test func poolingIncludesMediaAndExcludesPadding() {
        let hidden = MLXArray([1, 0, 0, 1, 999, 999], [1, 3, 2])
        let pooled = EmbeddingGemma2Model.pool(hidden, valid: MLXArray([1, 1, 0], [1, 3]))
        #expect(
            allClose(
                pooled, MLXArray([Float(0.5).squareRoot(), Float(0.5).squareRoot()], [1, 2]),
                atol: 1e-6
            ).all().item(Bool.self))
    }

    @Test func bf16PoolingAccumulatesInFloat32() {
        let hidden = MLXArray([Float(1), 0, 0.00390625, 1, 0.00390625, 1], [1, 3, 2]).asType(
            .bfloat16)
        let actual = EmbeddingGemma2Model.pool(hidden, valid: MLXArray.ones([1, 3]))
        let mean = sum(hidden.asType(.float32), axis: 1) / 3
        let expected = mean / sqrt(sum(mean.square(), axis: -1, keepDims: true))
        #expect(allClose(actual, expected, atol: 1e-7).all().item(Bool.self))
    }

    @Test func nativeRGBResizeMatchesPinnedProcessor() throws {
        let arrays = try loadArrays(url: fixture("processor.safetensors"))
        let image = try EmbeddingGemma2Image(
            rgb: arrays["rgb"]!.asArray(UInt8.self), width: 13, height: 11)
        for (height, width) in [(15, 17), (7, 5), (11, 17), (7, 13)] {
            let actual = EmbeddingGemma2ImageProcessing.resizedRGB(
                image, width: width, height: height)
            #expect(actual == arrays["rgb_\(height)_\(width)"]!.asArray(UInt8.self))
        }
    }

    @Test func processorRejectsAggregateMediaBudgetBeforeAllocatingEveryImage() throws {
        let processor = EmbeddingGemma2Processor(
            configuration: try configuration(), tokenizer: ProcessorTokenizer())
        let image = try EmbeddingGemma2Image(rgb: [0, 0, 0], width: 1, height: 1)
        #expect(throws: EmbeddingGemma2Error.self) {
            try processor.prepare(Array(repeating: .image(image), count: 128))
        }
        #expect(throws: EmbeddingGemma2Error.self) {
            try processor.prepare([.text("s24")])
        }
        #expect(throws: EmbeddingGemma2Error.self) {
            try processor.prepare([.audio([.nan])])
        }
    }

    @Test func registeredControlTokensAreRejectedAcrossAdjacentTextParts() throws {
        let c = try configuration()
        let processor = EmbeddingGemma2Processor(configuration: c, tokenizer: ProcessorTokenizer())
        for id in c.controlTokenIDs {
            #expect(throws: EmbeddingGemma2Error.self) {
                try processor.prepare([.text("hello s\(id) world")])
            }
        }
        #expect(throws: EmbeddingGemma2Error.self) {
            try processor.prepare([.text("hello s"), .text("1 world")])
        }
    }

    @Test(arguments: ["pad_token_id", "bos_token_id", "eos_token_id"])
    func vocabularyBoundsRejectSpecialTokenGather(_ key: String) throws {
        var json = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: fixture("config.json")))
                as? [String: Any])
        var text = try #require(json["text_config"] as? [String: Any])
        text[key] = 40
        json["text_config"] = text
        #expect(throws: EmbeddingGemma2Error.self) {
            try JSONDecoder().decode(
                EmbeddingGemma2Configuration.self,
                from: JSONSerialization.data(withJSONObject: json))
        }
    }

    @Test(arguments: [0, 1, 128]) func subFrameAudioNeverGathersOutsidePadding(_ count: Int) throws
    {
        let result = Gemma4AudioFeatureExtractor()(
            MLXArray(Array(repeating: Float(0), count: count)))
        #expect(result.features.shape == [1, 0, 128])
        #expect(result.mask.shape == [1, 0])
        let processor = EmbeddingGemma2Processor(
            configuration: try configuration(), tokenizer: ProcessorTokenizer())
        #expect(throws: EmbeddingGemma2Error.self) {
            try processor.prepare([.audio(Array(repeating: 0, count: count))])
        }
    }

    @Test(arguments: [0, 1]) func zeroPastAudioContextStillAttendsTheCurrentFrame(_ left: Int)
        throws
    {
        let c = try JSONDecoder().decode(
            Gemma4AudioConfiguration.self,
            from: Data(
                """
                {"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,
                 "attention_chunk_size":3,"attention_context_left":\(left),"attention_context_right":0}
                """.utf8))
        let mask = Gemma4AudioModel(config: c).buildCausalValidMask()
        #expect(
            mask.asArray(Bool.self) == [true, false, false, false, true, false, false, false, true])
    }

    @Test func nativeAudioFeaturesMatchPinnedProcessor() throws {
        let arrays = try loadArrays(url: fixture("processor.safetensors"))
        let actual = Gemma4AudioFeatureExtractor()(arrays["waveform"]!)
        #expect(actual.features.shape == arrays["mel"]!.shape)
        // The native FFT is FP32; the reference NumPy FFT accumulates in FP64.
        // Bound both the worst log-mel error and RMS, including bins near the floor.
        #expect(abs(actual.features - arrays["mel"]!).max().item(Float.self) < 0.002)
        #expect(sqrt(mean((actual.features - arrays["mel"]!).square())).item(Float.self) < 0.0006)
        #expect(actual.mask.asArray(Bool.self) == arrays["mel_valid"]!.asArray(Bool.self))
    }

    @Test func audioAttentionExcludesOuterContextBoundary() throws {
        let c = try JSONDecoder().decode(
            Gemma4AudioConfiguration.self,
            from: Data(
                """
                {"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
                 "attention_chunk_size": 3, "attention_context_left": 3, "attention_context_right": 2}
                """.utf8))
        let mask = Gemma4AudioModel(config: c).buildCausalValidMask()
        #expect(mask.shape == [3, 7])
        #expect(!mask[0, 0].item(Bool.self))  // distance == maxPast
        #expect(mask[0, 1].item(Bool.self))
        #expect(mask[0, 2].item(Bool.self))  // current key
        #expect(mask[0, 3].item(Bool.self))
        #expect(!mask[0, 4].item(Bool.self))  // future distance == maxFuture
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["EMBEDDINGGEMMA2_CHECKPOINT"] != nil))
    func pretrainedAllModalitiesMatchReference() throws {
        let environment = ProcessInfo.processInfo.environment
        let checkpoint = URL(filePath: try #require(environment["EMBEDDINGGEMMA2_CHECKPOINT"]))
        let root = URL(filePath: try #require(environment["EMBEDDINGGEMMA2_PARITY"]))
        let c = try JSONDecoder().decode(
            EmbeddingGemma2Configuration.self,
            from: Data(contentsOf: checkpoint.appending(path: "config.json")))
        let model = try EmbeddingGemma2Model(c, towers: [.vision, .audio])
        try MLXLMCommon.loadWeights(modelDirectory: checkpoint, model: model)
        let names = [
            "query", "text", "image", "video", "audio", "long_audio", "text_image",
            "image_text", "visual_audio", "audio_visual", "all_modalities", "heterogeneous_batch",
        ]
        for name in names {
            let arrays = try loadArrays(url: root.appending(path: "\(name).safetensors"))
            func media(_ prefix: String) -> [MLXArray] {
                (0 ..< 32).compactMap { arrays["\(prefix)_\($0)"] }
            }
            var audio: [(features: MLXArray, valid: MLXArray)] = []
            if let features = arrays["audio"], let valid = arrays["audio_valid"] {
                for i in 0 ..< features.dim(0) {
                    audio.append(
                        (
                            features[i].reshaped(1, features.dim(1), features.dim(2)),
                            valid[i].reshaped(1, valid.dim(1))
                        ))
                }
            }
            let output = try model(
                .init(
                    tokens: arrays["tokens"]!, valid: arrays["valid"]!,
                    images: media("image"), videoFrames: media("video"), audio: audio))
            eval(output)
            let cosine = sum(output * arrays["expected"]!, axis: -1).min().item(Float.self)
            print("EmbeddingGemma2 native parity", name, cosine)
            #expect(cosine > 0.9995, "\(name) cosine \(cosine)")
        }
        let textOnly = try EmbeddingGemma2Model(c, towers: [])
        try MLXLMCommon.loadWeights(modelDirectory: checkpoint, model: textOnly)
        #expect(
            textOnly.parameters().flattened().allSatisfy {
                !$0.0.hasPrefix("vision_tower") && !$0.0.hasPrefix("audio_tower")
                    && !$0.0.hasPrefix("embed_")
            })
        let query = try loadArrays(url: root.appending(path: "query.safetensors"))
        let textOutput = try textOnly(
            .init(
                tokens: query["tokens"]!, valid: query["valid"]!, images: [], videoFrames: [],
                audio: []))
        #expect(sum(textOutput * query["expected"]!, axis: -1).min().item(Float.self) > 0.99999)
    }
}
