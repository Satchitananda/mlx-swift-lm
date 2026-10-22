// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

// MARK: - Audio input for the unified Gemma 4 (`gemma4_unified`, 12B) VLM
//
// The unified checkpoints have no audio tower. Their processor
// (HF `Gemma4UnifiedAudioFeatureExtractor`) cuts the raw 16 kHz waveform into
// frames of `audio_samples_per_token` samples (640 = 40 ms); every frame is one
// audio soft token whose features are the raw samples, projected by `embed_audio`
// (RMSNorm -> Linear 640 -> hidden). The prompt reserves `ceil(samples / 640)`
// audio tokens per clip between `boa` and `eoa`. Both Gemma 4 processors expand
// placeholders and batch clips through the shared `Gemma4AudioInput`.

/// Records the chat-template messages and emits one audio placeholder per audio content
/// part (times `placeholdersPerAudioPart`), so the tests prove the processor's message
/// generation (not a hard-coded prompt) is what reserves the audio slot.
private final class AudioTestTokenizer: Tokenizer, @unchecked Sendable {
    let vocabularySize: Int = 64
    let bosToken: String? = nil
    let eosToken: String? = nil
    let eosTokenId: Int? = 1
    let unknownToken: String? = "<unk>"
    let unknownTokenId: Int? = 3

    /// Special-token strings the real Gemma 4 tokenizer carries. Unknown strings map to
    /// `unknownTokenId`, as the swift-transformers BPE tokenizer does.
    let specialTokens: [String: Int]
    let audioTokenId: Int
    let placeholdersPerAudioPart: Int

    private let lock = NSLock()
    private var _templatedMessages: [[[String: any Sendable]]] = []
    var templatedMessages: [[[String: any Sendable]]] { lock.withLock { _templatedMessages } }

    init(
        specialTokens: [String: Int] = ["<|audio>": 28, "<audio|>": 29, "<|audio|>": 30],
        audioTokenId: Int = 30,
        placeholdersPerAudioPart: Int = 1
    ) {
        self.specialTokens = specialTokens
        self.audioTokenId = audioTokenId
        self.placeholdersPerAudioPart = placeholdersPerAudioPart
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [2, 5] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        specialTokens[token] ?? unknownTokenId
    }

    func convertIdToToken(_ id: Int) -> String? {
        specialTokens.first { $0.value == id }?.key ?? String(id)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        lock.withLock { _templatedMessages.append(messages) }
        var tokens = [2]
        for message in messages {
            guard let parts = message["content"] as? [[String: String]] else { continue }
            for part in parts where part["type"] == "audio" {
                tokens.append(
                    contentsOf: repeatElement(audioTokenId, count: placeholdersPerAudioPart))
            }
        }
        tokens.append(5)
        return tokens
    }
}

struct Gemma4UnifiedAudioTests {

    /// 12B-shaped processor config: `audio_ms_per_token` at the top level, the sample
    /// rate under `feature_extractor`, no `audio_seq_length`, no boa/eoa ids.
    private static func processorConfig(extra: String = "", featureExtractorExtra: String = "")
        throws -> Gemma4UnifiedProcessorConfiguration
    {
        let json = """
            {
              "processor_class": "Gemma4UnifiedProcessor",
              "image_token_id": 31,
              "audio_token_id": 30,
              "boi_token_id": 26,
              "eoi_token_id": 27,
              \(extra)
              "image_processor": {
                "patch_size": 2,
                "pooling_kernel_size": 2,
                "model_patch_size": 4,
                "max_soft_tokens": 4,
                "size": { "height": 8, "width": 8 }
              },
              "feature_extractor": {
                "feature_extractor_type": "Gemma4UnifiedAudioFeatureExtractor",
                \(featureExtractorExtra)
                "sampling_rate": 16000,
                "num_mel_filters": 128,
                "fft_length": 512,
                "hop_length": 160
              },
              "audio_ms_per_token": 40
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4UnifiedProcessorConfiguration.self, from: Data(json.utf8))
    }

    private static func waveform(samples: Int) -> MLXArray {
        MLXArray((0 ..< samples).map { Float($0 % 97) / 97 - 0.5 })
    }

    // MARK: Configuration

    @Test("Unified processor config derives 640 samples per token from audio_ms_per_token")
    func configDerivesSamplesPerToken() throws {
        let config = try Self.processorConfig()
        #expect(config.audioMsPerToken == 40)
        #expect(config.audioSampleRate == 16_000)
        #expect(config.audioSamplesPerToken == 640)
        #expect(config.audioSeqLength == 750)
        #expect(config.boaTokenId == nil)
        #expect(config.eoaTokenId == nil)
    }

    @Test("Explicit feature_extractor.audio_samples_per_token and boa/eoa ids win")
    func configExplicitAudioFields() throws {
        let config = try Self.processorConfig(
            extra: "\"boa_token_id\": 256000, \"eoa_token_id\": 258883,",
            featureExtractorExtra: "\"audio_samples_per_token\": 512,")
        #expect(config.audioSamplesPerToken == 512)
        #expect(config.boaTokenId == 256_000)
        #expect(config.eoaTokenId == 258_883)
    }

    @Test("A non-positive samples-per-token fails decoding")
    func configRejectsZeroSamplesPerToken() {
        #expect(throws: DecodingError.self) {
            try JSONDecoder.json5().decode(
                Gemma4UnifiedProcessorConfiguration.self,
                from: Data(
                    """
                    { "processor_class": "Gemma4UnifiedProcessor", "audio_ms_per_token": 0 }
                    """.utf8))
        }
    }

    // MARK: Raw-frame extractor

    @Test(
        "softTokenCount = ceil(samples / 640)",
        arguments: [(0, 0), (1, 1), (639, 1), (640, 1), (641, 2), (16_000, 25), (480_000, 750)])
    func softTokenCountRule(samples: Int, expected: Int) {
        #expect(Gemma4UnifiedAudioFeatureExtractor().softTokenCount(samples: samples) == expected)
    }

    @Test("Extractor frames the raw samples, zero-pads the tail frame, marks every frame valid")
    func extractorFramesRawSamples() {
        let extractor = Gemma4UnifiedAudioFeatureExtractor()
        let wave = Self.waveform(samples: 1000)
        let (features, mask) = extractor(wave)
        eval(features, mask)

        #expect(features.shape == [1, 2, 640])
        #expect(features.dtype == .float32)
        #expect(mask.shape == [1, 2])
        #expect(mask.asArray(Bool.self) == [true, true])
        // Frame 0 is the first 640 samples untouched; frame 1 is the remaining 360 then zeros.
        #expect(features[0, 0].asArray(Float.self) == wave[..<640].asArray(Float.self))
        #expect(features[0, 1, ..<360].asArray(Float.self) == wave[640...].asArray(Float.self))
        #expect(features[0, 1, 360...].abs().max().item(Float.self) == 0)

        let (exact, _) = extractor(Self.waveform(samples: 1280))
        #expect(exact.shape == [1, 2, 640])
        let (empty, emptyMask) = extractor(MLXArray([Float]()))
        #expect(empty.shape == [1, 0, 640])
        #expect(emptyMask.shape == [1, 0])
    }

    // MARK: Prepare

    @Test(
        "Unified processor reserves boa + ceil(samples/640) audio tokens + eoa and carries raw frames"
    )
    func prepareExpandsAudioPlaceholderAndCarriesRawFrames() async throws {
        let tokenizer = AudioTestTokenizer()
        let processor = Gemma4UnifiedProcessor(try Self.processorConfig(), tokenizer: tokenizer)

        // 1000 samples at 16 kHz -> 2 frames of 640 (the second zero-padded).
        let input = try await processor.prepare(
            input: UserInput(
                prompt: "Transcribe this.", audios: [.array(Self.waveform(samples: 1000))]))

        // The message generator hands the template one audio part per clip.
        let parts = tokenizer.templatedMessages.first?.first?["content"] as? [[String: String]]
        #expect(parts?.filter { $0["type"] == "audio" }.count == 1)

        #expect(input.text.tokens.asArray(Int32.self) == [2, 28, 30, 30, 29, 5])
        let audio = try #require(input.audio)
        #expect(audio.features.shape == [1, 2, 640])
        #expect(audio.features.dtype == .float32)
        #expect(audio.mask?.shape == [1, 2])
        #expect(audio.mask?.asArray(Bool.self) == [true, true])
    }

    @Test("Without audio the prompt is the chat template's output byte for byte")
    func prepareWithoutAudioLeavesPromptUnchanged() async throws {
        let tokenizer = AudioTestTokenizer()
        let processor = Gemma4UnifiedProcessor(try Self.processorConfig(), tokenizer: tokenizer)

        let input = try await processor.prepare(input: UserInput(prompt: "Hello"))

        let templated = try tokenizer.applyChatTemplate(
            messages: tokenizer.templatedMessages[0], tools: nil, additionalContext: nil)
        #expect(input.text.tokens.asArray(Int32.self) == templated.map(Int32.init))
        #expect(input.text.tokens.asArray(Int32.self) == [2, 5])
        #expect(input.audio == nil)
    }

    @Test("Configured boa/eoa ids take precedence over the tokenizer's vocabulary")
    func prepareUsesConfiguredDelimiters() async throws {
        let processor = Gemma4UnifiedProcessor(
            try Self.processorConfig(extra: "\"boa_token_id\": 40, \"eoa_token_id\": 41,"),
            tokenizer: AudioTestTokenizer())

        let input = try await processor.prepare(
            input: UserInput(prompt: "x", audios: [.array(Self.waveform(samples: 640))]))

        #expect(input.text.tokens.asArray(Int32.self) == [2, 40, 30, 41, 5])
    }

    @Test("Missing boa/eoa in both the config and the tokenizer fails typed")
    func prepareFailsWithoutDelimiters() async throws {
        let processor = Gemma4UnifiedProcessor(
            try Self.processorConfig(),
            tokenizer: AudioTestTokenizer(specialTokens: ["<|audio|>": 30]))

        await #expect(throws: Gemma4Error.audioDelimiterTokensUnavailable) {
            try await processor.prepare(
                input: UserInput(prompt: "x", audios: [.array(Self.waveform(samples: 640))]))
        }
    }

    @Test("Placeholders that do not pair with clips fail typed")
    func prepareFailsOnPlaceholderMismatch() async throws {
        let twoPlaceholders = Gemma4UnifiedProcessor(
            try Self.processorConfig(),
            tokenizer: AudioTestTokenizer(placeholdersPerAudioPart: 2))
        await #expect(throws: Gemma4Error.audioPlaceholderMismatch(audios: 1, placeholders: 2)) {
            try await twoPlaceholders.prepare(
                input: UserInput(prompt: "x", audios: [.array(Self.waveform(samples: 640))]))
        }

        let noPlaceholder = Gemma4UnifiedProcessor(
            try Self.processorConfig(),
            tokenizer: AudioTestTokenizer(placeholdersPerAudioPart: 0))
        await #expect(throws: Gemma4Error.audioPlaceholderMismatch(audios: 1, placeholders: 0)) {
            try await noPlaceholder.prepare(
                input: UserInput(prompt: "x", audios: [.array(Self.waveform(samples: 640))]))
        }
    }

    @Test("Several clips batch right-padded to the longest, the mask marking the padding")
    func prepareBatchesClips() async throws {
        let processor = Gemma4UnifiedProcessor(
            try Self.processorConfig(), tokenizer: AudioTestTokenizer())

        let input = try await processor.prepare(
            input: UserInput(
                prompt: "compare",
                audios: [
                    .array(Self.waveform(samples: 1000)), .array(Self.waveform(samples: 2000)),
                ]))

        #expect(
            input.text.tokens.asArray(Int32.self)
                == [2, 28, 30, 30, 29, 28, 30, 30, 30, 30, 29, 5])
        let audio = try #require(input.audio)
        #expect(audio.features.shape == [2, 4, 640])
        #expect(audio.mask?.shape == [2, 4])
        #expect(
            audio.mask?.asArray(Bool.self) == [true, true, false, false, true, true, true, true])
        #expect(audio.features[0, 2...].abs().max().item(Float.self) == 0)
    }

    // MARK: Model contract

    /// The processor's `[1, N, samplesPerToken]` frames and `boa + N + eoa` prompt are exactly
    /// what `Gemma4Unified` consumes: `embed_audio` projects every frame and scatters it onto
    /// the N audio positions.
    @Test("Processor output feeds a unified model with an audio_config end to end")
    func processorOutputFeedsUnifiedModel() async throws {
        let modelConfig = try JSONDecoder.json5().decode(
            Gemma4UnifiedConfiguration.self,
            from: Data(
                """
                {
                  "model_type": "gemma4_unified",
                  "vocab_size": 32,
                  "image_token_id": 31,
                  "audio_token_id": 30,
                  "video_token_id": 25,
                  "boa_token_id": 28,
                  "eoa_token_id": 29,
                  "text_config": {
                    "model_type": "gemma4_unified_text",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 8,
                    "global_head_dim": 8,
                    "vocab_size": 32,
                    "vocab_size_per_layer_input": 32,
                    "num_kv_shared_layers": 0,
                    "hidden_size_per_layer_input": 0,
                    "sliding_window": 8,
                    "sliding_window_pattern": 1,
                    "attention_k_eq_v": true,
                    "use_double_wide_mlp": false,
                    "layer_types": ["full_attention"],
                    "tie_word_embeddings": true
                  },
                  "vision_config": null,
                  "audio_config": {
                    "model_type": "gemma4_unified_audio",
                    "audio_samples_per_token": 8,
                    "audio_embed_dim": 8,
                    "hidden_size": 8,
                    "output_proj_dims": 8,
                    "rms_norm_eps": 1e-6
                  }
                }
                """.utf8))
        let model = Gemma4Unified(modelConfig)
        eval(model)

        let processor = Gemma4UnifiedProcessor(
            try Self.processorConfig(featureExtractorExtra: "\"audio_samples_per_token\": 8,"),
            tokenizer: AudioTestTokenizer())
        // 20 samples / 8 per token -> 3 audio tokens.
        let input = try await processor.prepare(
            input: UserInput(prompt: "listen", audios: [.array(Self.waveform(samples: 20))]))
        #expect(input.text.tokens.asArray(Int32.self) == [2, 28, 30, 30, 30, 29, 5])
        #expect(input.audio?.features.shape == [1, 3, 8])

        let result = try model.prepare(
            input, cache: try model.newCache(parameters: nil), state: nil, prefill: .init())
        guard case .logits(let output) = result else {
            Issue.record("Expected Gemma4Unified.prepare to return logits")
            return
        }
        eval(output.logits)
        #expect(output.logits.shape == [1, 7, 32])
        #expect(output.logits.abs().max().item(Float.self).isFinite)
    }

    // MARK: Base Gemma 4 (mel + USM tower) through the shared helper

    private static func melProcessorConfig(audioSeqLength: Int = 750) throws
        -> Gemma4ProcessorConfiguration
    {
        try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self,
            from: Data(
                """
                {
                  "processor_class": "Gemma4Processor",
                  "audio_token_id": 30,
                  "boa_token_id": 28,
                  "eoa_token_id": 29,
                  "audio_seq_length": \(audioSeqLength),
                  "audio_sampling_rate": 16000
                }
                """.utf8))
    }

    @Test("Gemma4Processor reserves min(ceil(ceil(V/2)/2), audio_seq_length) mel tokens per clip")
    func melProcessorExpandsAudioPlaceholder() async throws {
        // 1 s at 16 kHz -> 99 valid mel frames -> ceil(ceil(99/2)/2) = 25 soft tokens.
        let processor = Gemma4Processor(
            try Self.melProcessorConfig(), tokenizer: AudioTestTokenizer())
        let input = try await processor.prepare(
            input: UserInput(prompt: "x", audios: [.array(Self.waveform(samples: 16_000))]))
        #expect(
            input.text.tokens.asArray(Int32.self)
                == [2, 28] + Array(repeating: Int32(30), count: 25) + [29, 5])
        let audio = try #require(input.audio)
        #expect(audio.features.shape == [1, 99, 128])
        #expect(audio.mask?.shape == [1, 99])

        let capped = Gemma4Processor(
            try Self.melProcessorConfig(audioSeqLength: 10), tokenizer: AudioTestTokenizer())
        let cappedInput = try await capped.prepare(
            input: UserInput(prompt: "x", audios: [.array(Self.waveform(samples: 16_000))]))
        #expect(
            cappedInput.text.tokens.asArray(Int32.self)
                == [2, 28] + Array(repeating: Int32(30), count: 10) + [29, 5])
    }

    @Test("Gemma4Processor fails typed when placeholders and clips do not pair up")
    func melProcessorFailsOnPlaceholderMismatch() async throws {
        let processor = Gemma4Processor(
            try Self.melProcessorConfig(),
            tokenizer: AudioTestTokenizer(placeholdersPerAudioPart: 2))
        await #expect(throws: Gemma4Error.audioPlaceholderMismatch(audios: 1, placeholders: 2)) {
            try await processor.prepare(
                input: UserInput(prompt: "x", audios: [.array(Self.waveform(samples: 640))]))
        }
    }
}
