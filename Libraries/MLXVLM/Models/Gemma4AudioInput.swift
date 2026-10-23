// Copyright © 2026 Apple Inc.

//
// Audio input shared by the two Gemma 4 processors.
//
// `Gemma4Processor` (E2B/E4B) feeds log-mel features to the USM-Conformer audio tower
// (`Gemma4AudioFeatureExtractor`). `Gemma4UnifiedProcessor` (12B) has no audio tower: HF
// `Gemma4UnifiedAudioFeatureExtractor` cuts the raw waveform into frames of
// `audio_samples_per_token` samples, and each frame is one audio soft token whose features
// are the raw samples, projected by `embed_audio` (RMSNorm -> Linear). What the two share —
// decoding every clip at the model's sample rate, stacking the clips into one right-padded
// batch, and expanding each audio placeholder into `boa + audio × N + eoa` — is
// `Gemma4AudioInput`.
//

import Foundation
import MLX
import MLXLMCommon

/// One clip's model-ready audio.
struct Gemma4AudioClipFeatures {
    /// `[1, T, D]`.
    let features: MLXArray
    /// `[1, T]`, `true == valid frame` (the HF `input_features_mask` convention).
    let mask: MLXArray
    /// Audio soft tokens the prompt reserves for this clip. It must equal the number of
    /// valid frames the model scatters, or the model throws
    /// `Gemma4Error.multimodalTokenCountMismatch`.
    let softTokenCount: Int
}

/// Encoder-free audio features for the unified Gemma 4 checkpoints: a port of HF
/// `Gemma4UnifiedAudioFeatureExtractor._extract_waveform_features` (mirrored by mlx-vlm
/// `models/gemma4_unified/processing_gemma4_unified.py`). The waveform is zero-padded to a
/// whole number of frames of `audioSamplesPerToken` samples and reshaped to
/// `[1, N, audioSamplesPerToken]` float32; every frame, including the zero-padded last one,
/// is valid. The reference applies no duration cap on this path (`audio_seq_length` is
/// unused), so neither does this.
public struct Gemma4UnifiedAudioFeatureExtractor: Sendable {
    public let sampleRate: Int
    public let audioSamplesPerToken: Int

    public init(sampleRate: Int = 16_000, audioSamplesPerToken: Int = 640) {
        precondition(audioSamplesPerToken > 0, "audio_samples_per_token must be positive")
        self.sampleRate = sampleRate
        self.audioSamplesPerToken = audioSamplesPerToken
    }

    /// Audio soft tokens for a clip of `samples` samples: `ceil(samples / audioSamplesPerToken)`
    /// (HF `Gemma4UnifiedProcessor._compute_audio_num_tokens`).
    public func softTokenCount(samples: Int) -> Int {
        (samples + audioSamplesPerToken - 1) / audioSamplesPerToken
    }

    /// - Parameter waveform: mono samples `[N]` at `sampleRate`.
    /// - Returns: raw frames `[1, T, audioSamplesPerToken]` (float32) and an all-valid mask
    ///   `[1, T]`, with `T == softTokenCount(samples: N)`.
    public func callAsFunction(_ waveform: MLXArray) -> (features: MLXArray, mask: MLXArray) {
        var samples = waveform.asType(.float32)
        if samples.ndim != 1 { samples = samples.reshaped(-1) }
        let count = samples.dim(0)
        let tokens = softTokenCount(samples: count)
        let padding = tokens * audioSamplesPerToken - count
        if padding > 0 {
            samples = padded(samples, widths: [IntOrPair((0, padding))])
        }
        return (
            samples.reshaped(1, tokens, audioSamplesPerToken),
            MLXArray.ones([1, tokens], dtype: .bool)
        )
    }

    func clipFeatures(_ waveform: MLXArray) -> Gemma4AudioClipFeatures {
        let (features, mask) = self(waveform)
        return Gemma4AudioClipFeatures(
            features: features, mask: mask, softTokenCount: features.dim(1))
    }
}

extension Gemma4AudioFeatureExtractor {
    /// Log-mel features plus the soft-token count the USM tower yields for the clip's valid
    /// frames (`audioTokenCount(validFrames:)`), capped at the processor's `audio_seq_length`.
    func clipFeatures(_ waveform: MLXArray, maxSoftTokens: Int) -> Gemma4AudioClipFeatures {
        let (features, mask) = self(waveform)
        let validFrames = mask.asType(.int32).sum().item(Int.self)
        return Gemma4AudioClipFeatures(
            features: features, mask: mask,
            softTokenCount: min(Self.audioTokenCount(validFrames: validFrames), maxSoftTokens))
    }
}

/// The audio preparation both Gemma 4 processors share. Every op runs on the caller's
/// default stream, so the arrays can be evaluated from another thread.
enum Gemma4AudioInput {
    /// Gemma 4 vocabulary strings for the audio delimiters, resolved through the tokenizer
    /// when a processor configuration carries no `boa_token_id` / `eoa_token_id`.
    static let boaToken = "<|audio>"
    static let eoaToken = "<audio|>"

    /// Decodes every clip at `sampleRate` (URL sources are resampled to mono at that rate;
    /// `.array` sources are taken as already being so), featurizes each, and stacks the
    /// clips into one batch right-padded to the longest clip, the mask marking the padding
    /// invalid.
    static func prepare(
        _ audios: [UserInput.Audio], sampleRate: Int,
        featurize: (MLXArray) -> Gemma4AudioClipFeatures
    ) async throws -> (audio: LMInput.ProcessedAudio, softTokenCounts: [Int]) {
        var processing = UserInput.AudioProcessing()
        processing.sampleRate = Double(sampleRate)

        var clips: [Gemma4AudioClipFeatures] = []
        clips.reserveCapacity(audios.count)
        for audio in audios {
            let waveform = try await audio.asMLXArray(processing: processing)
            clips.append(featurize(waveform))
        }

        let maxFrames = clips.map { $0.features.dim(1) }.max() ?? 0
        var features: [MLXArray] = []
        var masks: [MLXArray] = []
        for clip in clips {
            let frames = clip.features.dim(1)
            if frames < maxFrames {
                features.append(
                    padded(clip.features, widths: [0, IntOrPair((0, maxFrames - frames)), 0]))
                masks.append(
                    padded(
                        clip.mask.asType(.int32), widths: [0, IntOrPair((0, maxFrames - frames))]
                    )
                    .asType(.bool))
            } else {
                features.append(clip.features)
                masks.append(clip.mask)
            }
        }

        return (
            LMInput.ProcessedAudio(
                features: concatenated(features, axis: 0), mask: concatenated(masks, axis: 0)),
            clips.map(\.softTokenCount)
        )
    }

    /// Expands every `audioTokenId` placeholder in `tokens` into `boa + audio × N + eoa`, N
    /// being that clip's soft-token count. Placeholders and clips must pair up one to one.
    static func expandPlaceholders(
        in tokens: [Int], audioTokenId: Int, boaTokenId: Int, eoaTokenId: Int,
        softTokenCounts: [Int]
    ) throws -> [Int] {
        let placeholders = tokens.count(where: { $0 == audioTokenId })
        guard placeholders == softTokenCounts.count else {
            throw Gemma4Error.audioPlaceholderMismatch(
                audios: softTokenCounts.count, placeholders: placeholders)
        }

        var expanded: [Int] = []
        expanded.reserveCapacity(
            tokens.count + softTokenCounts.reduce(0, +) + 2 * softTokenCounts.count)
        var clip = 0
        for token in tokens {
            guard token == audioTokenId else {
                expanded.append(token)
                continue
            }
            expanded.append(boaTokenId)
            expanded.append(contentsOf: repeatElement(audioTokenId, count: softTokenCounts[clip]))
            expanded.append(eoaTokenId)
            clip += 1
        }
        return expanded
    }
}
