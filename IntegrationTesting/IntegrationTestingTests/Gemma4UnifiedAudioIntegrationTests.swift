// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

// MARK: - Unified Gemma 4 (12B) hears audio
//
// The unified checkpoints have no audio tower: `Gemma4UnifiedProcessor` cuts the raw
// 16 kHz waveform into 640-sample frames that `embed_audio` projects straight into the
// language model. This drives the real cached 8-bit 12B end to end: synthesize a known
// sentence with the system voice, ask for a transcript, and check the words came through.
//
// Cache-gated like `Gemma4EseriesLoadIntegrationTests`: enabled only when the checkpoint is
// already in the HF cache (~13 GB resident while it runs), never auto-downloaded.

private let unified12B = "mlx-community/gemma-4-12b-it-8bit"

@Suite(.serialized, .timeLimit(.minutes(20)))
struct Gemma4UnifiedAudioIntegrationTests {

    @Test(.enabled(if: hfSnapshotDir(modelId: unified12B) != nil))
    func transcribesSynthesizedSpeech() async throws {
        let directory = try #require(hfSnapshotDir(modelId: unified12B))
        let sentence = "The quick brown fox jumps over the lazy dog near the river bank."
        let clip = try Self.synthesizeSpeech(sentence)
        defer { try? FileManager.default.removeItem(at: clip.deletingLastPathComponent()) }

        let loadStart = ContinuousClock.now
        let context = try await VLMModelFactory.shared.load(
            from: directory, using: #huggingFaceTokenizerLoader())
        let loadSeconds = Self.seconds(since: loadStart)
        #expect(context.model is Gemma4Unified, "\(unified12B) must load as gemma4_unified")

        let prepareStart = ContinuousClock.now
        let input = try await context.processor.prepare(
            input: UserInput(chat: [
                .user("Transcribe the speech in this audio exactly.", audios: [.url(clip)])
            ]))
        let prepareSeconds = Self.seconds(since: prepareStart)

        // Raw 640-sample frames, one audio soft token each, bracketed by the delimiters the
        // processor resolved through the tokenizer (the 12B processor config carries no ids).
        let audio = try #require(input.audio)
        #expect(audio.features.dim(2) == 640)
        let tokens = input.text.tokens.asArray(Int32.self)
        #expect(tokens.contains(256_000), "missing <|audio> (boa)")
        #expect(tokens.contains(258_883), "missing <audio|> (eoa)")
        #expect(tokens.filter { $0 == 258_881 }.count == audio.features.dim(1))

        let generateStart = ContinuousClock.now
        let stream = try generate(
            input: input,
            parameters: GenerateParameters(maxTokens: 64, temperature: 0),
            context: context)
        var text = ""
        for await event in stream {
            if case .chunk(let chunk) = event { text += chunk }
        }
        let generateSeconds = Self.seconds(since: generateStart)

        print(
            """
            [Gemma4UnifiedAudio] load \(loadSeconds)s, prepare \(prepareSeconds)s \
            (\(audio.features.dim(1)) audio tokens), generate \(generateSeconds)s
            [Gemma4UnifiedAudio] answer: \(text)
            """)
        #expect(
            text.lowercased().contains("quick brown fox"),
            "expected the transcript to contain the spoken sentence; got: \(text)")
    }

    /// `/usr/bin/say` renders the sentence to an AIFF file; `UserInput.Audio.url` then
    /// decodes it to 16 kHz mono float through AVFoundation.
    private static func synthesizeSpeech(_ sentence: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(
            path: "gemma4-unified-audio-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appending(path: "speech.aiff")
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/say")
        process.arguments = ["-o", url.path, sentence]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw IntegrationTestFailure("say exited with status \(process.terminationStatus)")
        }
        return url
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = start.duration(to: .now)
        return Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
    }
}
