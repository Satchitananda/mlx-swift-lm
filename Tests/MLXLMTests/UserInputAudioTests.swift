// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

#if canImport(AVFoundation)
import AVFoundation
#endif

@Suite("UserInput.Audio source")
struct UserInputAudioTests {

    #if canImport(AVFoundation)
    /// The reader must deliver host-endian float32 whatever the file's byte order: a
    /// big-endian AIFF read with little-endian settings came back as byte-swapped garbage
    /// (non-finite samples with 1e37 magnitudes) and every audio model heard noise.
    @Test("A big-endian AIFF decodes to the same samples as a little-endian WAV")
    func bigEndianSourceDecodesLikeLittleEndian() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(
            path: "user-input-audio-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let sampleRate = 16_000.0
        let count = 4_000
        let tone = (0 ..< count).map {
            Float(0.5 * sin(2 * Double.pi * 440 * Double($0) / sampleRate))
        }
        func write(_ name: String, bigEndian: Bool) throws -> URL {
            let url = directory.appending(path: name)
            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsBigEndianKey: bigEndian,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsNonInterleaved: false,
            ]
            // Scoped so the file is closed (flushed) before it is read back.
            let file = try AVAudioFile(
                forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32,
                interleaved: false)
            guard
                let buffer = AVAudioPCMBuffer(
                    pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(count)),
                let channel = buffer.floatChannelData?[0]
            else { throw CocoaError(.fileWriteUnknown) }
            buffer.frameLength = AVAudioFrameCount(count)
            tone.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: count) }
            try file.write(from: buffer)
            return url
        }

        var processing = UserInput.AudioProcessing()
        processing.sampleRate = sampleRate
        let aiff = try await UserInput.Audio.url(try write("tone.aiff", bigEndian: true))
            .asMLXArray(processing: processing).asArray(Float.self)
        let wav = try await UserInput.Audio.url(try write("tone.wav", bigEndian: false))
            .asMLXArray(processing: processing).asArray(Float.self)

        #expect(aiff.count == count)
        #expect(wav.count == count)
        let finite = aiff.allSatisfy(\.isFinite)
        #expect(finite, "big-endian source decoded to non-finite samples")
        let tolerance: Float = 2.0 / 32_768  // 16-bit quantization
        #expect(zip(aiff, tone).allSatisfy { abs($0 - $1) <= tolerance })
        #expect(zip(wav, tone).allSatisfy { abs($0 - $1) <= tolerance })
        #expect(zip(aiff, wav).allSatisfy { abs($0 - $1) <= tolerance })
    }
    #endif

    @Test("The url factory makes a url source")
    func urlFactoryMakesAURLSource() {
        let audio = UserInput.Audio.url(URL(fileURLWithPath: "/tmp/example.wav"))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/example.wav")
    }

    @Test("The array factory makes an array source")
    func arrayFactoryMakesAnArraySource() {
        let audio = UserInput.Audio.array(MLXArray([0.25, -0.5, 1] as [Float]))
        guard case .array(let array) = audio.source else {
            Issue.record("expected an array source")
            return
        }
        #expect(array.asArray(Float.self) == [0.25, -0.5, 1])
    }

    @Test("A factory works as an implicit member")
    func factoryWorksAsAnImplicitMember() {
        let audio: UserInput.Audio = .url(URL(fileURLWithPath: "/tmp/implicit.wav"))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/implicit.wav")
    }

    @Test("A factory can be passed where a function of one argument is expected")
    func factoryWorksAsAFunctionValue() {
        let urls = [URL(fileURLWithPath: "/tmp/a.wav"), URL(fileURLWithPath: "/tmp/b.wav")]
        let audios = urls.map(UserInput.Audio.url)
        let paths = audios.compactMap { audio -> String? in
            guard case .url(let url) = audio.source else { return nil }
            return url.path
        }
        #expect(paths == ["/tmp/a.wav", "/tmp/b.wav"])
    }

    @Test("A factory works as a function value without a type context")
    func factoryWorksWithoutATypeContext() {
        let make = UserInput.Audio.url
        let audio = make(URL(fileURLWithPath: "/tmp/bare.wav"))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/bare.wav")
    }

    @Test("The initializer stores the source")
    func initializerStoresTheSource() {
        let audio = UserInput.Audio(source: .url(URL(fileURLWithPath: "/tmp/init.wav")))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/init.wav")
    }
}
