import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Explicit local benchmark only. Uses exact reference-processor tensors,
/// bypassing the BF16-only public container without changing its contract.
@Suite(.serialized) struct EmbeddingGemma2RuntimeBenchmark {
    private struct Manifest: Decodable {
        struct Sample: Decodable {
            let id: String
            let lane: String
            let file: String
        }
        let samples: [Sample]
    }
    private struct Trial: Codable {
        let batch: Int
        let repeatIndex: Int
        let lanes: [String: Double]
        let seconds: Double
        let actualBatchCounts: [Int: Int]
        let vectors: [[Float]]
    }
    private struct Report: Codable {
        let scope: String
        let loadSeconds: Double
        let ids: [String]
        let lanes: [String]
        let metalPeakBytes: Int
        let quantizedLayers: Int
        let trials: [Trial]
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["EMBEDDINGGEMMA2_RUNTIME_BENCHMARK"] != nil))
    func preparedTensorBenchmark() throws {
        let environment = ProcessInfo.processInfo.environment
        let manifestURL = URL(
            filePath: try #require(environment["EMBEDDINGGEMMA2_RUNTIME_BENCHMARK"]))
        let directory = URL(filePath: try #require(environment["EMBEDDINGGEMMA2_RUNTIME_MODEL"]))
        let output = URL(filePath: try #require(environment["EMBEDDINGGEMMA2_RUNTIME_OUTPUT"]))
        try #require(!FileManager.default.fileExists(atPath: output.path))
        try FileManager.default.createDirectory(
            at: output, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        if let library = environment["EMBEDDINGGEMMA2_RUNTIME_METALLIB"] {
            GPU.metallib = URL(filePath: library)
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        let samples = manifest.samples
        try #require(!samples.isEmpty && samples.count <= 500)
        try #require(Set(samples.map(\.id)).count == samples.count)
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        let c = try JSONDecoder().decode(EmbeddingGemma2Configuration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let loadStart = ContinuousClock.now
        let model = try EmbeddingGemma2Model(c, towers: [.vision, .audio])
        let quantization = base.perLayerQuantization?.quantization
        if let quantization {
            try #require(
                quantization.bits == 8 && quantization.groupSize == 64
                    && quantization.mode == .affine)
        }
        try MLXLMCommon.loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        let loadSeconds = seconds(since: loadStart)
        let quantizedLayers = model.leafModules().flattened().filter { $0.1 is any Quantized }.count
        try #require(quantization == nil || quantizedLayers > 0)

        func prepared(_ sample: Manifest.Sample) throws -> EmbeddingGemma2Prepared {
            let arrays = try loadArrays(url: URL(filePath: sample.file))
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
            return .init(
                tokens: try #require(arrays["tokens"]), valid: try #require(arrays["valid"]),
                images: media("image"), videoFrames: media("video"), audio: audio)
        }
        let laneOrder = ["image", "video", "audio", "joint", "text"]
        for lane in laneOrder {
            let sample = try #require(samples.first(where: { $0.lane == lane }))
            let warm = try model(prepared(sample), visionBatchSize: 2)
            eval(warm)
        }
        var trials: [Trial] = []
        for batch in [2, 1, 4] {
            for repeatIndex in 0 ..< 3 {
                var vectors = [String: [Float]]()
                var laneSeconds = [String: Double]()
                var actualBatchCounts = [Int: Int]()
                for lane in laneOrder {
                    let selected = samples.filter { $0.lane == lane }
                    var elapsed: Double = 0
                    for start in stride(from: 0, to: selected.count, by: batch) {
                        let chunk = Array(selected[start ..< min(start + batch, selected.count)])
                        // Disk reads are outside the measured prepared-tensor interval.
                        let rows = try chunk.map(prepared)
                        var index = 0
                        while index < rows.count {
                            var end = index + 1
                            while end < rows.count,
                                rows[end].tokens.dim(1) == rows[index].tokens.dim(1),
                                rows[index].tokens.dim(1) * (end - index + 1) <= 8192
                            { end += 1 }
                            let stamp = ContinuousClock.now
                            let input = try EmbeddingGemma2Processor.batch(
                                Array(rows[index ..< end]), padToken: c.text.padToken)
                            let embedded = try model(input, visionBatchSize: 2)
                            eval(embedded)
                            for offset in index ..< end {
                                let vector = embedded[offset - index].asArray(Float.self)
                                #expect(vector.count == 768 && vector.allSatisfy(\.isFinite))
                                #expect(
                                    abs(vector.reduce(0.0) { $0 + Double($1) * Double($1) } - 1)
                                        <= 0.002)
                                vectors[chunk[offset].id] = vector
                            }
                            elapsed += seconds(since: stamp)
                            actualBatchCounts[end - index, default: 0] += 1
                            index = end
                        }
                    }
                    laneSeconds[lane] = elapsed
                }
                let trial = Trial(
                    batch: batch, repeatIndex: repeatIndex, lanes: laneSeconds,
                    seconds: laneSeconds.values.reduce(0, +), actualBatchCounts: actualBatchCounts,
                    vectors: try samples.map { try #require(vectors[$0.id]) })
                trials.append(trial)
                print(
                    "Native prepared benchmark batch=\(batch) repeat=\(repeatIndex) seconds=\(trial.seconds)"
                )
            }
        }
        let report = Report(
            scope: "prepared_native_forward_collation_and_output_only", loadSeconds: loadSeconds,
            ids: samples.map(\.id), lanes: samples.map(\.lane), metalPeakBytes: Memory.peakMemory,
            quantizedLayers: quantizedLayers, trials: trials)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(report).write(
            to: output.appending(path: "report.private.json"), options: .atomic)
    }

    private func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}
