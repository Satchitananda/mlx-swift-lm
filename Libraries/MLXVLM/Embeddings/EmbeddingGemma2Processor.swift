import Foundation
import MLX
import MLXLMCommon

/// RGB bytes are upright, row-major, and in sRGB. Decoder buffers and MLX arrays
/// never cross the serialized encoder boundary.
public struct EmbeddingGemma2Image: Sendable {
    public let rgb: [UInt8]
    public let width: Int
    public let height: Int
    public init(rgb: [UInt8], width: Int, height: Int) throws {
        guard (1 ... 16_384).contains(width), (1 ... 16_384).contains(height),
            width * height <= 32_000_000, rgb.count == width * height * 3
        else { throw EmbeddingGemma2Error.invalidInput }
        self.rgb = rgb
        self.width = width
        self.height = height
    }
}

public struct EmbeddingGemma2Frame: Sendable {
    public let image: EmbeddingGemma2Image
    public let timestampMillis: Int64
    public init(image: EmbeddingGemma2Image, timestampMillis: Int64) throws {
        guard timestampMillis >= 0 else { throw EmbeddingGemma2Error.invalidInput }
        self.image = image
        self.timestampMillis = timestampMillis
    }
}

public enum EmbeddingGemma2Part: Sendable {
    case text(String)
    case image(EmbeddingGemma2Image)
    /// Visual tokens only. Include an explicit audio part to embed the soundtrack.
    case video([EmbeddingGemma2Frame])
    /// Mono, 16 kHz floating-point PCM, bounded to 30 s per request.
    case audio([Float])
}

/// Pinned processor defaults: images 280, video frames 140 soft tokens, 16×16
/// patches, 3×3 pooling, no image normalization, and no in-prompt video timestamps.
/// Source PTS remains available on the caller's frames and catalog interval.
enum EmbeddingGemma2ImageProcessing {
    static func targetSize(width: Int, height: Int, maxTokens: Int) -> (width: Int, height: Int) {
        let multiple = 48
        let factor = sqrt(Double(maxTokens * multiple * multiple) / Double(width * height))
        var h = Int(floor(Double(height) * factor / Double(multiple)))
        var w = Int(floor(Double(width) * factor / Double(multiple)))
        if h == 0 {
            h = 1
            w = min(max(width / height, 1), maxTokens)
        }
        if w == 0 {
            w = 1
            h = min(max(height / width, 1), maxTokens)
        }
        return (w * multiple, h * multiple)
    }

    private struct Kernel {
        let start: Int
        let weights: [Int]
    }
    private static func kernels(input: Int, output: Int) -> (values: [Kernel], precision: Int) {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        let maxSize = Int(ceil(support)) * 2 + 1
        var rows: [(Int, [Double])] = []
        for i in 0 ..< output {
            let center = scale * (Double(i) + 0.5)
            let start = max(Int(center - support + 0.5), 0)
            let count = min(max(min(Int(center + support + 0.5), input) - start, 0), maxSize)
            var weights = (0 ..< count).map { j -> Double in
                let x = abs((Double(j + start) - center + 0.5) / filterScale)
                if x < 1 { return ((1.5 * x - 2.5) * x) * x + 1 }
                if x < 2 { return ((-0.5 * x + 2.5) * x - 4) * x + 2 }
                return 0
            }
            let total = weights.reduce(0, +)
            if total != 0 { weights = weights.map { $0 / total } }
            rows.append((start, weights))
        }
        // The pinned NumPy/PIL processor uses 22-bit bicubic coefficients and
        // rounds/clips after each separable pass. Match it rather than letting
        // Core Image introduce a different kernel, tone curve, or overshoot.
        // https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c
        let precision = 22
        let factor = Double(1 << precision)
        return (
            rows.map { start, weights in
                Kernel(
                    start: start,
                    weights: weights.map { value in
                        Int(value * factor + (value < 0 ? -0.5 : 0.5))
                    })
            }, precision
        )
    }

    static func resizedRGB(_ image: EmbeddingGemma2Image, width: Int, height: Int) -> [UInt8] {
        var horizontal = image.rgb
        if width != image.width {
            let kernel = kernels(input: image.width, output: width)
            horizontal = [UInt8](repeating: 0, count: width * image.height * 3)
            image.rgb.withUnsafeBufferPointer { source in
                horizontal.withUnsafeMutableBufferPointer { destination in
                    for x in 0 ..< width {
                        let row = kernel.values[x]
                        row.weights.withUnsafeBufferPointer { weights in
                            for y in 0 ..< image.height {
                                var r = 1 << (kernel.precision - 1)
                                var g = r
                                var b = r
                                var j = 0
                                while j < weights.count {
                                    let index = (y * image.width + row.start + j) * 3
                                    let weight = weights[j]
                                    r += Int(source[index]) * weight
                                    g += Int(source[index + 1]) * weight
                                    b += Int(source[index + 2]) * weight
                                    j += 1
                                }
                                let index = (y * width + x) * 3
                                destination[index] = UInt8(clamping: r >> kernel.precision)
                                destination[index + 1] = UInt8(clamping: g >> kernel.precision)
                                destination[index + 2] = UInt8(clamping: b >> kernel.precision)
                            }
                        }
                    }
                }
            }
        }
        guard height != image.height else { return horizontal }
        let kernel = kernels(input: image.height, output: height)
        var output = [UInt8](repeating: 0, count: width * height * 3)
        horizontal.withUnsafeBufferPointer { source in
            output.withUnsafeMutableBufferPointer { destination in
                for y in 0 ..< height {
                    let row = kernel.values[y]
                    row.weights.withUnsafeBufferPointer { weights in
                        for x in 0 ..< width {
                            var r = 1 << (kernel.precision - 1)
                            var g = r
                            var b = r
                            var j = 0
                            while j < weights.count {
                                let index = ((row.start + j) * width + x) * 3
                                let weight = weights[j]
                                r += Int(source[index]) * weight
                                g += Int(source[index + 1]) * weight
                                b += Int(source[index + 2]) * weight
                                j += 1
                            }
                            let index = (y * width + x) * 3
                            destination[index] = UInt8(clamping: r >> kernel.precision)
                            destination[index + 1] = UInt8(clamping: g >> kernel.precision)
                            destination[index + 2] = UInt8(clamping: b >> kernel.precision)
                        }
                    }
                }
            }
        }
        return output
    }

    static func prepare(_ image: EmbeddingGemma2Image, maxTokens: Int) -> (
        pixels: MLXArray, tokens: Int
    ) {
        let size = targetSize(width: image.width, height: image.height, maxTokens: maxTokens)
        let resized = resizedRGB(image, width: size.width, height: size.height)
        let pixels =
            MLXArray(resized, [1, size.height, size.width, 3]).asType(.float32) / Float(255)
        return (pixels.transposed(0, 3, 1, 2), (size.width / 48) * (size.height / 48))
    }
}

struct EmbeddingGemma2Processor {
    let configuration: EmbeddingGemma2Configuration
    let tokenizer: any Tokenizer
    private let reservedTokens: Set<String>
    private let audioExtractor = Gemma4AudioFeatureExtractor()

    init(configuration: EmbeddingGemma2Configuration, tokenizer: any Tokenizer) {
        self.configuration = configuration
        self.tokenizer = tokenizer
        reservedTokens = Set(configuration.controlTokenIDs.compactMap(tokenizer.convertIdToToken))
    }

    func prepare(_ parts: [EmbeddingGemma2Part]) throws -> EmbeddingGemma2Prepared {
        guard !parts.isEmpty, parts.count <= 128 else { throw EmbeddingGemma2Error.invalidInput }
        guard reservedTokens.count == configuration.controlTokenIDs.count,
            reservedTokens.allSatisfy({ !$0.isEmpty })
        else {
            throw EmbeddingGemma2Error.invalidConfiguration
        }
        var rendered = ""
        var textSegment = ""
        var mediaTokens = 2  // BOS/EOS are part of the expanded budget.
        var textBytes = 0
        func reserveMedia(_ count: Int) throws {
            mediaTokens += count + 2
            guard mediaTokens <= 8192 else {
                throw EmbeddingGemma2Error.tokenBudgetExceeded(actual: mediaTokens, limit: 8192)
            }
        }
        var images: [MLXArray] = []
        var frames: [MLXArray] = []
        var audio: [(features: MLXArray, valid: MLXArray)] = []
        for part in parts {
            switch part {
            case .text(let text):
                textBytes += text.utf8.count
                guard !text.isEmpty, text.utf8.count <= 1_000_000,
                    textBytes <= 1_000_000
                else { throw EmbeddingGemma2Error.invalidInput }
                textSegment += text
                guard !reservedTokens.contains(where: textSegment.contains)
                else { throw EmbeddingGemma2Error.invalidInput }
                rendered += text
            case .image(let image):
                textSegment = ""
                let size = EmbeddingGemma2ImageProcessing.targetSize(
                    width: image.width, height: image.height, maxTokens: 280)
                try reserveMedia((size.width / 48) * (size.height / 48))
                let prepared = EmbeddingGemma2ImageProcessing.prepare(image, maxTokens: 280)
                images.append(prepared.pixels)
                rendered += block(
                    begin: configuration.beginImageToken, media: configuration.imageToken,
                    end: configuration.endImageToken, count: prepared.tokens)
            case .video(let video):
                textSegment = ""
                guard !video.isEmpty, video.count <= 32,
                    zip(video, video.dropFirst()).allSatisfy({
                        $0.timestampMillis < $1.timestampMillis
                    })
                else { throw EmbeddingGemma2Error.invalidInput }
                for frame in video {
                    let size = EmbeddingGemma2ImageProcessing.targetSize(
                        width: frame.image.width, height: frame.image.height, maxTokens: 140)
                    try reserveMedia((size.width / 48) * (size.height / 48))
                    let prepared = EmbeddingGemma2ImageProcessing.prepare(
                        frame.image, maxTokens: 140)
                    frames.append(prepared.pixels)
                    rendered += block(
                        begin: configuration.beginImageToken, media: configuration.videoToken,
                        end: configuration.endImageToken, count: prepared.tokens)
                }
            case .audio(let samples):
                textSegment = ""
                guard !samples.isEmpty, samples.count <= 480_000, samples.allSatisfy(\.isFinite)
                else { throw EmbeddingGemma2Error.invalidInput }
                let prepared = audioExtractor(MLXArray(samples))
                let count = Gemma4AudioFeatureExtractor.audioTokenCount(
                    validFrames: sum(prepared.mask.asType(.int32)).item(Int.self))
                guard count > 0 else { throw EmbeddingGemma2Error.invalidInput }
                try reserveMedia(count)
                audio.append((prepared.features, prepared.mask))
                // Dynamic count comes from the actual mel validity mask. In
                // particular, 13 s audio must not be capped to nominal 280.
                rendered += block(
                    begin: configuration.beginAudioToken, media: configuration.audioToken,
                    end: configuration.endAudioToken, count: count)
            }
        }
        let ids = tokenizer.encode(text: rendered, addSpecialTokens: true)
        guard ids.first == configuration.text.bosToken, ids.last == configuration.text.eosToken
        else {
            throw EmbeddingGemma2Error.invalidConfiguration
        }
        guard ids.count <= 8192 else {
            throw EmbeddingGemma2Error.tokenBudgetExceeded(actual: ids.count, limit: 8192)
        }
        let tokens = MLXArray(ids, [1, ids.count])
        return EmbeddingGemma2Prepared(
            tokens: tokens, valid: MLXArray.ones(tokens.shape).asType(.bool),
            images: images, videoFrames: frames, audio: audio)
    }

    /// Right padding preserves each sample's positions and excludes padding
    /// from both attention keys and pooling. Media remains in row-major order.
    static func batch(_ rows: [EmbeddingGemma2Prepared], padToken: Int) throws
        -> EmbeddingGemma2Prepared
    {
        guard !rows.isEmpty, rows.count <= 4,
            rows.allSatisfy({
                $0.tokens.ndim == 2 && $0.tokens.dim(0) == 1
                    && $0.tokens.shape == $0.valid.shape && $0.tokens.dim(1) > 0
            })
        else { throw EmbeddingGemma2Error.invalidInput }
        if rows.count == 1 { return rows[0] }
        let length = rows.map { $0.tokens.dim(1) }.max()!
        guard length <= 8192 else {
            throw EmbeddingGemma2Error.tokenBudgetExceeded(actual: length, limit: 8192)
        }
        let tokens = rows.map { row in
            let padding = length - row.tokens.dim(1)
            return padding == 0
                ? row.tokens
                : concatenated(
                    [
                        row.tokens,
                        MLXArray.full([1, padding], values: MLXArray(padToken)).asType(
                            row.tokens.dtype),
                    ], axis: 1)
        }
        let masks = rows.map { row in
            let padding = length - row.valid.dim(1)
            return padding == 0
                ? row.valid
                : concatenated(
                    [
                        row.valid, MLXArray.zeros([1, padding]).asType(.bool),
                    ], axis: 1)
        }
        return EmbeddingGemma2Prepared(
            tokens: concatenated(tokens, axis: 0),
            valid: concatenated(masks, axis: 0), images: rows.flatMap(\.images),
            videoFrames: rows.flatMap(\.videoFrames), audio: rows.flatMap(\.audio))
    }

    private func block(begin: Int, media: Int, end: Int, count: Int) -> String {
        tokenizer.convertIdToToken(begin)!
            + String(repeating: tokenizer.convertIdToToken(media)!, count: count)
            + tokenizer.convertIdToToken(end)!
    }
}
