import Foundation

/// The encoder checkpoint configuration. This is distinct from Gemma 4's causal decoder.
public struct EmbeddingGemma2Configuration: Decodable, Sendable {
    public struct Text: Decodable, Sendable {
        public struct Layer: Decodable, Sendable {
            let headDim: Int?
            let attentionHeads: Int?
            let keyValueHeads: Int?
            enum CodingKeys: String, CodingKey {
                case headDim = "head_dim"
                case attentionHeads = "num_attention_heads"
                case keyValueHeads = "num_key_value_heads"
            }
        }
        public struct Rope: Decodable, Sendable {
            let theta: Float
            let type: String
            enum CodingKeys: String, CodingKey {
                case theta = "rope_theta"
                case type = "rope_type"
            }
        }
        let vocabularySize: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let layers: Int
        let attentionHeads: Int
        let keyValueHeads: Int
        let headDim: Int
        let perLayerWidth: Int
        public let embeddingDimensions: Int
        let eps: Float
        let slidingWindow: Int
        let attentionBias: Bool
        let padToken: Int
        let bosToken: Int
        let eosToken: Int
        let layerTypes: [String]
        let layerOverrides: [String: Layer]
        let rope: [String: Rope]

        enum CodingKeys: String, CodingKey {
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case layers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case keyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case perLayerWidth = "hidden_size_per_layer_input"
            case embeddingDimensions = "embedding_dim"
            case eps = "rms_norm_eps"
            case slidingWindow = "sliding_window"
            case attentionBias = "attention_bias"
            case padToken = "pad_token_id"
            case bosToken = "bos_token_id"
            case eosToken = "eos_token_id"
            case layerTypes = "layer_types"
            case layerOverrides = "per_layer_config"
            case rope = "rope_parameters"
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            layers = try c.decode(Int.self, forKey: .layers)
            attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
            keyValueHeads = try c.decode(Int.self, forKey: .keyValueHeads)
            headDim = try c.decode(Int.self, forKey: .headDim)
            perLayerWidth = try c.decodeIfPresent(Int.self, forKey: .perLayerWidth) ?? 0
            embeddingDimensions = try c.decode(Int.self, forKey: .embeddingDimensions)
            eps = try c.decodeIfPresent(Float.self, forKey: .eps) ?? 1e-6
            slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            padToken = try c.decodeIfPresent(Int.self, forKey: .padToken) ?? 0
            bosToken = try c.decodeIfPresent(Int.self, forKey: .bosToken) ?? 2
            eosToken = try c.decodeIfPresent(Int.self, forKey: .eosToken) ?? 1
            layerTypes = try c.decode([String].self, forKey: .layerTypes)
            let overrides =
                try c.decodeIfPresent([String: Layer].self, forKey: .layerOverrides) ?? [:]
            var normalized: [String: Layer] = [:]
            for (key, value) in overrides {
                guard let layer = Int(key), (0 ..< layers).contains(layer) else {
                    throw EmbeddingGemma2Error.invalidConfiguration
                }
                let key = String(format: "%02d", layer)
                guard normalized[key] == nil else {
                    throw EmbeddingGemma2Error.invalidConfiguration
                }
                normalized[key] = value
            }
            layerOverrides = normalized
            rope = try c.decode([String: Rope].self, forKey: .rope)
            guard vocabularySize > 0, hiddenSize > 0, intermediateSize > 0, layers > 0,
                attentionHeads > 0, keyValueHeads > 0, headDim > 0, headDim.isMultiple(of: 2),
                perLayerWidth >= 0, embeddingDimensions > 0, eps > 0, slidingWindow >= 0,
                [padToken, bosToken, eosToken].allSatisfy({ (0 ..< vocabularySize).contains($0) }),
                layerTypes.count == layers,
                layerTypes.allSatisfy({ ["sliding_attention", "full_attention"].contains($0) }),
                layerTypes.allSatisfy({ rope[$0]?.type == "default" && (rope[$0]?.theta ?? 0) > 0 })
            else { throw EmbeddingGemma2Error.invalidConfiguration }
            for i in 0 ..< layers {
                let override = layerOverrides[String(format: "%02d", i)]
                let heads = override?.attentionHeads ?? attentionHeads
                let kvHeads = override?.keyValueHeads ?? keyValueHeads
                let dimension = override?.headDim ?? headDim
                guard heads > 0, kvHeads > 0, heads.isMultiple(of: kvHeads),
                    dimension > 0, dimension.isMultiple(of: 2)
                else { throw EmbeddingGemma2Error.invalidConfiguration }
            }
        }
    }

    let modelType: String
    public let text: Text
    let vision: Gemma4VisionConfiguration?
    let audio: Gemma4AudioConfiguration?
    let imageToken: Int
    let audioToken: Int
    let videoToken: Int
    let beginImageToken: Int
    let endImageToken: Int
    let beginAudioToken: Int
    let endAudioToken: Int

    var controlTokenIDs: [Int] {
        [
            text.bosToken, text.eosToken, imageToken, videoToken, audioToken,
            beginImageToken, endImageToken, beginAudioToken, endAudioToken,
        ]
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case vision = "vision_config"
        case audio = "audio_config"
        case imageToken = "image_token_id"
        case audioToken = "audio_token_id"
        case videoToken = "video_token_id"
        case beginImageToken = "boi_token_id"
        case endImageToken = "eoi_token_id"
        case beginAudioToken = "boa_token_id"
        case endAudioToken = "eoa_token_id"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        guard modelType == "embedding_gemma2" else {
            throw EmbeddingGemma2Error.invalidConfiguration
        }
        text = try c.decode(Text.self, forKey: .text)
        vision = try c.decodeIfPresent(Gemma4VisionConfiguration.self, forKey: .vision)
        audio = try c.decodeIfPresent(Gemma4AudioConfiguration.self, forKey: .audio)
        imageToken = try c.decodeIfPresent(Int.self, forKey: .imageToken) ?? 258_880
        audioToken = try c.decodeIfPresent(Int.self, forKey: .audioToken) ?? 258_881
        videoToken = try c.decodeIfPresent(Int.self, forKey: .videoToken) ?? 258_884
        beginImageToken = try c.decodeIfPresent(Int.self, forKey: .beginImageToken) ?? 255_999
        endImageToken = try c.decodeIfPresent(Int.self, forKey: .endImageToken) ?? 258_882
        beginAudioToken = try c.decodeIfPresent(Int.self, forKey: .beginAudioToken) ?? 256_000
        endAudioToken = try c.decodeIfPresent(Int.self, forKey: .endAudioToken) ?? 258_883
        guard controlTokenIDs.allSatisfy({ (0 ..< text.vocabularySize).contains($0) }) else {
            throw EmbeddingGemma2Error.invalidConfiguration
        }
    }
}

public enum EmbeddingGemma2Error: Error, Sendable {
    case invalidConfiguration
    case unsupportedPrecision
    case invalidInput
    case tokenBudgetExceeded(actual: Int, limit: Int)
    case missingTower
    case mediaTokenMismatch(expected: Int, actual: Int)
    case invalidOutput
}

public enum EmbeddingGemma2Tower: String, Codable, Sendable, Hashable {
    case vision, audio
}
