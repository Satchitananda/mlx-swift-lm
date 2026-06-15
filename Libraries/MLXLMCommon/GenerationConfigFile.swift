// Copyright © 2024 Apple Inc.

import Foundation

/// JSON wrapper for `generation_config.json` file.
///
/// This file can override values from `config.json`, particularly `eos_token_id`.
/// Following mlx-lm Python behavior, if `generation_config.json` exists and contains
/// `eos_token_id`, it takes precedence over the value in `config.json`.
///
/// Also surfaces recommended sampling parameters (`temperature`, `top_k`, `top_p`)
/// so callers can apply model-card defaults instead of hardcoded values.
public struct GenerationConfigFile: Codable, Sendable {
    public var eosTokenIds: IntOrIntArray?
    public var stopStrings: Set<String>
    public var temperature: Float?
    public var topK: Int?
    public var topP: Float?

    enum CodingKeys: String, CodingKey {
        case eosTokenIds = "eos_token_id"
        case stopStrings = "stop_strings"
        case stop
        case temperature = "temperature"
        case topK = "top_k"
        case topP = "top_p"
    }

    public init(eosTokenIds: IntOrIntArray? = nil, stopStrings: Set<String> = []) {
        self.eosTokenIds = eosTokenIds
        self.stopStrings = stopStrings
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        eosTokenIds = try container.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenIds)
        temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        topP = try container.decodeIfPresent(Float.self, forKey: .topP)

        stopStrings = []
        stopStrings.formUnion(Self.decodeStringSet(from: container, forKey: .stopStrings))
        stopStrings.formUnion(Self.decodeStringSet(from: container, forKey: .stop))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(eosTokenIds, forKey: .eosTokenIds)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(topK, forKey: .topK)
        try container.encodeIfPresent(topP, forKey: .topP)
        if !stopStrings.isEmpty {
            try container.encode(stopStrings.sorted(), forKey: .stopStrings)
        }
    }

    private static func decodeStringSet(
        from container: KeyedDecodingContainer<CodingKeys>,
        forKey key: CodingKeys
    ) -> Set<String> {
        if let values = try? container.decode([String].self, forKey: key) {
            return Set(values)
        }
        if let value = try? container.decode(String.self, forKey: key) {
            return [value]
        }
        return []
    }
}
