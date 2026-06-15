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
    public var temperature: Float?
    public var topK: Int?
    public var topP: Float?

    enum CodingKeys: String, CodingKey {
        case eosTokenIds = "eos_token_id"
        case temperature = "temperature"
        case topK       = "top_k"
        case topP       = "top_p"
    }
}
