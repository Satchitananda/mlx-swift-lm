// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXGuidedGeneration

/// `minimum` / `maximum` on a `"type": "number"` property, on a
/// byte-fallback vocabulary (token id = byte value, 255 = EOS).
///
/// XGrammar compiles a bounded number to a regex. Through v0.1.30 that regex
/// rejected every fraction in (0, 1) under `"minimum": 0`, and under a bounded
/// `maximum` it admitted fractions past the bound (`30.5` under
/// `"maximum": 30`). v0.2.x fixed both.
@Suite
struct JSONSchemaNumberRangeTests {

    private static let eos = 255

    private static func schema(maximum: Int? = nil) -> String {
        let bound = maximum.map { #", "maximum": \#($0)"# } ?? ""
        return #"""
            {"type": "object",
             "properties": {"startSeconds": {"type": "number", "minimum": 0\#(bound)}},
             "required": ["startSeconds"],
             "additionalProperties": false}
            """#
    }

    private func tokenizer() throws -> GrammarTokenizer {
        let vocab: [String] = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
        return try GrammarTokenizer(
            vocab: vocab, vocabType: .byteFallback, eosTokenId: Int32(Self.eos))
    }

    private func constraint(maximum: Int?, format: JSONSchemaFormat?) throws -> GrammarConstraint {
        let schema = Self.schema(maximum: maximum)
        if let format {
            return try GrammarConstraint(
                tokenizer: try tokenizer(), jsonSchema: schema, format: format)
        }
        return try GrammarConstraint(tokenizer: try tokenizer(), jsonSchema: schema)
    }

    /// Whether the grammar accepts every byte of `text` and can then
    /// terminate.
    private func accepts(_ text: String, maximum: Int? = nil, format: JSONSchemaFormat? = nil)
        throws -> Bool
    {
        let constraint = try constraint(maximum: maximum, format: format)
        for byte in text.utf8 {
            do {
                _ = try constraint.commitToken(Int32(byte))
            } catch GrammarError.commitFailed {
                return false
            }
        }
        let mask = try constraint.computeMask()
        return (UInt32(bitPattern: mask.mask[Self.eos / 32]) >> UInt32(Self.eos % 32)) & 1 == 1
    }

    @Test("minimum 0 accepts fractions below 1", arguments: [nil, JSONSchemaFormat.compact])
    func minimumZeroAcceptsFractionsBelowOne(format: JSONSchemaFormat?) throws {
        for text in [
            #"{"startSeconds":0.5}"#, #"{"startSeconds":0.05}"#, #"{"startSeconds":0}"#,
            #"{"startSeconds":0.0}"#, #"{"startSeconds":12.25}"#, #"{"startSeconds":30}"#,
        ] {
            #expect(try accepts(text, format: format), "\(text)")
        }
        #expect(try !accepts(#"{"startSeconds":-0.5}"#, format: format))
    }

    @Test(
        "a bounded maximum rejects fractions past it", arguments: [nil, JSONSchemaFormat.compact])
    func boundedMaximumRejectsFractionsPastIt(format: JSONSchemaFormat?) throws {
        for text in [
            #"{"startSeconds":30.5}"#, #"{"startSeconds":30.01}"#, #"{"startSeconds":31}"#,
        ] {
            #expect(try !accepts(text, maximum: 30, format: format), "\(text)")
        }
        for text in [
            #"{"startSeconds":30}"#, #"{"startSeconds":29.5}"#, #"{"startSeconds":0.5}"#,
            #"{"startSeconds":0}"#,
        ] {
            #expect(try accepts(text, maximum: 30, format: format), "\(text)")
        }
    }
}
