// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXGuidedGeneration

/// Acceptance of whole JSON strings under each `JSONSchemaFormat`, on a
/// byte-fallback vocabulary (token id = byte value, 255 = EOS).
@Suite
struct JSONSchemaFormatTests {

    private static let schema = #"""
        {"type": "object",
         "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
         "required": ["a", "b"]}
        """#

    private static let compactText = #"{"a":1,"b":"x"}"#
    private static let spacedText = #"{"a": 1, "b": "x"}"#
    private static let looseText = #"{ "a" : 1 ,"b":"x" }"#
    private static let indent2Text = "{\n  \"a\": 1,\n  \"b\": \"x\"\n}"
    private static let indent4Text = "{\n    \"a\": 1,\n    \"b\": \"x\"\n}"
    private static let extraPropertyText = #"{"a":1,"b":"x","c":2}"#

    private static let eos = 255

    private func tokenizer() throws -> GrammarTokenizer {
        let vocab: [String] = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
        return try GrammarTokenizer(
            vocab: vocab, vocabType: .byteFallback, eosTokenId: Int32(Self.eos))
    }

    private func constraint(_ format: JSONSchemaFormat?) throws -> GrammarConstraint {
        if let format {
            return try GrammarConstraint(
                tokenizer: try tokenizer(), jsonSchema: Self.schema, format: format)
        }
        return try GrammarConstraint(tokenizer: try tokenizer(), jsonSchema: Self.schema)
    }

    /// Whether `constraint` accepts every byte of `text` and can then
    /// terminate.
    private func accepts(_ constraint: GrammarConstraint, _ text: String) throws -> Bool {
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

    private func accepts(_ format: JSONSchemaFormat?, _ text: String) throws -> Bool {
        try accepts(try constraint(format), text)
    }

    @Test("the default format is XGrammar's defaults and matches the format-less init")
    func defaultFormatMatchesTheFormatlessInit() throws {
        #expect(
            JSONSchemaFormat.default
                == JSONSchemaFormat(
                    anyWhitespace: true, indent: nil, separators: nil, strictMode: true))
        for text in [
            Self.compactText, Self.spacedText, Self.looseText, Self.indent2Text,
            Self.extraPropertyText,
        ] {
            #expect(try accepts(.default, text) == accepts(nil, text), "\(text)")
        }
    }

    @Test("compact strict rejects whitespace between tokens and extra properties")
    func compactStrictRejectsWhitespaceAndExtraProperties() throws {
        #expect(
            JSONSchemaFormat.compact
                == JSONSchemaFormat(
                    anyWhitespace: false, separators: .init(item: ",", key: ":"),
                    strictMode: true))
        #expect(try accepts(.compact, Self.compactText))
        for text in [
            Self.spacedText, Self.looseText, Self.indent2Text, #"{"a":1, "b":"x"}"#,
            #"{"a": 1,"b":"x"}"#, Self.extraPropertyText,
        ] {
            #expect(try !accepts(.compact, text), "\(text)")
        }
    }

    @Test("any whitespace accepts spacing and newlines but stays strict")
    func anyWhitespaceAcceptsSpacing() throws {
        let format = JSONSchemaFormat(anyWhitespace: true, strictMode: true)
        for text in [Self.compactText, Self.spacedText, Self.looseText, Self.indent2Text] {
            #expect(try accepts(format, text), "\(text)")
        }
        #expect(try !accepts(format, Self.extraPropertyText))
    }

    @Test("indent 2 accepts only the two-space indented form")
    func indent2AcceptsTheIndentedForm() throws {
        let format = JSONSchemaFormat(anyWhitespace: false, indent: 2, strictMode: true)
        #expect(try accepts(format, Self.indent2Text))
        for text in [Self.compactText, Self.spacedText, Self.indent4Text, Self.extraPropertyText] {
            #expect(try !accepts(format, text), "\(text)")
        }
    }

    @Test("without any whitespace or separators, XGrammar requires a space after , and :")
    func noWhitespaceWithoutSeparatorsIsNotCompact() throws {
        let format = JSONSchemaFormat(anyWhitespace: false)
        #expect(try accepts(format, Self.spacedText))
        #expect(try !accepts(format, Self.compactText))
    }

    @Test("custom separators are required exactly")
    func customSeparatorsAreRequired() throws {
        let format = JSONSchemaFormat(
            anyWhitespace: false, separators: .init(item: " , ", key: " = "))
        // `key` replaces the colon itself.
        #expect(try accepts(format, #"{"a" = 1 , "b" = "x"}"#))
        #expect(try !accepts(format, Self.compactText))
    }

    @Test("non-strict mode accepts properties the schema does not declare")
    func nonStrictAcceptsExtraProperties() throws {
        let format = JSONSchemaFormat(
            anyWhitespace: false, separators: .init(item: ",", key: ":"), strictMode: false)
        #expect(try accepts(format, Self.extraPropertyText))
        #expect(try accepts(format, Self.compactText))
    }

    @Test("a fresh matcher keeps the compiled format")
    func freshMatcherKeepsTheFormat() throws {
        let template = try constraint(.compact)
        let first = try template.freshMatcher()
        #expect(try accepts(first, Self.compactText))
        let second = try template.freshMatcher()
        #expect(try !accepts(second, Self.spacedText))
    }

    @Test("invalid format values throw before compiling")
    func invalidFormatValuesThrow() throws {
        let negativeIndent = JSONSchemaFormat(anyWhitespace: false, indent: -1)
        #expect(throws: GrammarError.self) { _ = try constraint(negativeIndent) }
        let nulSeparator = JSONSchemaFormat(
            anyWhitespace: false, separators: .init(item: ",\0", key: ":"))
        #expect(throws: GrammarError.self) { _ = try constraint(nulSeparator) }
    }
}
