// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXGuidedGeneration

/// Token ids of the synthetic vocabulary: 256 byte-fallback pieces, then two
/// stop tokens.
private enum Vocab {
    static let eos = 256
    static let endOfTurn = 257
    static let size = 258
    static let quote = 0x22
    static let letterA = 0x61

    static let pieces: [String] =
        (0 ..< 256).map { String(format: "<0x%02X>", $0) } + ["<eos>", "<end_of_turn>"]
}

/// Byte-level host tokenizer over `Vocab`: ids below 256 are bytes.
private struct StopVocabTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.filter { $0 < 256 }.map(UInt8.init), encoding: .utf8) ?? ""
    }
    func convertTokenToId(_ token: String) -> Int? {
        if let index = Vocab.pieces.firstIndex(of: token), index >= 256 { return index }
        return token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil
    }
    func convertIdToToken(_ id: Int) -> String? {
        id >= 0 && id < Vocab.size ? Vocab.pieces[id] : nil
    }
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct UnusedProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        fatalError("the test supplies the prepared input")
    }
}

/// Prefers `<end_of_turn>`, then the quote character, at every step.
private final class PrefersEndOfTurnModel: Module, LanguageModel {
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let count = input.tokens.size
        var row = (0 ..< Vocab.size).map { -Float($0) / 1000 }
        row[Vocab.endOfTurn] = 20
        row[Vocab.quote] = 10
        let rows = Array(repeating: row, count: count).flatMap { $0 }
        return LMOutput(logits: MLXArray(rows).reshaped(1, count, Vocab.size))
    }
}

/// Returns the scripted tokens in order, one per step.
private final class ScriptedSampler: LogitSampler {
    private var tokens: [Int]
    init(_ tokens: [Int]) { self.tokens = tokens }
    func sample(logits: MLXArray) -> MLXArray {
        MLXArray([UInt32(tokens.isEmpty ? Vocab.eos : tokens.removeFirst())])
    }
}

@Suite(.serialized)
struct MultiStopGrammarTokenizerTests {

    private static let stringSchema = #"{"type": "string"}"#

    private func allows(_ mask: MaskResult, _ token: Int) -> Bool {
        (UInt32(bitPattern: mask.mask[token / 32]) >> UInt32(token % 32)) & 1 == 1
    }

    private func tokenizer(stops: [Int32]) throws -> GrammarTokenizer {
        try GrammarTokenizer(vocab: Vocab.pieces, vocabType: .byteFallback, stopTokenIds: stops)
    }

    // MARK: Masking

    @Test("every registered stop token is masked until the grammar can terminate")
    func everyStopIsMaskedUntilTheGrammarCanTerminate() throws {
        let constraint = try GrammarConstraint(
            tokenizer: try tokenizer(stops: [Int32(Vocab.eos), Int32(Vocab.endOfTurn)]),
            jsonSchema: Self.stringSchema)
        #expect(constraint.stopTokenIds == [Int32(Vocab.eos), Int32(Vocab.endOfTurn)])

        _ = try constraint.commitToken(Int32(Vocab.quote))
        _ = try constraint.commitToken(Int32(Vocab.letterA))
        let inside = try constraint.computeMask()
        #expect(inside.needsApply)
        #expect(!allows(inside, Vocab.eos))
        #expect(!allows(inside, Vocab.endOfTurn))
        #expect(throws: GrammarError.self) {
            _ = try constraint.commitToken(Int32(Vocab.endOfTurn))
        }

        _ = try constraint.commitToken(Int32(Vocab.quote))
        let complete = try constraint.computeMask()
        #expect(allows(complete, Vocab.eos))
        #expect(allows(complete, Vocab.endOfTurn))
        #expect(try constraint.commitToken(Int32(Vocab.endOfTurn)).isTerminated)
    }

    @Test("a fresh matcher keeps the full stop set")
    func freshMatcherKeepsTheFullStopSet() throws {
        let template = try GrammarConstraint(
            tokenizer: try tokenizer(stops: [Int32(Vocab.eos), Int32(Vocab.endOfTurn)]),
            jsonSchema: Self.stringSchema)
        let fresh = try template.freshMatcher()
        #expect(fresh.stopTokenIds == template.stopTokenIds)

        _ = try fresh.commitToken(Int32(Vocab.quote))
        let inside = try fresh.computeMask()
        #expect(!allows(inside, Vocab.eos))
        #expect(!allows(inside, Vocab.endOfTurn))
    }

    @Test("the single-EOS initializer registers only that token")
    func eosInitializerRegistersOneStop() throws {
        let grammarTokenizer = try GrammarTokenizer(
            vocab: Vocab.pieces, vocabType: .byteFallback, eosTokenId: Int32(Vocab.eos))
        #expect(grammarTokenizer.stopTokenIds == [Int32(Vocab.eos)])

        // The unregistered `<end_of_turn>` is grammar text, so it stays
        // allowed inside a string.
        let constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer, jsonSchema: Self.stringSchema)
        _ = try constraint.commitToken(Int32(Vocab.quote))
        let inside = try constraint.computeMask()
        #expect(!allows(inside, Vocab.eos))
        #expect(allows(inside, Vocab.endOfTurn))
    }

    // MARK: Validation

    @Test("invalid stop sets throw typed errors")
    func invalidStopSetsThrow() {
        #expect(throws: GrammarTokenizerError.emptyStopTokenIds) {
            _ = try tokenizer(stops: [])
        }
        #expect(
            throws: GrammarTokenizerError.stopTokenIdOutOfRange(
                Int32(Vocab.size), vocabSize: Vocab.size)
        ) {
            _ = try tokenizer(stops: [Int32(Vocab.eos), Int32(Vocab.size)])
        }
        #expect(throws: GrammarTokenizerError.stopTokenIdOutOfRange(-1, vocabSize: Vocab.size)) {
            _ = try tokenizer(stops: [-1])
        }
        #expect(throws: GrammarTokenizerError.duplicateStopTokenId(Int32(Vocab.eos))) {
            _ = try tokenizer(stops: [Int32(Vocab.eos), Int32(Vocab.endOfTurn), Int32(Vocab.eos)])
        }
        #expect(
            throws: GrammarTokenizerError.stopTokenIdOutOfRange(
                Int32(Vocab.size), vocabSize: Vocab.size)
        ) {
            _ = try GrammarTokenizer(
                vocab: Vocab.pieces, vocabType: .byteFallback, eosTokenId: Int32(Vocab.size))
        }
    }

    // MARK: Generation loop

    private func generate(
        constraint: GrammarConstraint,
        sampler: (any LogitSampler)? = nil
    ) throws -> String {
        let context = ModelContext(
            configuration: ModelConfiguration(
                id: "prefers-end-of-turn",
                eosTokenIds: [Vocab.eos, Vocab.endOfTurn]),
            model: PrefersEndOfTurnModel(), processor: UnusedProcessor(),
            tokenizer: StopVocabTokenizer())
        var output = ""
        _ = try GuidedGenerationLoop.run(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])), context: context,
            constraint: constraint, maxTokens: 16, vocabSize: Vocab.size,
            logitSampler: sampler
        ) {
            output += $0
            return true
        }
        return output
    }

    private func eosOnlyTokenizer() throws -> GrammarTokenizer {
        try GrammarTokenizer(
            vocab: Vocab.pieces, vocabType: .byteFallback, eosTokenId: Int32(Vocab.eos))
    }

    @Test("the loop cannot stop on a registered extra stop before the grammar completes")
    func loopMasksTheExtraStopInsideTheValue() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let constraint = try GrammarConstraint(
                tokenizer: try tokenizer(stops: [Int32(Vocab.eos), Int32(Vocab.endOfTurn)]),
                jsonSchema: Self.stringSchema)
            // `<end_of_turn>` is the model's favourite, but it is masked until
            // the string closes.
            #expect(try generate(constraint: constraint) == #""""#)
        }
    }

    @Test("an unregistered stop inside the value is premature, not success")
    func unregisteredStopInsideTheValueIsPremature() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let constraint = try GrammarConstraint(
                tokenizer: try eosOnlyTokenizer(), jsonSchema: Self.stringSchema)
            #expect(throws: GuidedGenerationError.prematureEOS) {
                _ = try generate(
                    constraint: constraint,
                    sampler: ScriptedSampler([Vocab.quote, Vocab.endOfTurn]))
            }
        }
    }

    @Test("an unregistered stop where the grammar can terminate still ends the output")
    func unregisteredStopWhereTheGrammarCanTerminateSucceeds() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            // After "a" the grammar can terminate (EOS is allowed), and the
            // unregistered `<end_of_turn>` is also allowed as its literal text.
            let constraint = try GrammarConstraint(
                tokenizer: try eosOnlyTokenizer(), grammar: #"root ::= "a" "<end_of_turn>"?"#)
            let output = try generate(
                constraint: constraint,
                sampler: ScriptedSampler([Vocab.letterA, Vocab.endOfTurn]))
            #expect(output == "a")
        }
    }
}
