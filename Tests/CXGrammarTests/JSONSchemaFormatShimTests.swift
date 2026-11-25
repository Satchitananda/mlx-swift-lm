// Copyright © 2026 Apple Inc.
//
// Direct C-API tests for `xg_compile_json_schema_with_format`'s argument
// contract. Grammar behaviour under each format is covered through the
// Swift bridge in MLXGuidedGenerationTests/JSONSchemaFormatTests.

import Foundation
import MLXCXGrammar
import Testing

@Suite
struct JSONSchemaFormatShimTests {

    private static let schema = #"{"type":"object","properties":{"a":{"type":"integer"}}}"#

    /// Runs `body` with a compiler over a small raw vocabulary.
    private func withCompiler(_ body: (OpaquePointer?) throws -> Void) throws {
        let vocab = ["{", "}", "\"", ":", ",", " ", "a", "1", "<eos>"]
        let cStrings = vocab.map { $0.utf8CString }
        var info: OpaquePointer?
        let stops: [Int32] = [Int32(vocab.count - 1)]
        let status: XGStatus = cStrings.withUnsafeCStringArray { pointers in
            stops.withUnsafeBufferPointer { stopBuf in
                xg_tokenizer_info_new(
                    pointers.baseAddress, pointers.count, XG_VOCAB_TYPE_RAW,
                    stopBuf.baseAddress, stopBuf.count, &info)
            }
        }
        try #require(status == XG_OK)
        defer { xg_tokenizer_info_free(info) }

        var compiler: OpaquePointer?
        try #require(xg_grammar_compiler_new(info, &compiler) == XG_OK)
        defer { xg_grammar_compiler_free(compiler) }
        try body(compiler)
    }

    @Test
    func nullFormatCompilesWithDefaults() throws {
        try withCompiler { compiler in
            var compiled: OpaquePointer?
            let status = Self.schema.withCString {
                xg_compile_json_schema_with_format(compiler, $0, nil, &compiled)
            }
            #expect(status == XG_OK)
            #expect(compiled != nil)
            xg_compiled_grammar_free(compiled)
        }
    }

    @Test
    func compactFormatCompiles() throws {
        try withCompiler { compiler in
            var compiled: OpaquePointer?
            let status: XGStatus = Self.schema.withCString { schemaPtr in
                ",".withCString { item in
                    ":".withCString { key in
                        var format = XGJSONSchemaFormat(
                            any_whitespace: 0, indent: -1, item_separator: item,
                            key_separator: key, strict_mode: 1)
                        return xg_compile_json_schema_with_format(
                            compiler, schemaPtr, &format, &compiled)
                    }
                }
            }
            #expect(status == XG_OK)
            #expect(compiled != nil)
            xg_compiled_grammar_free(compiled)
        }
    }

    @Test
    func oneNullSeparatorIsRejected() throws {
        try withCompiler { compiler in
            var compiled: OpaquePointer?
            let status: XGStatus = Self.schema.withCString { schemaPtr in
                ",".withCString { item in
                    var format = XGJSONSchemaFormat(
                        any_whitespace: 0, indent: -1, item_separator: item,
                        key_separator: nil, strict_mode: 1)
                    return xg_compile_json_schema_with_format(
                        compiler, schemaPtr, &format, &compiled)
                }
            }
            #expect(status == XG_ERR_INVALID_ARG)
            #expect(compiled == nil)
        }
    }
}

extension Array where Element == ContiguousArray<CChar> {
    /// Calls `body` with pointers to each NUL-terminated string.
    fileprivate func withUnsafeCStringArray<R>(
        _ body: (UnsafeBufferPointer<UnsafePointer<CChar>?>) throws -> R
    ) rethrows -> R {
        func recurse(_ index: Int, _ pointers: [UnsafePointer<CChar>?]) throws -> R {
            guard index < count else {
                return try pointers.withUnsafeBufferPointer { try body($0) }
            }
            return try self[index].withUnsafeBufferPointer {
                try recurse(index + 1, pointers + [$0.baseAddress])
            }
        }
        return try recurse(0, [])
    }
}
