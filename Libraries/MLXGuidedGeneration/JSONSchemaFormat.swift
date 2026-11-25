// Copyright © 2026 Apple Inc.

/// How a JSON-Schema grammar formats its output and how strictly it follows
/// the schema. The fields map 1:1 onto XGrammar's
/// `GrammarCompiler::CompileJSONSchema` parameters.
///
/// The default equals XGrammar's defaults, which is what
/// `GrammarConstraint(tokenizer:jsonSchema:)` compiled before this option
/// existed: any whitespace between tokens, strict mode.
public struct JSONSchemaFormat: Sendable, Hashable {

    /// The strings between array items or object members (`item`) and
    /// between an object key and its value (`key`).
    public struct Separators: Sendable, Hashable {
        public var item: String
        public var key: String

        public init(item: String, key: String) {
            self.item = item
            self.key = key
        }
    }

    /// Allow any JSON whitespace between tokens. When true, XGrammar
    /// ignores `indent` and `separators`.
    public var anyWhitespace: Bool
    /// Spaces per nesting level, with one member or item per line. `nil`
    /// keeps the output on one line. Ignored when `anyWhitespace` is true.
    public var indent: Int?
    /// Separators to require. `nil` keeps XGrammar's defaults: `", "` and
    /// `": "` without an indent, `","` and `": "` with one. Ignored when
    /// `anyWhitespace` is true.
    public var separators: Separators?
    /// Forbid object properties and array items the schema does not
    /// declare.
    public var strictMode: Bool

    public init(
        anyWhitespace: Bool = true,
        indent: Int? = nil,
        separators: Separators? = nil,
        strictMode: Bool = true
    ) {
        self.anyWhitespace = anyWhitespace
        self.indent = indent
        self.separators = separators
        self.strictMode = strictMode
    }

    /// XGrammar's defaults: any whitespace, strict mode.
    public static let `default` = JSONSchemaFormat()

    /// Single-line JSON with no whitespace between tokens (`{"a":1,"b":2}`),
    /// strict mode.
    ///
    /// Turning off `anyWhitespace` alone is not compact: XGrammar's default
    /// separators then require a space after every comma and colon.
    public static let compact = JSONSchemaFormat(
        anyWhitespace: false, separators: Separators(item: ",", key: ":"))
}
