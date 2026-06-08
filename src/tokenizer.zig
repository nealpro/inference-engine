//! Tokenizer adapters used for validation, fixtures, and prompt formatting.

const std = @import("std");

const gguf = @import("gguf.zig");

/// Errors returned by tokenizer loading and token conversion.
pub const TokenizerError = error{
    MissingTokenizer,
    UnsupportedTokenizer,
    UnknownToken,
    InvalidTokenId,
};

/// Owned tokenizer vocabulary and special-token metadata.
pub const Tokenizer = struct {
    kind: Kind,
    tokens: []const []const u8,
    bos_id: ?u32 = null,
    eos_id: ?u32 = null,
    pad_id: ?u32 = null,
    chat_template: ?[]const u8 = null,

    /// Tokenizer source or adapter kind.
    pub const Kind = enum {
        placeholder,
        word_level,
        bpe_reference,
        unigram_reference,
        gguf_vocab,
    };

    /// Returns a tokenizer placeholder that cannot encode real text.
    pub fn placeholder() Tokenizer {
        return .{
            .kind = .placeholder,
            .tokens = &.{},
        };
    }

    /// Releases owned vocabulary and chat-template storage.
    pub fn deinit(self: Tokenizer, allocator: std.mem.Allocator) void {
        for (self.tokens) |token| allocator.free(token);
        allocator.free(self.tokens);
        if (self.chat_template) |template| allocator.free(template);
    }

    /// Builds a tokenizer view from GGUF tokenizer metadata.
    pub fn fromGguf(allocator: std.mem.Allocator, parsed: gguf.ParsedGguf) !Tokenizer {
        const tokens = parsed.metadataStringArray("tokenizer.ggml.tokens") orelse return error.MissingTokenizer;
        var owned = try allocator.alloc([]const u8, tokens.len);
        var written: usize = 0;
        errdefer {
            for (owned[0..written]) |token| allocator.free(token);
            allocator.free(owned);
        }
        for (tokens) |token| {
            owned[written] = try allocator.dupe(u8, token);
            written += 1;
        }

        return .{
            .kind = .gguf_vocab,
            .tokens = owned,
            .bos_id = parsed.metadataU32("tokenizer.ggml.bos_token_id"),
            .eos_id = parsed.metadataU32("tokenizer.ggml.eos_token_id"),
            .pad_id = parsed.metadataU32("tokenizer.ggml.padding_token_id"),
            .chat_template = if (parsed.metadataString("tokenizer.chat_template")) |template|
                try allocator.dupe(u8, template)
            else
                null,
        };
    }

    /// Loads a small reference tokenizer from tokenizer.json bytes.
    pub fn fromJsonSlice(allocator: std.mem.Allocator, bytes: []const u8) !Tokenizer {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
        defer parsed.deinit();

        const root = parsed.value.object;
        const model_obj = root.get("model") orelse return error.UnsupportedTokenizer;
        if (model_obj != .object) return error.UnsupportedTokenizer;
        const model_type_value = model_obj.object.get("type") orelse return error.UnsupportedTokenizer;
        if (model_type_value != .string) return error.UnsupportedTokenizer;
        const model_type = model_type_value.string;

        const parsed_tokens_and_kind = if (std.mem.eql(u8, model_type, "WordLevel")) blk: {
            const vocab = model_obj.object.get("vocab") orelse return error.UnsupportedTokenizer;
            break :blk .{ try parseObjectVocab(allocator, vocab), Kind.word_level };
        } else if (std.mem.eql(u8, model_type, "BPE")) blk: {
            const vocab = model_obj.object.get("vocab") orelse return error.UnsupportedTokenizer;
            break :blk .{ try parseObjectVocab(allocator, vocab), Kind.bpe_reference };
        } else if (std.mem.eql(u8, model_type, "Unigram")) blk: {
            const vocab = model_obj.object.get("vocab") orelse return error.UnsupportedTokenizer;
            break :blk .{ try parseUnigramVocab(allocator, vocab), Kind.unigram_reference };
        } else {
            return error.UnsupportedTokenizer;
        };
        const tokens = parsed_tokens_and_kind[0];
        const kind = parsed_tokens_and_kind[1];
        errdefer freeTokens(allocator, tokens);

        return .{
            .kind = kind,
            .tokens = tokens,
            .bos_id = specialId(root.get("bos_token")),
            .eos_id = specialId(root.get("eos_token")),
            .pad_id = specialId(root.get("pad_token")),
            .chat_template = if (root.get("chat_template")) |template|
                if (template == .string) try allocator.dupe(u8, template.string) else null
            else
                null,
        };
    }

    /// Counts raw prompt bytes for the current scaffold.
    pub fn countPromptBytes(_: Tokenizer, prompt: []const u8) usize {
        return prompt.len;
    }

    /// Encodes text with a simple longest-token match over the loaded vocab.
    pub fn encode(self: Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        if (self.kind == .placeholder) return error.UnsupportedTokenizer;

        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(allocator);

        var index: usize = 0;
        while (index < text.len) {
            if (std.ascii.isWhitespace(text[index])) {
                index += 1;
                continue;
            }

            var best_id: ?u32 = null;
            var best_len: usize = 0;
            for (self.tokens, 0..) |token, id| {
                if (token.len == 0) continue;
                if (text.len - index >= token.len and std.mem.eql(u8, text[index .. index + token.len], token)) {
                    if (token.len > best_len) {
                        best_id = @intCast(id);
                        best_len = token.len;
                    }
                }
            }

            const id = best_id orelse return error.UnknownToken;
            try ids.append(allocator, id);
            index += best_len;
        }

        return ids.toOwnedSlice(allocator);
    }

    /// Decodes token ids by concatenating their vocabulary entries.
    pub fn decode(self: Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);

        for (ids) |id| {
            if (id >= self.tokens.len) return error.InvalidTokenId;
            try out.appendSlice(allocator, self.tokens[id]);
        }
        return out.toOwnedSlice(allocator);
    }

    /// Applies the current single-user Gemma chat-template fallback.
    pub fn applySingleUserChatTemplate(
        self: Tokenizer,
        allocator: std.mem.Allocator,
        prompt: []const u8,
    ) ![]u8 {
        _ = self;
        return std.fmt.allocPrint(
            allocator,
            "<start_of_turn>user\n{s}<end_of_turn>\n<start_of_turn>model\n",
            .{prompt},
        );
    }
};

fn parseObjectVocab(allocator: std.mem.Allocator, vocab: std.json.Value) ![]const []const u8 {
    if (vocab != .object) return error.UnsupportedTokenizer;

    var max_id: usize = 0;
    var iter = vocab.object.iterator();
    while (iter.next()) |entry| {
        const id = jsonInt(entry.value_ptr.*) orelse return error.UnsupportedTokenizer;
        if (id < 0) return error.UnsupportedTokenizer;
        max_id = @max(max_id, @as(usize, @intCast(id)));
    }

    var tokens = try allocator.alloc([]const u8, max_id + 1);
    @memset(tokens, &.{});
    errdefer freeTokens(allocator, tokens);

    iter = vocab.object.iterator();
    while (iter.next()) |entry| {
        const id = @as(usize, @intCast(jsonInt(entry.value_ptr.*).?));
        tokens[id] = try allocator.dupe(u8, entry.key_ptr.*);
    }

    for (tokens) |token| {
        if (token.len == 0) return error.UnsupportedTokenizer;
    }
    return tokens;
}

fn parseUnigramVocab(allocator: std.mem.Allocator, vocab: std.json.Value) ![]const []const u8 {
    if (vocab != .array) return error.UnsupportedTokenizer;

    var tokens = try allocator.alloc([]const u8, vocab.array.items.len);
    @memset(tokens, &.{});
    errdefer freeTokens(allocator, tokens);

    for (vocab.array.items, 0..) |entry, id| {
        if (entry != .array or entry.array.items.len == 0) return error.UnsupportedTokenizer;
        const token = entry.array.items[0];
        if (token != .string) return error.UnsupportedTokenizer;
        tokens[id] = try allocator.dupe(u8, token.string);
    }
    return tokens;
}

fn freeTokens(allocator: std.mem.Allocator, tokens: []const []const u8) void {
    for (tokens) |token| if (token.len > 0) allocator.free(token);
    allocator.free(tokens);
}

fn jsonInt(value: std.json.Value) ?i64 {
    return switch (value) {
        .integer => |int| int,
        .number_string => |raw| std.fmt.parseInt(i64, raw, 10) catch null,
        else => null,
    };
}

fn specialId(value: ?std.json.Value) ?u32 {
    if (value == null) return null;
    return switch (value.?) {
        .integer => |int| if (int >= 0) @intCast(int) else null,
        else => null,
    };
}

test "word-level tokenizer encodes and decodes greedily" {
    const json =
        \\{"model":{"type":"WordLevel","vocab":{"hello":0,"world":1,"hell":2}},"bos_token":0,"eos_token":1}
    ;
    const tok = try Tokenizer.fromJsonSlice(std.testing.allocator, json);
    defer tok.deinit(std.testing.allocator);

    const ids = try tok.encode(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, ids);

    const text = try tok.decode(std.testing.allocator, ids);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings("helloworld", text);
}

test "unsupported tokenizer json fails loudly" {
    const json =
        \\{"model":{"type":"SentencePiece","vocab":[]}}
    ;
    try std.testing.expectError(error.UnsupportedTokenizer, Tokenizer.fromJsonSlice(std.testing.allocator, json));
}

test "BPE tokenizer json is loaded as reference adapter" {
    const json =
        \\{"model":{"type":"BPE","vocab":{"<start_of_turn>user":0,"<end_of_turn>":1,"hello":2,"<start_of_turn>model":3},"merges":[]}}
    ;
    const tok = try Tokenizer.fromJsonSlice(std.testing.allocator, json);
    defer tok.deinit(std.testing.allocator);

    try std.testing.expectEqual(Tokenizer.Kind.bpe_reference, tok.kind);
    const formatted = try tok.applySingleUserChatTemplate(std.testing.allocator, "hello");
    defer std.testing.allocator.free(formatted);
    try std.testing.expectEqualStrings(
        "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n",
        formatted,
    );

    const ids = try tok.encode(std.testing.allocator, formatted);
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 1, 3 }, ids);
}
