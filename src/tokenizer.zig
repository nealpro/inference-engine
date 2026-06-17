//! Tokenizer adapters used for validation, fixtures, and prompt formatting.

const std = @import("std");

const gguf = @import("gguf.zig");

/// Errors returned by tokenizer loading and token conversion.
pub const TokenizerError = error{
    MissingTokenizer,
    UnsupportedTokenizer,
    UnknownToken,
    InvalidTokenId,
    MalformedMerge,
};

/// One byte-pair merge rule loaded from tokenizer metadata.
pub const MergeRule = struct {
    left: []const u8,
    right: []const u8,

    fn deinit(self: MergeRule, allocator: std.mem.Allocator) void {
        allocator.free(self.left);
        allocator.free(self.right);
    }
};

/// Owned tokenizer vocabulary and special-token metadata.
pub const Tokenizer = struct {
    kind: Kind,
    tokens: []const []const u8,
    bos_id: ?u32 = null,
    eos_id: ?u32 = null,
    pad_id: ?u32 = null,
    unknown_id: ?u32 = null,
    add_space_prefix: bool = false,
    add_bos_token: bool = false,
    merges: []const MergeRule = &.{},
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
        for (self.merges) |merge| merge.deinit(allocator);
        allocator.free(self.merges);
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
        const merges = try parseMergeRules(allocator, parsed.metadataStringArray("tokenizer.ggml.merges") orelse &.{});
        errdefer freeMergeRules(allocator, merges);

        return .{
            .kind = .gguf_vocab,
            .tokens = owned,
            .bos_id = parsed.metadataU32("tokenizer.ggml.bos_token_id"),
            .eos_id = parsed.metadataU32("tokenizer.ggml.eos_token_id"),
            .pad_id = parsed.metadataU32("tokenizer.ggml.padding_token_id"),
            .unknown_id = parsed.metadataU32("tokenizer.ggml.unknown_token_id"),
            .add_space_prefix = parsed.metadataBool("tokenizer.ggml.add_space_prefix") orelse false,
            .add_bos_token = parsed.metadataBool("tokenizer.ggml.add_bos_token") orelse false,
            .merges = merges,
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

    /// Encodes text with the loaded tokenizer adapter.
    pub fn encode(self: Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        if (self.kind == .placeholder) return error.UnsupportedTokenizer;
        if (self.kind == .gguf_vocab and self.merges.len > 0) return self.encodeBpe(allocator, text);
        return self.encodeLongestTokenMatch(allocator, text);
    }

    fn encodeLongestTokenMatch(self: Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
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

    fn encodeBpe(self: Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(allocator);

        if (self.add_bos_token) {
            if (self.bos_id) |id| try ids.append(allocator, id);
        }

        const prepared = if (self.add_space_prefix and (text.len == 0 or text[0] != ' '))
            try std.fmt.allocPrint(allocator, " {s}", .{text})
        else
            try allocator.dupe(u8, text);
        defer allocator.free(prepared);

        var index: usize = 0;
        while (index < prepared.len) {
            if (self.longestSpecialTokenAt(prepared[index..])) |match| {
                try ids.append(allocator, match.id);
                index += match.len;
                continue;
            }

            const start = index;
            while (index < prepared.len and self.longestSpecialTokenAt(prepared[index..]) == null) {
                index += utf8SequenceLen(prepared[index]) catch return error.UnknownToken;
            }
            try self.encodeBpeChunk(allocator, prepared[start..index], &ids);
        }

        return ids.toOwnedSlice(allocator);
    }

    fn encodeBpeChunk(self: Tokenizer, allocator: std.mem.Allocator, text: []const u8, ids: *std.ArrayList(u32)) !void {
        if (text.len == 0) return;

        var pieces: std.ArrayList([]const u8) = .empty;
        defer pieces.deinit(allocator);

        var index: usize = 0;
        while (index < text.len) {
            const len = utf8SequenceLen(text[index]) catch return error.UnknownToken;
            try pieces.append(allocator, text[index .. index + len]);
            index += len;
        }

        while (pieces.items.len > 1) {
            var best_rank: ?usize = null;
            var best_index: usize = 0;
            for (0..pieces.items.len - 1) |piece_index| {
                if (self.mergeRank(pieces.items[piece_index], pieces.items[piece_index + 1])) |rank| {
                    if (best_rank == null or rank < best_rank.?) {
                        best_rank = rank;
                        best_index = piece_index;
                    }
                }
            }
            if (best_rank == null) break;

            const left = pieces.items[best_index];
            const right = pieces.items[best_index + 1];
            const merged = try std.mem.concat(allocator, u8, &.{ left, right });
            if (!pointsInto(text, left)) allocator.free(left);
            if (!pointsInto(text, right)) allocator.free(right);
            pieces.items[best_index] = merged;
            _ = pieces.orderedRemove(best_index + 1);
        }

        for (pieces.items) |piece| {
            defer if (!pointsInto(text, piece)) allocator.free(piece);
            const id = self.tokenId(piece) orelse self.unknown_id orelse return error.UnknownToken;
            try ids.append(allocator, id);
        }
    }

    const TokenMatch = struct {
        id: u32,
        len: usize,
    };

    fn longestSpecialTokenAt(self: Tokenizer, text: []const u8) ?TokenMatch {
        var best: ?TokenMatch = null;
        for (self.tokens, 0..) |token, id| {
            if (!isSpecialToken(token)) continue;
            if (text.len < token.len or !std.mem.eql(u8, text[0..token.len], token)) continue;
            if (best == null or token.len > best.?.len) best = .{ .id = @intCast(id), .len = token.len };
        }
        return best;
    }

    fn tokenId(self: Tokenizer, token: []const u8) ?u32 {
        for (self.tokens, 0..) |candidate, id| {
            if (std.mem.eql(u8, candidate, token)) return @intCast(id);
        }
        return null;
    }

    fn mergeRank(self: Tokenizer, left: []const u8, right: []const u8) ?usize {
        for (self.merges, 0..) |merge, rank| {
            if (std.mem.eql(u8, merge.left, left) and std.mem.eql(u8, merge.right, right)) return rank;
        }
        return null;
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

fn parseMergeRules(allocator: std.mem.Allocator, raw_merges: []const []const u8) ![]const MergeRule {
    const merges = try allocator.alloc(MergeRule, raw_merges.len);
    var written: usize = 0;
    errdefer {
        for (merges[0..written]) |merge| merge.deinit(allocator);
        allocator.free(merges);
    }

    for (raw_merges) |raw| {
        const split = std.mem.indexOfScalar(u8, raw, ' ') orelse return error.MalformedMerge;
        const left = raw[0..split];
        const right = raw[split + 1 ..];
        if (left.len == 0 or right.len == 0 or std.mem.indexOfScalar(u8, right, ' ') != null) return error.MalformedMerge;
        const left_owned = try allocator.dupe(u8, left);
        errdefer allocator.free(left_owned);
        const right_owned = try allocator.dupe(u8, right);
        merges[written] = .{
            .left = left_owned,
            .right = right_owned,
        };
        written += 1;
    }

    return merges;
}

fn freeMergeRules(allocator: std.mem.Allocator, merges: []const MergeRule) void {
    for (merges) |merge| merge.deinit(allocator);
    allocator.free(merges);
}

fn utf8SequenceLen(first: u8) !usize {
    return std.unicode.utf8ByteSequenceLength(first) catch error.UnknownToken;
}

fn pointsInto(parent: []const u8, child: []const u8) bool {
    if (child.len == 0) return true;
    const parent_start = @intFromPtr(parent.ptr);
    const parent_end = parent_start + parent.len;
    const child_start = @intFromPtr(child.ptr);
    const child_end = child_start + child.len;
    return child_start >= parent_start and child_end <= parent_end;
}

fn isSpecialToken(token: []const u8) bool {
    return token.len >= 2 and token[0] == '<' and std.mem.indexOfScalar(u8, token, '>') != null;
}

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

test "GGUF BPE tokenizer applies merge ranks and special tokens" {
    var tokens = try std.testing.allocator.alloc([]const u8, 8);
    tokens[0] = try std.testing.allocator.dupe(u8, "h");
    tokens[1] = try std.testing.allocator.dupe(u8, "e");
    tokens[2] = try std.testing.allocator.dupe(u8, "he");
    tokens[3] = try std.testing.allocator.dupe(u8, "l");
    tokens[4] = try std.testing.allocator.dupe(u8, "hel");
    tokens[5] = try std.testing.allocator.dupe(u8, "hello");
    tokens[6] = try std.testing.allocator.dupe(u8, "<bos>");
    tokens[7] = try std.testing.allocator.dupe(u8, "<end_of_turn>");

    const merges = try parseMergeRules(std.testing.allocator, &.{ "h e", "he l", "hel l", "hell o" });
    var tok = Tokenizer{
        .kind = .gguf_vocab,
        .tokens = tokens,
        .bos_id = 6,
        .add_bos_token = true,
        .merges = merges,
    };
    defer tok.deinit(std.testing.allocator);

    const ids = try tok.encode(std.testing.allocator, "hello<end_of_turn>");
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 6, 5, 7 }, ids);
}
