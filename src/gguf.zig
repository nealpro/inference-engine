//! GGUF metadata and tensor-directory parsing for model validation.

const std = @import("std");

const model = @import("model.zig");

/// Errors returned while parsing GGUF metadata and tensor descriptors.
pub const GgufError = error{
    BadMagic,
    UnsupportedVersion,
    MalformedGguf,
    UnsupportedMetadataValue,
    UnsupportedQuantization,
    MissingMetadata,
    MissingTensor,
    ShapeMismatch,
    StreamTooLong,
};

const max_string_len = 16 * 1024 * 1024;
const default_alignment: u64 = 32;

/// GGUF metadata value tags supported by the parser.
pub const MetadataValueTag = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

/// Parsed GGUF metadata value.
pub const MetadataValue = union(enum) {
    u32: u32,
    i32: i32,
    u64: u64,
    f32: f32,
    bool: bool,
    string: []const u8,
    string_array: []const []const u8,
    u32_array: []const u32,
    i32_array: []const i32,
    bool_array: []const bool,
    unsupported: void,

    /// Releases owned metadata value storage.
    pub fn deinit(self: MetadataValue, allocator: std.mem.Allocator) void {
        switch (self) {
            .string => |value| allocator.free(value),
            .string_array => |values| {
                for (values) |value| allocator.free(value);
                allocator.free(values);
            },
            .u32_array => |values| allocator.free(values),
            .i32_array => |values| allocator.free(values),
            .bool_array => |values| allocator.free(values),
            else => {},
        }
    }
};

/// Parsed GGUF metadata key/value entry.
pub const MetadataEntry = struct {
    key: []const u8,
    value: MetadataValue,

    /// Releases the owned key and value storage.
    pub fn deinit(self: MetadataEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.key);
        self.value.deinit(allocator);
    }
};

/// Owned GGUF header, metadata, and tensor directory.
pub const ParsedGguf = struct {
    version: u32,
    alignment: u64,
    tensor_data_offset: u64,
    metadata: []MetadataEntry,
    tensors: []model.TensorInfo,

    /// Releases all owned parser output.
    pub fn deinit(self: ParsedGguf, allocator: std.mem.Allocator) void {
        for (self.metadata) |entry| entry.deinit(allocator);
        allocator.free(self.metadata);
        for (self.tensors) |tensor| tensor.deinit(allocator);
        allocator.free(self.tensors);
    }

    /// Returns a string metadata value by key when present.
    pub fn metadataString(self: ParsedGguf, key: []const u8) ?[]const u8 {
        for (self.metadata) |entry| {
            if (std.mem.eql(u8, entry.key, key)) {
                return switch (entry.value) {
                    .string => |value| value,
                    else => null,
                };
            }
        }
        return null;
    }

    /// Returns an integer metadata value by key when representable as u32.
    pub fn metadataU32(self: ParsedGguf, key: []const u8) ?u32 {
        for (self.metadata) |entry| {
            if (std.mem.eql(u8, entry.key, key)) {
                return switch (entry.value) {
                    .u32 => |value| value,
                    .u64 => |value| std.math.cast(u32, value),
                    .i32 => |value| if (value >= 0) @intCast(value) else null,
                    else => null,
                };
            }
        }
        return null;
    }

    /// Returns a string-array metadata value by key when present.
    pub fn metadataStringArray(self: ParsedGguf, key: []const u8) ?[]const []const u8 {
        for (self.metadata) |entry| {
            if (std.mem.eql(u8, entry.key, key)) {
                return switch (entry.value) {
                    .string_array => |values| values,
                    else => null,
                };
            }
        }
        return null;
    }

    /// Returns an i32-array metadata value by key when present.
    pub fn metadataI32Array(self: ParsedGguf, key: []const u8) ?[]const i32 {
        for (self.metadata) |entry| {
            if (std.mem.eql(u8, entry.key, key)) {
                return switch (entry.value) {
                    .i32_array => |values| values,
                    else => null,
                };
            }
        }
        return null;
    }

    /// Returns a bool-array metadata value by key when present.
    pub fn metadataBoolArray(self: ParsedGguf, key: []const u8) ?[]const bool {
        for (self.metadata) |entry| {
            if (std.mem.eql(u8, entry.key, key)) {
                return switch (entry.value) {
                    .bool_array => |values| values,
                    else => null,
                };
            }
        }
        return null;
    }

    /// Returns a float32 metadata value by key when present.
    pub fn metadataF32(self: ParsedGguf, key: []const u8) ?f32 {
        for (self.metadata) |entry| {
            if (std.mem.eql(u8, entry.key, key)) {
                return switch (entry.value) {
                    .f32 => |value| value,
                    else => null,
                };
            }
        }
        return null;
    }

    /// Finds a tensor descriptor by exact tensor name.
    pub fn tensorByName(self: ParsedGguf, name: []const u8) ?model.TensorInfo {
        for (self.tensors) |tensor| {
            if (std.mem.eql(u8, tensor.name, name)) return tensor;
        }
        return null;
    }

    /// Builds model-level metadata from GGUF metadata and tensor descriptors.
    pub fn buildModelSpec(self: ParsedGguf, allocator: std.mem.Allocator) !model.ModelSpec {
        const architecture = self.metadataString("general.architecture") orelse return error.MissingMetadata;
        const arch_owned = try allocator.dupe(u8, architecture);
        errdefer allocator.free(arch_owned);

        var key_buf: [256]u8 = undefined;
        const context_key = try std.fmt.bufPrint(&key_buf, "{s}.context_length", .{architecture});
        const context_length = self.metadataU32(context_key);
        const embedding_key = try std.fmt.bufPrint(&key_buf, "{s}.embedding_length", .{architecture});
        const embedding_length = self.metadataU32(embedding_key);
        const block_key = try std.fmt.bufPrint(&key_buf, "{s}.block_count", .{architecture});
        const block_count = self.metadataU32(block_key);
        const ff_key = try std.fmt.bufPrint(&key_buf, "{s}.feed_forward_length", .{architecture});
        const feed_forward_length = self.metadataU32(ff_key);
        const head_key = try std.fmt.bufPrint(&key_buf, "{s}.attention.head_count", .{architecture});
        const attention_head_count = self.metadataU32(head_key);
        const kv_head_key = try std.fmt.bufPrint(&key_buf, "{s}.attention.head_count_kv", .{architecture});
        const attention_head_count_kv = self.metadataU32(kv_head_key);
        const attention_head_count_kv_per_layer = try duplicateU32FromI32Array(
            allocator,
            self.metadataI32Array(kv_head_key),
        );
        errdefer if (attention_head_count_kv_per_layer) |values| allocator.free(values);
        const rope_key = try std.fmt.bufPrint(&key_buf, "{s}.rope.dimension_count", .{architecture});
        const rope_dimension_count = self.metadataU32(rope_key);
        const rope_swa_key = try std.fmt.bufPrint(&key_buf, "{s}.rope.dimension_count_swa", .{architecture});
        const rope_dimension_count_swa = self.metadataU32(rope_swa_key);
        const key_length_key = try std.fmt.bufPrint(&key_buf, "{s}.attention.key_length", .{architecture});
        const attention_key_length = self.metadataU32(key_length_key);
        const value_length_key = try std.fmt.bufPrint(&key_buf, "{s}.attention.value_length", .{architecture});
        const attention_value_length = self.metadataU32(value_length_key);
        const key_length_swa_key = try std.fmt.bufPrint(&key_buf, "{s}.attention.key_length_swa", .{architecture});
        const attention_key_length_swa = self.metadataU32(key_length_swa_key);
        const value_length_swa_key = try std.fmt.bufPrint(&key_buf, "{s}.attention.value_length_swa", .{architecture});
        const attention_value_length_swa = self.metadataU32(value_length_swa_key);
        const sliding_pattern_key = try std.fmt.bufPrint(&key_buf, "{s}.attention.sliding_window_pattern", .{architecture});
        const sliding_window_pattern = try duplicateBoolArray(allocator, self.metadataBoolArray(sliding_pattern_key));
        errdefer if (sliding_window_pattern) |values| allocator.free(values);
        const rope_freq_base_key = try std.fmt.bufPrint(&key_buf, "{s}.rope.freq_base", .{architecture});
        const rope_freq_base = self.metadataF32(rope_freq_base_key);
        const rope_freq_base_swa_key = try std.fmt.bufPrint(&key_buf, "{s}.rope.freq_base_swa", .{architecture});
        const rope_freq_base_swa = self.metadataF32(rope_freq_base_swa_key);
        const rms_eps_key = try std.fmt.bufPrint(&key_buf, "{s}.attention.layer_norm_rms_epsilon", .{architecture});
        const attention_layer_norm_rms_epsilon = self.metadataF32(rms_eps_key);
        const final_logit_softcapping_key = try std.fmt.bufPrint(&key_buf, "{s}.final_logit_softcapping", .{architecture});
        const final_logit_softcapping = self.metadataF32(final_logit_softcapping_key);
        const sliding_window_key = try std.fmt.bufPrint(&key_buf, "{s}.attention.sliding_window", .{architecture});
        const attention_sliding_window = self.metadataU32(sliding_window_key);

        var quantization = model.QuantizationSummary{};
        for (self.tensors) |tensor| quantization.observe(tensor.kind);

        const name = if (self.metadataString("general.name")) |value|
            try allocator.dupe(u8, value)
        else
            null;
        errdefer if (name) |owned| allocator.free(owned);

        const vocab_size: ?u32 = if (self.metadataStringArray("tokenizer.ggml.tokens")) |tokens|
            std.math.cast(u32, tokens.len)
        else
            null;

        const spec = model.ModelSpec{
            .name = name,
            .architecture = arch_owned,
            .context_length = context_length,
            .embedding_length = embedding_length,
            .block_count = block_count,
            .feed_forward_length = feed_forward_length,
            .attention_head_count = attention_head_count,
            .attention_head_count_kv = attention_head_count_kv orelse uniformU32Value(attention_head_count_kv_per_layer),
            .attention_head_count_kv_per_layer = attention_head_count_kv_per_layer,
            .rope_dimension_count = rope_dimension_count,
            .rope_dimension_count_swa = rope_dimension_count_swa,
            .attention_key_length = attention_key_length,
            .attention_value_length = attention_value_length,
            .attention_key_length_swa = attention_key_length_swa,
            .attention_value_length_swa = attention_value_length_swa,
            .sliding_window_pattern = sliding_window_pattern,
            .rope_freq_base = rope_freq_base,
            .rope_freq_base_swa = rope_freq_base_swa,
            .attention_layer_norm_rms_epsilon = attention_layer_norm_rms_epsilon,
            .final_logit_softcapping = final_logit_softcapping,
            .attention_sliding_window = attention_sliding_window,
            .vocab_size = vocab_size,
            .tensor_count = self.tensors.len,
            .quantization = quantization,
        };

        try validateTensorShapes(self, spec);
        return spec;
    }
};

/// Parses a complete GGUF byte slice.
pub fn parseFromSlice(allocator: std.mem.Allocator, bytes: []const u8) !ParsedGguf {
    var reader = std.Io.Reader.fixed(bytes);
    return parse(allocator, &reader, bytes.len);
}

fn duplicateU32FromI32Array(allocator: std.mem.Allocator, maybe_values: ?[]const i32) !?[]const u32 {
    const values = maybe_values orelse return null;
    const owned = try allocator.alloc(u32, values.len);
    errdefer allocator.free(owned);
    for (values, 0..) |value, index| {
        if (value < 0) return error.MalformedGguf;
        owned[index] = @intCast(value);
    }
    return owned;
}

fn duplicateBoolArray(allocator: std.mem.Allocator, maybe_values: ?[]const bool) !?[]const bool {
    const values = maybe_values orelse return null;
    return try allocator.dupe(bool, values);
}

fn uniformU32Value(maybe_values: ?[]const u32) ?u32 {
    const values = maybe_values orelse return null;
    if (values.len == 0) return null;
    const first = values[0];
    for (values[1..]) |value| {
        if (value != first) return null;
    }
    return first;
}

/// Parses GGUF metadata and tensor descriptors from a filesystem path.
pub fn parseFromPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !ParsedGguf {
    var file = if (std.Io.Dir.path.isAbsolute(path))
        try std.Io.Dir.openFileAbsolute(io, path, .{})
    else
        try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    var buffer: [8192]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    return parse(allocator, &file_reader.interface, stat.size);
}

fn parse(
    allocator: std.mem.Allocator,
    reader: *std.Io.Reader,
    file_size: ?u64,
) !ParsedGguf {
    const magic = try reader.takeArray(4);
    if (!std.mem.eql(u8, magic, "GGUF")) return error.BadMagic;

    const version = try reader.takeInt(u32, .little);
    if (version != 2 and version != 3) return error.UnsupportedVersion;

    const tensor_count = try reader.takeInt(u64, .little);
    const metadata_count = try reader.takeInt(u64, .little);
    const tensor_count_usize = std.math.cast(usize, tensor_count) orelse return error.MalformedGguf;
    const metadata_count_usize = std.math.cast(usize, metadata_count) orelse return error.MalformedGguf;

    var metadata = try allocator.alloc(MetadataEntry, metadata_count_usize);
    var metadata_written: usize = 0;
    errdefer {
        for (metadata[0..metadata_written]) |entry| entry.deinit(allocator);
        allocator.free(metadata);
    }

    var alignment = default_alignment;
    for (metadata) |*entry| {
        entry.* = .{
            .key = try readString(allocator, reader),
            .value = try readMetadataValue(allocator, reader),
        };
        metadata_written += 1;
        if (std.mem.eql(u8, entry.key, "general.alignment")) {
            switch (entry.value) {
                .u32 => |value| alignment = value,
                .u64 => |value| alignment = value,
                else => {},
            }
        }
    }

    var tensors = try allocator.alloc(model.TensorInfo, tensor_count_usize);
    var tensor_written: usize = 0;
    errdefer {
        for (tensors[0..tensor_written]) |tensor| tensor.deinit(allocator);
        allocator.free(tensors);
    }

    for (tensors) |*tensor| {
        const name = try readString(allocator, reader);
        errdefer allocator.free(name);

        const dim_count_raw = try reader.takeInt(u32, .little);
        if (dim_count_raw == 0 or dim_count_raw > 8) return error.MalformedGguf;
        const dim_count = std.math.cast(usize, dim_count_raw) orelse return error.MalformedGguf;
        const dims = try allocator.alloc(u64, dim_count);
        errdefer allocator.free(dims);
        for (dims) |*dim| dim.* = try reader.takeInt(u64, .little);

        const raw_kind = try reader.takeInt(u32, .little);
        const kind = model.QuantizationType.fromGgmlType(raw_kind);
        if (!kind.isSupported()) return error.UnsupportedQuantization;
        const offset = try reader.takeInt(u64, .little);

        tensor.* = .{
            .name = name,
            .dims = dims,
            .kind = kind,
            .offset = offset,
            .byte_len = try model.tensorByteLen(kind, try elementCount(dims)),
        };
        tensor_written += 1;
    }

    const tensor_data_offset = alignForward(reader.seek, alignment);
    if (file_size) |size| {
        for (tensors) |tensor| {
            const abs_start = tensor_data_offset + tensor.offset;
            const abs_end = std.math.add(u64, abs_start, tensor.byte_len) catch return error.MalformedGguf;
            if (abs_end > size) return error.MalformedGguf;
        }
    }

    return .{
        .version = version,
        .alignment = alignment,
        .tensor_data_offset = tensor_data_offset,
        .metadata = metadata,
        .tensors = tensors,
    };
}

fn readMetadataValue(allocator: std.mem.Allocator, reader: *std.Io.Reader) !MetadataValue {
    const raw_tag = try reader.takeInt(u32, .little);
    const tag = metadataValueTag(raw_tag) orelse return error.UnsupportedMetadataValue;
    return switch (tag) {
        .uint8 => .{ .u32 = try reader.takeByte() },
        .int8 => .{ .i32 = try reader.takeByteSigned() },
        .uint16 => .{ .u32 = try reader.takeInt(u16, .little) },
        .int16 => .{ .i32 = try reader.takeInt(i16, .little) },
        .uint32 => .{ .u32 = try reader.takeInt(u32, .little) },
        .int32 => .{ .i32 = try reader.takeInt(i32, .little) },
        .uint64 => .{ .u64 = try reader.takeInt(u64, .little) },
        .int64 => .{ .unsupported = try skipAndReturn(reader, i64) },
        .float32 => .{ .f32 = @bitCast(try reader.takeInt(u32, .little)) },
        .float64 => .{ .unsupported = try skipAndReturn(reader, u64) },
        .bool => .{ .bool = (try reader.takeByte()) != 0 },
        .string => .{ .string = try readString(allocator, reader) },
        .array => try readArrayMetadataValue(allocator, reader),
    };
}

fn skipAndReturn(reader: *std.Io.Reader, comptime T: type) !void {
    _ = try reader.takeInt(T, .little);
}

fn readArrayMetadataValue(allocator: std.mem.Allocator, reader: *std.Io.Reader) !MetadataValue {
    const raw_inner = try reader.takeInt(u32, .little);
    const inner = metadataValueTag(raw_inner) orelse return error.UnsupportedMetadataValue;
    const len_raw = try reader.takeInt(u64, .little);
    const len = std.math.cast(usize, len_raw) orelse return error.StreamTooLong;

    return switch (inner) {
        .string => blk: {
            var values = try allocator.alloc([]const u8, len);
            var written: usize = 0;
            errdefer {
                for (values[0..written]) |value| allocator.free(value);
                allocator.free(values);
            }
            for (values) |*value| {
                value.* = try readString(allocator, reader);
                written += 1;
            }
            break :blk .{ .string_array = values };
        },
        .uint32 => blk: {
            const values = try allocator.alloc(u32, len);
            errdefer allocator.free(values);
            for (values) |*value| value.* = try reader.takeInt(u32, .little);
            break :blk .{ .u32_array = values };
        },
        .int32 => blk: {
            const values = try allocator.alloc(i32, len);
            errdefer allocator.free(values);
            for (values) |*value| value.* = try reader.takeInt(i32, .little);
            break :blk .{ .i32_array = values };
        },
        .bool => blk: {
            const values = try allocator.alloc(bool, len);
            errdefer allocator.free(values);
            for (values) |*value| value.* = (try reader.takeByte()) != 0;
            break :blk .{ .bool_array = values };
        },
        else => {
            try skipArrayValues(reader, inner, len);
            return .{ .unsupported = {} };
        },
    };
}

fn skipArrayValues(reader: *std.Io.Reader, inner: MetadataValueTag, len: usize) !void {
    for (0..len) |_| {
        switch (inner) {
            .uint8, .int8, .bool => _ = try reader.takeByte(),
            .uint16, .int16 => _ = try reader.takeInt(u16, .little),
            .uint32, .int32, .float32 => _ = try reader.takeInt(u32, .little),
            .uint64, .int64, .float64 => _ = try reader.takeInt(u64, .little),
            .string => {
                const string_len = try reader.takeInt(u64, .little);
                if (string_len > max_string_len) return error.StreamTooLong;
                try reader.discardAll64(string_len);
            },
            .array => return error.UnsupportedMetadataValue,
        }
    }
}

fn metadataValueTag(raw: u32) ?MetadataValueTag {
    return switch (raw) {
        0 => .uint8,
        1 => .int8,
        2 => .uint16,
        3 => .int16,
        4 => .uint32,
        5 => .int32,
        6 => .float32,
        7 => .bool,
        8 => .string,
        9 => .array,
        10 => .uint64,
        11 => .int64,
        12 => .float64,
        else => null,
    };
}

fn readString(allocator: std.mem.Allocator, reader: *std.Io.Reader) ![]const u8 {
    const len_raw = try reader.takeInt(u64, .little);
    if (len_raw > max_string_len) return error.StreamTooLong;
    const len = std.math.cast(usize, len_raw) orelse return error.StreamTooLong;
    return try reader.readAlloc(allocator, len);
}

fn elementCount(dims: []const u64) !u64 {
    var total: u64 = 1;
    if (dims.len == 0) return error.MalformedGguf;
    for (dims) |dim| {
        if (dim == 0) return error.MalformedGguf;
        total = std.math.mul(u64, total, dim) catch return error.MalformedGguf;
    }
    return total;
}

fn validateTensorShapes(gguf: ParsedGguf, spec: model.ModelSpec) !void {
    for (gguf.tensors) |tensor| {
        _ = try tensor.elementCount();
    }

    if (gguf.tensorByName("token_embd.weight")) |embedding| {
        if (spec.embedding_length) |hidden| {
            if (!dimsContain(embedding.dims, hidden)) return error.ShapeMismatch;
        }
        if (spec.vocab_size) |vocab| {
            if (!dimsContain(embedding.dims, vocab)) return error.ShapeMismatch;
        }
    }
}

fn dimsContain(dims: []const u64, expected: u32) bool {
    for (dims) |dim| {
        if (dim == expected) return true;
    }
    return false;
}

fn alignForward(value: u64, alignment: u64) u64 {
    if (alignment == 0) return value;
    const remainder = value % alignment;
    return if (remainder == 0) value else value + (alignment - remainder);
}

test "GGUF parser reads metadata and tensor directory" {
    const bytes = try fixtureGguf(std.testing.allocator, false);
    defer std.testing.allocator.free(bytes);

    var parsed = try parseFromSlice(std.testing.allocator, bytes);
    defer parsed.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 3), parsed.version);
    try std.testing.expectEqualStrings("toy", parsed.metadataString("general.architecture").?);
    try std.testing.expectEqual(@as(u32, 2), parsed.metadataU32("toy.embedding_length").?);
    try std.testing.expectEqualSlices(i32, &.{1}, parsed.metadataI32Array("toy.attention.head_count_kv").?);
    try std.testing.expectEqualSlices(bool, &.{true}, parsed.metadataBoolArray("toy.attention.sliding_window_pattern").?);
    try std.testing.expectEqual(@as(usize, 1), parsed.tensors.len);
    try std.testing.expectEqual(model.QuantizationType.q6_k, parsed.tensors[0].kind);

    var spec = try parsed.buildModelSpec(std.testing.allocator);
    defer spec.deinit(std.testing.allocator);
    try spec.validateForTextInference();
}

test "GGUF parser rejects bad magic" {
    var bytes = try fixtureGguf(std.testing.allocator, false);
    defer std.testing.allocator.free(bytes);
    bytes[0] = 'B';
    try std.testing.expectError(error.BadMagic, parseFromSlice(std.testing.allocator, bytes));
}

test "GGUF parser rejects unsupported quantization" {
    const bytes = try fixtureGguf(std.testing.allocator, true);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectError(error.UnsupportedQuantization, parseFromSlice(std.testing.allocator, bytes));
}

fn fixtureGguf(allocator: std.mem.Allocator, unsupported_quant: bool) ![]u8 {
    var bytes: std.ArrayList(u8) = .empty;
    errdefer bytes.deinit(allocator);

    const string_metadata = .{
        .{ "general.architecture", "toy" },
        .{ "general.name", "toy model" },
    };
    const u32_metadata = .{
        .{ "general.alignment", 32 },
        .{ "toy.context_length", 8 },
        .{ "toy.embedding_length", 2 },
        .{ "toy.block_count", 1 },
        .{ "toy.attention.head_count", 1 },
    };
    const metadata_count = string_metadata.len + u32_metadata.len + 3;

    try bytes.appendSlice(allocator, "GGUF");
    try appendInt(&bytes, allocator, u32, 3);
    try appendInt(&bytes, allocator, u64, 1);
    try appendInt(&bytes, allocator, u64, metadata_count);

    inline for (string_metadata) |entry| try appendStringMeta(&bytes, allocator, entry[0], entry[1]);
    inline for (u32_metadata) |entry| try appendU32Meta(&bytes, allocator, entry[0], entry[1]);
    try appendI32ArrayMeta(&bytes, allocator, "toy.attention.head_count_kv", &.{1});
    try appendBoolArrayMeta(&bytes, allocator, "toy.attention.sliding_window_pattern", &.{true});
    try appendStringArrayMeta(&bytes, allocator, "tokenizer.ggml.tokens", &.{ "a", "b" });

    try appendString(&bytes, allocator, "token_embd.weight");
    try appendInt(&bytes, allocator, u32, 2);
    try appendInt(&bytes, allocator, u64, 2);
    try appendInt(&bytes, allocator, u64, 2);
    const tensor_kind: u32 = if (unsupported_quant) 999 else 14;
    try appendInt(&bytes, allocator, u32, tensor_kind);
    try appendInt(&bytes, allocator, u64, 0);

    while (bytes.items.len % 32 != 0) try bytes.append(allocator, 0);
    try bytes.appendNTimes(allocator, 0, 210);

    return bytes.toOwnedSlice(allocator);
}

fn appendStringMeta(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
    try appendString(bytes, allocator, key);
    try appendInt(bytes, allocator, u32, @intFromEnum(MetadataValueTag.string));
    try appendString(bytes, allocator, value);
}

fn appendStringArrayMeta(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, values: []const []const u8) !void {
    try appendString(bytes, allocator, key);
    try appendInt(bytes, allocator, u32, @intFromEnum(MetadataValueTag.array));
    try appendInt(bytes, allocator, u32, @intFromEnum(MetadataValueTag.string));
    try appendInt(bytes, allocator, u64, values.len);
    for (values) |value| try appendString(bytes, allocator, value);
}

fn appendI32ArrayMeta(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, values: []const i32) !void {
    try appendString(bytes, allocator, key);
    try appendInt(bytes, allocator, u32, @intFromEnum(MetadataValueTag.array));
    try appendInt(bytes, allocator, u32, @intFromEnum(MetadataValueTag.int32));
    try appendInt(bytes, allocator, u64, values.len);
    for (values) |value| try appendInt(bytes, allocator, i32, value);
}

fn appendBoolArrayMeta(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, values: []const bool) !void {
    try appendString(bytes, allocator, key);
    try appendInt(bytes, allocator, u32, @intFromEnum(MetadataValueTag.array));
    try appendInt(bytes, allocator, u32, @intFromEnum(MetadataValueTag.bool));
    try appendInt(bytes, allocator, u64, values.len);
    for (values) |value| try bytes.append(allocator, if (value) 1 else 0);
}

fn appendU32Meta(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, value: u32) !void {
    try appendString(bytes, allocator, key);
    try appendInt(bytes, allocator, u32, @intFromEnum(MetadataValueTag.uint32));
    try appendInt(bytes, allocator, u32, value);
}

fn appendString(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try appendInt(bytes, allocator, u64, value.len);
    try bytes.appendSlice(allocator, value);
}

fn appendInt(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, comptime T: type, value: anytype) !void {
    var buf: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buf, @intCast(value), .little);
    try bytes.appendSlice(allocator, &buf);
}
