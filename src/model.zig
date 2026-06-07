const std = @import("std");

pub const target_architecture_prefix = "gemma4";

pub const ModelError = error{
    MissingModelPath,
    MissingMetadata,
    MissingTensor,
    MalformedModel,
    UnsupportedArchitecture,
    UnsupportedQuantization,
    ShapeMismatch,
};

pub const QuantizationType = enum {
    f32,
    f16,
    q4_0,
    unknown,

    pub fn fromGgmlType(value: u32) QuantizationType {
        return switch (value) {
            0 => .f32,
            1 => .f16,
            2 => .q4_0,
            else => .unknown,
        };
    }

    pub fn label(self: QuantizationType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .q4_0 => "Q4_0",
            .unknown => "unknown",
        };
    }

    pub fn isSupported(self: QuantizationType) bool {
        return switch (self) {
            .f32, .f16, .q4_0 => true,
            .unknown => false,
        };
    }
};

pub const TensorInfo = struct {
    name: []const u8,
    dims: []const u64,
    kind: QuantizationType,
    offset: u64,
    byte_len: u64,

    pub fn deinit(self: TensorInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.dims);
    }

    pub fn elementCount(self: TensorInfo) ModelError!u64 {
        var total: u64 = 1;
        if (self.dims.len == 0) return error.MalformedModel;
        for (self.dims) |dim| {
            if (dim == 0) return error.MalformedModel;
            total = std.math.mul(u64, total, dim) catch return error.MalformedModel;
        }
        return total;
    }
};

pub const QuantizationSummary = struct {
    has_f32: bool = false,
    has_f16: bool = false,
    has_q4_0: bool = false,
    has_unknown: bool = false,

    pub fn observe(self: *QuantizationSummary, kind: QuantizationType) void {
        switch (kind) {
            .f32 => self.has_f32 = true,
            .f16 => self.has_f16 = true,
            .q4_0 => self.has_q4_0 = true,
            .unknown => self.has_unknown = true,
        }
    }

    pub fn write(self: QuantizationSummary, writer: anytype) !void {
        var wrote = false;
        if (self.has_f32) {
            try writer.writeAll("F32");
            wrote = true;
        }
        if (self.has_f16) {
            if (wrote) try writer.writeAll(", ");
            try writer.writeAll("F16");
            wrote = true;
        }
        if (self.has_q4_0) {
            if (wrote) try writer.writeAll(", ");
            try writer.writeAll("Q4_0");
            wrote = true;
        }
        if (self.has_unknown) {
            if (wrote) try writer.writeAll(", ");
            try writer.writeAll("unknown");
            wrote = true;
        }
        if (!wrote) try writer.writeAll("none");
    }
};

pub const ModelSpec = struct {
    name: ?[]const u8 = null,
    architecture: []const u8,
    context_length: ?u32 = null,
    embedding_length: ?u32 = null,
    block_count: ?u32 = null,
    feed_forward_length: ?u32 = null,
    attention_head_count: ?u32 = null,
    attention_head_count_kv: ?u32 = null,
    rope_dimension_count: ?u32 = null,
    vocab_size: ?u32 = null,
    tensor_count: usize,
    quantization: QuantizationSummary,

    pub fn deinit(self: ModelSpec, allocator: std.mem.Allocator) void {
        if (self.name) |name| allocator.free(name);
        allocator.free(self.architecture);
    }

    pub fn validateForTextInference(self: ModelSpec) ModelError!void {
        if (self.architecture.len == 0) return error.MissingMetadata;
        if (self.tensor_count == 0) return error.MissingTensor;
        if (self.embedding_length == null) return error.MissingMetadata;
        if (self.block_count == null) return error.MissingMetadata;
        if (self.attention_head_count == null) return error.MissingMetadata;
        if (self.quantization.has_unknown) return error.UnsupportedQuantization;
    }
};

pub fn validateGemma4TextModel(spec: ModelSpec, tensors: []const TensorInfo) ModelError!void {
    try validateGemma4TextSpec(spec);
    try validateGemma4TextTensors(tensors);
}

pub fn validateGemma4TextSpec(spec: ModelSpec) ModelError!void {
    try spec.validateForTextInference();

    if (!std.mem.startsWith(u8, spec.architecture, target_architecture_prefix)) {
        return error.UnsupportedArchitecture;
    }
    if (spec.context_length == null) return error.MissingMetadata;
    if (spec.feed_forward_length == null) return error.MissingMetadata;
    if (spec.attention_head_count_kv == null) return error.MissingMetadata;
    if (spec.rope_dimension_count == null) return error.MissingMetadata;
    if (spec.vocab_size == null) return error.MissingMetadata;
    if (!spec.quantization.has_q4_0) return error.UnsupportedQuantization;
}

pub fn validateGemma4TextTensors(tensors: []const TensorInfo) ModelError!void {
    if (!hasTensor(tensors, "token_embd.weight")) return error.MissingTensor;
    if (!hasTensor(tensors, "output_norm.weight")) return error.MissingTensor;

    const required_block_suffixes = [_][]const u8{
        ".attn_norm.weight",
        ".attn_q.weight",
        ".attn_k.weight",
        ".attn_v.weight",
        ".attn_output.weight",
        ".ffn_norm.weight",
        ".ffn_gate.weight",
        ".ffn_up.weight",
        ".ffn_down.weight",
    };
    for (required_block_suffixes) |suffix| {
        if (!hasBlockTensorWithSuffix(tensors, suffix)) return error.MissingTensor;
    }
}

fn hasTensor(tensors: []const TensorInfo, name: []const u8) bool {
    for (tensors) |tensor| {
        if (std.mem.eql(u8, tensor.name, name)) return true;
    }
    return false;
}

fn hasBlockTensorWithSuffix(tensors: []const TensorInfo, suffix: []const u8) bool {
    for (tensors) |tensor| {
        if (std.mem.startsWith(u8, tensor.name, "blk.") and std.mem.endsWith(u8, tensor.name, suffix)) {
            return true;
        }
    }
    return false;
}

pub fn tensorByteLen(kind: QuantizationType, element_count: u64) ModelError!u64 {
    return switch (kind) {
        .f32 => std.math.mul(u64, element_count, 4) catch error.MalformedModel,
        .f16 => std.math.mul(u64, element_count, 2) catch error.MalformedModel,
        .q4_0 => blk: {
            if (element_count == 0) return error.MalformedModel;
            const blocks = (element_count + 31) / 32;
            break :blk std.math.mul(u64, blocks, 18) catch error.MalformedModel;
        },
        .unknown => error.UnsupportedQuantization,
    };
}

test "quantization labels and support" {
    try std.testing.expectEqual(QuantizationType.f32, QuantizationType.fromGgmlType(0));
    try std.testing.expectEqual(QuantizationType.f16, QuantizationType.fromGgmlType(1));
    try std.testing.expectEqual(QuantizationType.q4_0, QuantizationType.fromGgmlType(2));
    try std.testing.expectEqual(QuantizationType.unknown, QuantizationType.fromGgmlType(999));
    try std.testing.expect(QuantizationType.q4_0.isSupported());
    try std.testing.expect(!QuantizationType.unknown.isSupported());
}

test "tensor byte length supports q4_0 blocks" {
    try std.testing.expectEqual(@as(u64, 18), try tensorByteLen(.q4_0, 32));
    try std.testing.expectEqual(@as(u64, 36), try tensorByteLen(.q4_0, 33));
}

test "Gemma 4 text spec validator requires target architecture and q4_0" {
    var spec = ModelSpec{
        .architecture = "gemma4",
        .context_length = 256 * 1024,
        .embedding_length = 2,
        .block_count = 1,
        .feed_forward_length = 4,
        .attention_head_count = 1,
        .attention_head_count_kv = 1,
        .rope_dimension_count = 2,
        .vocab_size = 8,
        .tensor_count = 11,
        .quantization = .{ .has_q4_0 = true },
    };
    try validateGemma4TextSpec(spec);

    spec.architecture = "llama";
    try std.testing.expectError(error.UnsupportedArchitecture, validateGemma4TextSpec(spec));
}

test "Gemma 4 text tensor validator checks required families" {
    const tensors = [_]TensorInfo{
        tensorInfo("token_embd.weight"),
        tensorInfo("output_norm.weight"),
        tensorInfo("blk.0.attn_norm.weight"),
        tensorInfo("blk.0.attn_q.weight"),
        tensorInfo("blk.0.attn_k.weight"),
        tensorInfo("blk.0.attn_v.weight"),
        tensorInfo("blk.0.attn_output.weight"),
        tensorInfo("blk.0.ffn_norm.weight"),
        tensorInfo("blk.0.ffn_gate.weight"),
        tensorInfo("blk.0.ffn_up.weight"),
        tensorInfo("blk.0.ffn_down.weight"),
    };
    try validateGemma4TextTensors(&tensors);

    try std.testing.expectError(error.MissingTensor, validateGemma4TextTensors(tensors[0 .. tensors.len - 1]));
}

fn tensorInfo(name: []const u8) TensorInfo {
    return .{
        .name = name,
        .dims = &.{ 1, 1 },
        .kind = .q4_0,
        .offset = 0,
        .byte_len = 18,
    };
}
