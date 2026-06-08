//! Model metadata, tensor descriptors, and Gemma 4 text validation.

const std = @import("std");

/// GGUF architecture prefix accepted by the current Gemma 4 target.
pub const target_architecture_prefix = "gemma4";

/// Errors returned by model metadata and tensor validation.
pub const ModelError = error{
    MissingModelPath,
    MissingMetadata,
    MissingTensor,
    MalformedModel,
    UnsupportedArchitecture,
    UnsupportedQuantization,
    ShapeMismatch,
};

/// GGML tensor storage kinds recognized by the current loader.
pub const QuantizationType = enum {
    f32,
    f16,
    q4_0,
    unknown,

    /// Converts a GGML type id to the local quantization enum.
    pub fn fromGgmlType(value: u32) QuantizationType {
        return switch (value) {
            0 => .f32,
            1 => .f16,
            2 => .q4_0,
            else => .unknown,
        };
    }

    /// Returns the user-facing label for this quantization kind.
    pub fn label(self: QuantizationType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .q4_0 => "Q4_0",
            .unknown => "unknown",
        };
    }

    /// True when the tensor kind can be represented by this scaffold.
    pub fn isSupported(self: QuantizationType) bool {
        return switch (self) {
            .f32, .f16, .q4_0 => true,
            .unknown => false,
        };
    }
};

/// GGUF tensor directory entry with owned name and shape slices.
pub const TensorInfo = struct {
    name: []const u8,
    dims: []const u64,
    kind: QuantizationType,
    offset: u64,
    byte_len: u64,

    /// Releases owned tensor metadata.
    pub fn deinit(self: TensorInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.dims);
    }

    /// Returns the checked product of all tensor dimensions.
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

const Gemma4TextSpecValues = struct {
    block_count: u32,
    embedding_length: u32,
    feed_forward_length: u32,
    attention_head_count: u32,
    attention_head_count_kv: u32,
    vocab_size: u32,

    fn fromSpec(spec: ModelSpec) ModelError!Gemma4TextSpecValues {
        return .{
            .block_count = spec.block_count orelse return error.MissingMetadata,
            .embedding_length = spec.embedding_length orelse return error.MissingMetadata,
            .feed_forward_length = spec.feed_forward_length orelse return error.MissingMetadata,
            .attention_head_count = spec.attention_head_count orelse return error.MissingMetadata,
            .attention_head_count_kv = spec.attention_head_count_kv orelse return error.MissingMetadata,
            .vocab_size = spec.vocab_size orelse return error.MissingMetadata,
        };
    }

    fn kvWidth(self: Gemma4TextSpecValues) ?u32 {
        const product = std.math.mul(u32, self.embedding_length, self.attention_head_count_kv) catch return null;
        if (self.attention_head_count == 0 or product % self.attention_head_count != 0) return null;
        return product / self.attention_head_count;
    }
};

/// Summary of tensor quantization kinds present in a model.
pub const QuantizationSummary = struct {
    has_f32: bool = false,
    has_f16: bool = false,
    has_q4_0: bool = false,
    has_unknown: bool = false,

    /// Records one tensor kind in the summary.
    pub fn observe(self: *QuantizationSummary, kind: QuantizationType) void {
        switch (kind) {
            .f32 => self.has_f32 = true,
            .f16 => self.has_f16 = true,
            .q4_0 => self.has_q4_0 = true,
            .unknown => self.has_unknown = true,
        }
    }

    /// Writes a compact comma-separated quantization summary.
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

/// Model-level metadata required before Gemma 4 text inference can run.
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

    /// Releases owned metadata strings.
    pub fn deinit(self: ModelSpec, allocator: std.mem.Allocator) void {
        if (self.name) |name| allocator.free(name);
        allocator.free(self.architecture);
    }

    /// Checks the generic text-inference metadata requirements.
    pub fn validateForTextInference(self: ModelSpec) ModelError!void {
        if (self.architecture.len == 0) return error.MissingMetadata;
        if (self.tensor_count == 0) return error.MissingTensor;
        if (self.embedding_length == null) return error.MissingMetadata;
        if (self.block_count == null) return error.MissingMetadata;
        if (self.attention_head_count == null) return error.MissingMetadata;
        if (self.quantization.has_unknown) return error.UnsupportedQuantization;
    }
};

/// Validates Gemma 4 text metadata and required tensor families.
pub fn validateGemma4TextModel(spec: ModelSpec, tensors: []const TensorInfo) ModelError!void {
    try validateGemma4TextSpec(spec);
    try validateGemma4TextTensors(tensors);
    try validateGemma4TextTensorShapes(spec, tensors);
}

/// Validates Gemma 4 text model-level metadata.
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

/// Validates that required Gemma 4 text tensor families are present.
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

fn validateGemma4TextTensorShapes(spec: ModelSpec, tensors: []const TensorInfo) ModelError!void {
    const values = try Gemma4TextSpecValues.fromSpec(spec);

    const embedding = tensorByName(tensors, "token_embd.weight") orelse return error.MissingTensor;
    try expectMatrixContaining(embedding, values.vocab_size, values.embedding_length);

    const output_norm = tensorByName(tensors, "output_norm.weight") orelse return error.MissingTensor;
    try expectVector(output_norm, values.embedding_length);

    const block_count = std.math.cast(usize, values.block_count) orelse return error.MalformedModel;
    const kv_width = values.kvWidth();
    var name_buf: [128]u8 = undefined;
    for (0..block_count) |layer| {
        try expectLayerVector(tensors, &name_buf, layer, "attn_norm.weight", values.embedding_length);
        try expectLayerMatrixHiddenHidden(tensors, &name_buf, layer, "attn_q.weight", values.embedding_length);
        try expectLayerKvMatrix(tensors, &name_buf, layer, "attn_k.weight", values.embedding_length, kv_width);
        try expectLayerKvMatrix(tensors, &name_buf, layer, "attn_v.weight", values.embedding_length, kv_width);
        try expectLayerMatrixHiddenHidden(tensors, &name_buf, layer, "attn_output.weight", values.embedding_length);
        try expectLayerVector(tensors, &name_buf, layer, "ffn_norm.weight", values.embedding_length);
        try expectLayerMatrixContaining(tensors, &name_buf, layer, "ffn_gate.weight", values.embedding_length, values.feed_forward_length);
        try expectLayerMatrixContaining(tensors, &name_buf, layer, "ffn_up.weight", values.embedding_length, values.feed_forward_length);
        try expectLayerMatrixContaining(tensors, &name_buf, layer, "ffn_down.weight", values.embedding_length, values.feed_forward_length);
    }
}

fn hasTensor(tensors: []const TensorInfo, name: []const u8) bool {
    for (tensors) |tensor| {
        if (std.mem.eql(u8, tensor.name, name)) return true;
    }
    return false;
}

fn tensorByName(tensors: []const TensorInfo, name: []const u8) ?TensorInfo {
    for (tensors) |tensor| {
        if (std.mem.eql(u8, tensor.name, name)) return tensor;
    }
    return null;
}

fn hasBlockTensorWithSuffix(tensors: []const TensorInfo, suffix: []const u8) bool {
    for (tensors) |tensor| {
        if (std.mem.startsWith(u8, tensor.name, "blk.") and std.mem.endsWith(u8, tensor.name, suffix)) {
            return true;
        }
    }
    return false;
}

fn layerTensorBySuffix(tensors: []const TensorInfo, name_buf: []u8, layer: usize, suffix: []const u8) ModelError!TensorInfo {
    const name = std.fmt.bufPrint(name_buf, "blk.{d}.{s}", .{ layer, suffix }) catch return error.MalformedModel;
    return tensorByName(tensors, name) orelse error.MissingTensor;
}

fn expectLayerVector(tensors: []const TensorInfo, name_buf: []u8, layer: usize, suffix: []const u8, expected: u32) ModelError!void {
    try expectVector(try layerTensorBySuffix(tensors, name_buf, layer, suffix), expected);
}

fn expectLayerMatrixHiddenHidden(tensors: []const TensorInfo, name_buf: []u8, layer: usize, suffix: []const u8, hidden: u32) ModelError!void {
    try expectMatrixExact(try layerTensorBySuffix(tensors, name_buf, layer, suffix), hidden, hidden);
}

fn expectLayerMatrixContaining(tensors: []const TensorInfo, name_buf: []u8, layer: usize, suffix: []const u8, a: u32, b: u32) ModelError!void {
    try expectMatrixContaining(try layerTensorBySuffix(tensors, name_buf, layer, suffix), a, b);
}

fn expectLayerKvMatrix(
    tensors: []const TensorInfo,
    name_buf: []u8,
    layer: usize,
    suffix: []const u8,
    hidden: u32,
    kv_width: ?u32,
) ModelError!void {
    const tensor = try layerTensorBySuffix(tensors, name_buf, layer, suffix);
    if (kv_width) |width| {
        try expectMatrixContaining(tensor, hidden, width);
    } else {
        try expectMatrixWithDim(tensor, hidden);
    }
}

fn expectVector(tensor: TensorInfo, len: u32) ModelError!void {
    if (tensor.dims.len != 1 or tensor.dims[0] != len) return error.ShapeMismatch;
}

fn expectMatrixExact(tensor: TensorInfo, rows_or_cols_a: u32, rows_or_cols_b: u32) ModelError!void {
    if (tensor.dims.len != 2) return error.ShapeMismatch;
    const a = @as(u64, rows_or_cols_a);
    const b = @as(u64, rows_or_cols_b);
    if (!((tensor.dims[0] == a and tensor.dims[1] == b) or (tensor.dims[0] == b and tensor.dims[1] == a))) {
        return error.ShapeMismatch;
    }
}

fn expectMatrixContaining(tensor: TensorInfo, a: u32, b: u32) ModelError!void {
    if (tensor.dims.len != 2) return error.ShapeMismatch;
    if (!dimsContain(tensor.dims, a) or !dimsContain(tensor.dims, b)) return error.ShapeMismatch;
}

fn expectMatrixWithDim(tensor: TensorInfo, expected: u32) ModelError!void {
    if (tensor.dims.len != 2 or !dimsContain(tensor.dims, expected)) return error.ShapeMismatch;
}

fn dimsContain(dims: []const u64, expected: u32) bool {
    for (dims) |dim| {
        if (dim == expected) return true;
    }
    return false;
}

/// Returns the byte length for a tensor with the given kind and element count.
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

test "Gemma 4 text model validator checks every layer and shapes" {
    const tensors = [_]TensorInfo{
        tensorInfoDims("token_embd.weight", &.{ 8, 4 }),
        tensorInfoDims("output_norm.weight", &.{4}),
        tensorInfoDims("blk.0.attn_norm.weight", &.{4}),
        tensorInfoDims("blk.0.attn_q.weight", &.{ 4, 4 }),
        tensorInfoDims("blk.0.attn_k.weight", &.{ 2, 4 }),
        tensorInfoDims("blk.0.attn_v.weight", &.{ 2, 4 }),
        tensorInfoDims("blk.0.attn_output.weight", &.{ 4, 4 }),
        tensorInfoDims("blk.0.ffn_norm.weight", &.{4}),
        tensorInfoDims("blk.0.ffn_gate.weight", &.{ 8, 4 }),
        tensorInfoDims("blk.0.ffn_up.weight", &.{ 8, 4 }),
        tensorInfoDims("blk.0.ffn_down.weight", &.{ 4, 8 }),
        tensorInfoDims("blk.1.attn_norm.weight", &.{4}),
        tensorInfoDims("blk.1.attn_q.weight", &.{ 4, 4 }),
        tensorInfoDims("blk.1.attn_k.weight", &.{ 2, 4 }),
        tensorInfoDims("blk.1.attn_v.weight", &.{ 2, 4 }),
        tensorInfoDims("blk.1.attn_output.weight", &.{ 4, 4 }),
        tensorInfoDims("blk.1.ffn_norm.weight", &.{4}),
        tensorInfoDims("blk.1.ffn_gate.weight", &.{ 8, 4 }),
        tensorInfoDims("blk.1.ffn_up.weight", &.{ 8, 4 }),
        tensorInfoDims("blk.1.ffn_down.weight", &.{ 4, 8 }),
    };
    try validateGemma4TextModel(validGemma4Spec(2), &tensors);

    try std.testing.expectError(error.MissingTensor, validateGemma4TextModel(
        validGemma4Spec(2),
        tensors[0 .. tensors.len - 1],
    ));
}

test "Gemma 4 text model validator rejects bad top-level and layer shapes" {
    const bad_embedding = [_]TensorInfo{
        tensorInfoDims("token_embd.weight", &.{ 8, 5 }),
        tensorInfoDims("output_norm.weight", &.{4}),
        tensorInfoDims("blk.0.attn_norm.weight", &.{4}),
        tensorInfoDims("blk.0.attn_q.weight", &.{ 4, 4 }),
        tensorInfoDims("blk.0.attn_k.weight", &.{ 2, 4 }),
        tensorInfoDims("blk.0.attn_v.weight", &.{ 2, 4 }),
        tensorInfoDims("blk.0.attn_output.weight", &.{ 4, 4 }),
        tensorInfoDims("blk.0.ffn_norm.weight", &.{4}),
        tensorInfoDims("blk.0.ffn_gate.weight", &.{ 8, 4 }),
        tensorInfoDims("blk.0.ffn_up.weight", &.{ 8, 4 }),
        tensorInfoDims("blk.0.ffn_down.weight", &.{ 4, 8 }),
    };
    try std.testing.expectError(error.ShapeMismatch, validateGemma4TextModel(validGemma4Spec(1), &bad_embedding));

    const bad_norm = [_]TensorInfo{
        tensorInfoDims("token_embd.weight", &.{ 8, 4 }),
        tensorInfoDims("output_norm.weight", &.{4}),
        tensorInfoDims("blk.0.attn_norm.weight", &.{ 4, 1 }),
        tensorInfoDims("blk.0.attn_q.weight", &.{ 4, 4 }),
        tensorInfoDims("blk.0.attn_k.weight", &.{ 2, 4 }),
        tensorInfoDims("blk.0.attn_v.weight", &.{ 2, 4 }),
        tensorInfoDims("blk.0.attn_output.weight", &.{ 4, 4 }),
        tensorInfoDims("blk.0.ffn_norm.weight", &.{4}),
        tensorInfoDims("blk.0.ffn_gate.weight", &.{ 8, 4 }),
        tensorInfoDims("blk.0.ffn_up.weight", &.{ 8, 4 }),
        tensorInfoDims("blk.0.ffn_down.weight", &.{ 4, 8 }),
    };
    try std.testing.expectError(error.ShapeMismatch, validateGemma4TextModel(validGemma4Spec(1), &bad_norm));

    const bad_ffn = [_]TensorInfo{
        tensorInfoDims("token_embd.weight", &.{ 8, 4 }),
        tensorInfoDims("output_norm.weight", &.{4}),
        tensorInfoDims("blk.0.attn_norm.weight", &.{4}),
        tensorInfoDims("blk.0.attn_q.weight", &.{ 4, 4 }),
        tensorInfoDims("blk.0.attn_k.weight", &.{ 2, 4 }),
        tensorInfoDims("blk.0.attn_v.weight", &.{ 2, 4 }),
        tensorInfoDims("blk.0.attn_output.weight", &.{ 4, 4 }),
        tensorInfoDims("blk.0.ffn_norm.weight", &.{4}),
        tensorInfoDims("blk.0.ffn_gate.weight", &.{ 7, 4 }),
        tensorInfoDims("blk.0.ffn_up.weight", &.{ 8, 4 }),
        tensorInfoDims("blk.0.ffn_down.weight", &.{ 4, 8 }),
    };
    try std.testing.expectError(error.ShapeMismatch, validateGemma4TextModel(validGemma4Spec(1), &bad_ffn));
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

fn tensorInfoDims(name: []const u8, dims: []const u64) TensorInfo {
    return .{
        .name = name,
        .dims = dims,
        .kind = .q4_0,
        .offset = 0,
        .byte_len = 18,
    };
}

fn validGemma4Spec(block_count: u32) ModelSpec {
    return .{
        .architecture = "gemma4",
        .context_length = 16,
        .embedding_length = 4,
        .block_count = block_count,
        .feed_forward_length = 8,
        .attention_head_count = 2,
        .attention_head_count_kv = 1,
        .rope_dimension_count = 2,
        .vocab_size = 8,
        .tensor_count = 2 + (@as(usize, block_count) * 9),
        .quantization = .{ .has_q4_0 = true },
    };
}
