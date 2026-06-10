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
    q6_k,
    unknown,

    /// Converts a GGML type id to the local quantization enum.
    pub fn fromGgmlType(value: u32) QuantizationType {
        return switch (value) {
            0 => .f32,
            1 => .f16,
            2 => .q4_0,
            14 => .q6_k,
            else => .unknown,
        };
    }

    /// Returns the user-facing label for this quantization kind.
    pub fn label(self: QuantizationType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .q4_0 => "Q4_0",
            .q6_k => "Q6_K",
            .unknown => "unknown",
        };
    }

    /// True when the tensor kind can be represented by this scaffold.
    pub fn isSupported(self: QuantizationType) bool {
        return switch (self) {
            .f32, .f16, .q4_0, .q6_k => true,
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
    attention_head_count_kv_per_layer: ?[]const u32,
    sliding_window_pattern: ?[]const bool,
    rope_dimension_count: ?u32,
    rope_dimension_count_swa: ?u32,
    attention_key_length: ?u32,
    attention_value_length: ?u32,
    attention_key_length_swa: ?u32,
    attention_value_length_swa: ?u32,
    vocab_size: u32,

    fn fromSpec(spec: ModelSpec) ModelError!Gemma4TextSpecValues {
        return .{
            .block_count = spec.block_count orelse return error.MissingMetadata,
            .embedding_length = spec.embedding_length orelse return error.MissingMetadata,
            .feed_forward_length = spec.feed_forward_length orelse return error.MissingMetadata,
            .attention_head_count = spec.attention_head_count orelse return error.MissingMetadata,
            .attention_head_count_kv = spec.attention_head_count_kv orelse blk: {
                const values = spec.attention_head_count_kv_per_layer orelse return error.MissingMetadata;
                if (values.len == 0) return error.MissingMetadata;
                break :blk values[0];
            },
            .attention_head_count_kv_per_layer = spec.attention_head_count_kv_per_layer,
            .sliding_window_pattern = spec.sliding_window_pattern,
            .rope_dimension_count = spec.rope_dimension_count,
            .rope_dimension_count_swa = spec.rope_dimension_count_swa,
            .attention_key_length = spec.attention_key_length,
            .attention_value_length = spec.attention_value_length,
            .attention_key_length_swa = spec.attention_key_length_swa,
            .attention_value_length_swa = spec.attention_value_length_swa,
            .vocab_size = spec.vocab_size orelse return error.MissingMetadata,
        };
    }

    fn isSlidingLayer(self: Gemma4TextSpecValues, layer: usize) ModelError!bool {
        if (self.sliding_window_pattern) |pattern| {
            if (layer >= pattern.len) return error.MissingMetadata;
            return pattern[layer];
        }
        return true;
    }

    fn kvHeadsForLayer(self: Gemma4TextSpecValues, layer: usize) ModelError!u32 {
        if (self.attention_head_count_kv_per_layer) |values| {
            if (layer >= values.len) return error.MissingMetadata;
            return values[layer];
        }
        return self.attention_head_count_kv;
    }

    fn headDimForLayer(self: Gemma4TextSpecValues, layer: usize) ModelError!u32 {
        const sliding = try self.isSlidingLayer(layer);
        if (sliding) {
            if (self.attention_key_length_swa) |value| return value;
            if (self.rope_dimension_count_swa) |value| return value;
        } else {
            if (self.attention_key_length) |value| return value;
            if (self.rope_dimension_count) |value| return value;
        }
        const kv_heads = try self.kvHeadsForLayer(layer);
        const product = std.math.mul(u32, self.embedding_length, kv_heads) catch return error.MalformedModel;
        if (self.attention_head_count == 0 or product % self.attention_head_count != 0) return error.MalformedModel;
        return product / self.attention_head_count;
    }

    fn valueDimForLayer(self: Gemma4TextSpecValues, layer: usize) ModelError!u32 {
        const sliding = try self.isSlidingLayer(layer);
        if (sliding) {
            if (self.attention_value_length_swa) |value| return value;
        } else {
            if (self.attention_value_length) |value| return value;
        }
        return try self.headDimForLayer(layer);
    }
};

/// Summary of tensor quantization kinds present in a model.
pub const QuantizationSummary = struct {
    has_f32: bool = false,
    has_f16: bool = false,
    has_q4_0: bool = false,
    has_q6_k: bool = false,
    has_unknown: bool = false,

    /// Records one tensor kind in the summary.
    pub fn observe(self: *QuantizationSummary, kind: QuantizationType) void {
        switch (kind) {
            .f32 => self.has_f32 = true,
            .f16 => self.has_f16 = true,
            .q4_0 => self.has_q4_0 = true,
            .q6_k => self.has_q6_k = true,
            .unknown => self.has_unknown = true,
        }
    }

    /// Writes a compact comma-separated quantization summary.
    pub fn write(self: QuantizationSummary, writer: anytype) !void {
        var wrote = false;

        const order = comptime [_]QuantizationType{ .f32, .f16, .q4_0, .q6_k, .unknown };
        inline for (order) |kind| {
            const present = switch (kind) {
                .f32 => self.has_f32,
                .f16 => self.has_f16,
                .q4_0 => self.has_q4_0,
                .q6_k => self.has_q6_k,
                .unknown => self.has_unknown,
            };
            if (present) {
                if (wrote) try writer.writeAll(", ");
                try writer.writeAll(kind.label());
                wrote = true;
            }
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
    attention_head_count_kv_per_layer: ?[]const u32 = null,
    rope_dimension_count: ?u32 = null,
    rope_dimension_count_swa: ?u32 = null,
    attention_key_length: ?u32 = null,
    attention_value_length: ?u32 = null,
    attention_key_length_swa: ?u32 = null,
    attention_value_length_swa: ?u32 = null,
    sliding_window_pattern: ?[]const bool = null,
    rope_freq_base: ?f32 = null,
    rope_freq_base_swa: ?f32 = null,
    attention_layer_norm_rms_epsilon: ?f32 = null,
    final_logit_softcapping: ?f32 = null,
    attention_sliding_window: ?u32 = null,
    vocab_size: ?u32 = null,
    tensor_count: usize,
    quantization: QuantizationSummary,

    /// Releases owned metadata strings.
    pub fn deinit(self: ModelSpec, allocator: std.mem.Allocator) void {
        if (self.name) |name| allocator.free(name);
        allocator.free(self.architecture);
        if (self.attention_head_count_kv_per_layer) |values| allocator.free(values);
        if (self.sliding_window_pattern) |values| allocator.free(values);
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
    if (spec.attention_head_count_kv == null and spec.attention_head_count_kv_per_layer == null) return error.MissingMetadata;
    if (spec.rope_dimension_count == null) return error.MissingMetadata;
    if (spec.vocab_size == null) return error.MissingMetadata;
    if (!spec.quantization.has_q4_0) return error.UnsupportedQuantization;
    try validateOfficialGemma4_12bSpec(spec);
}

/// Validates that required Gemma 4 text tensor families are present.
pub fn validateGemma4TextTensors(tensors: []const TensorInfo) ModelError!void {
    if (!hasTensor(tensors, "token_embd.weight")) return error.MissingTensor;
    if (!hasTensor(tensors, "output_norm.weight")) return error.MissingTensor;

    const required_block_suffixes = [_][]const u8{
        ".attn_norm.weight",
        ".attn_q.weight",
        ".attn_q_norm.weight",
        ".attn_k.weight",
        ".attn_k_norm.weight",
        ".attn_output.weight",
        ".ffn_norm.weight",
        ".ffn_gate.weight",
        ".ffn_up.weight",
        ".ffn_down.weight",
        ".post_attention_norm.weight",
        ".post_ffw_norm.weight",
        ".layer_output_scale.weight",
    };
    inline for (required_block_suffixes) |suffix| {
        if (!hasBlockTensorWithSuffix(tensors, suffix)) return error.MissingTensor;
    }
    if (!hasBlockTensorWithSuffix(tensors, ".attn_v.weight")) return error.MissingTensor;
}

fn validateGemma4TextTensorShapes(spec: ModelSpec, tensors: []const TensorInfo) ModelError!void {
    const values = try Gemma4TextSpecValues.fromSpec(spec);

    const embedding = tensorByName(tensors, "token_embd.weight") orelse return error.MissingTensor;
    try expectMatrixContaining(embedding, values.vocab_size, values.embedding_length);

    const output_norm = tensorByName(tensors, "output_norm.weight") orelse return error.MissingTensor;
    try expectVector(output_norm, values.embedding_length);
    if (tensorByName(tensors, "rope_freqs.weight")) |rope_freqs| {
        if (values.rope_dimension_count_swa) |swa_dim| {
            try expectVector(rope_freqs, swa_dim);
        }
    }

    const block_count = std.math.cast(usize, values.block_count) orelse return error.MalformedModel;
    var name_buf: [128]u8 = undefined;
    for (0..block_count) |layer| {
        const sliding = try values.isSlidingLayer(layer);
        const head_dim = try values.headDimForLayer(layer);
        const value_dim = try values.valueDimForLayer(layer);
        const kv_heads = try values.kvHeadsForLayer(layer);
        const q_width = std.math.mul(u32, values.attention_head_count, head_dim) catch return error.MalformedModel;
        const k_width = std.math.mul(u32, kv_heads, head_dim) catch return error.MalformedModel;
        const v_width = std.math.mul(u32, kv_heads, value_dim) catch return error.MalformedModel;

        try expectLayerVector(tensors, &name_buf, layer, "attn_norm.weight", values.embedding_length);
        try expectLayerVector(tensors, &name_buf, layer, "attn_q_norm.weight", head_dim);
        try expectLayerVector(tensors, &name_buf, layer, "attn_k_norm.weight", head_dim);
        try expectLayerMatrixContaining(tensors, &name_buf, layer, "attn_q.weight", values.embedding_length, q_width);
        try expectLayerMatrixContaining(tensors, &name_buf, layer, "attn_k.weight", values.embedding_length, k_width);
        if (sliding) {
            try expectLayerMatrixContaining(tensors, &name_buf, layer, "attn_v.weight", values.embedding_length, v_width);
        } else if (layerTensorBySuffixOptional(tensors, &name_buf, layer, "attn_v.weight") != null) {
            return error.ShapeMismatch;
        }
        try expectLayerMatrixContaining(tensors, &name_buf, layer, "attn_output.weight", values.embedding_length, q_width);
        try expectLayerVector(tensors, &name_buf, layer, "ffn_norm.weight", values.embedding_length);
        try expectLayerMatrixContaining(tensors, &name_buf, layer, "ffn_gate.weight", values.embedding_length, values.feed_forward_length);
        try expectLayerMatrixContaining(tensors, &name_buf, layer, "ffn_up.weight", values.embedding_length, values.feed_forward_length);
        try expectLayerMatrixContaining(tensors, &name_buf, layer, "ffn_down.weight", values.embedding_length, values.feed_forward_length);
        try expectLayerVector(tensors, &name_buf, layer, "post_attention_norm.weight", values.embedding_length);
        try expectLayerVector(tensors, &name_buf, layer, "post_ffw_norm.weight", values.embedding_length);
        try expectLayerVector(tensors, &name_buf, layer, "layer_output_scale.weight", 1);
    }
}

fn validateOfficialGemma4_12bSpec(spec: ModelSpec) ModelError!void {
    if (!looksLikeOfficialGemma4_12b(spec)) return;
    try expectOptionalU32(spec.block_count, 48);
    try expectOptionalU32(spec.context_length, 262144);
    try expectOptionalU32(spec.embedding_length, 3840);
    try expectOptionalU32(spec.feed_forward_length, 15360);
    try expectOptionalU32(spec.attention_head_count, 16);
    try expectOptionalU32(spec.rope_dimension_count, 512);
    try expectOptionalU32(spec.rope_dimension_count_swa, 256);
    try expectOptionalU32(spec.attention_key_length, 512);
    try expectOptionalU32(spec.attention_value_length, 512);
    try expectOptionalU32(spec.attention_key_length_swa, 256);
    try expectOptionalU32(spec.attention_value_length_swa, 256);
    try expectOptionalU32(spec.vocab_size, 262144);

    const block_count = spec.block_count orelse return error.MissingMetadata;
    const block_count_usize = std.math.cast(usize, block_count) orelse return error.MalformedModel;
    const kv_heads = spec.attention_head_count_kv_per_layer orelse return error.MissingMetadata;
    const sliding_pattern = spec.sliding_window_pattern orelse return error.MissingMetadata;
    if (kv_heads.len != block_count_usize or sliding_pattern.len != block_count_usize) return error.MissingMetadata;
    for (0..block_count_usize) |layer| {
        const expected_sliding = ((layer + 1) % 6) != 0;
        if (sliding_pattern[layer] != expected_sliding) return error.MalformedModel;
        const expected_kv_heads: u32 = if (expected_sliding) 8 else 1;
        if (kv_heads[layer] != expected_kv_heads) return error.MalformedModel;
    }
}

fn looksLikeOfficialGemma4_12b(spec: ModelSpec) bool {
    return spec.block_count == 48 or
        spec.embedding_length == 3840 or
        spec.vocab_size == 262144;
}

fn expectOptionalU32(value: ?u32, expected: u32) ModelError!void {
    if (value == null) return error.MissingMetadata;
    if (value.? != expected) return error.MalformedModel;
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

fn layerTensorBySuffixOptional(tensors: []const TensorInfo, name_buf: []u8, layer: usize, suffix: []const u8) ?TensorInfo {
    const name = std.fmt.bufPrint(name_buf, "blk.{d}.{s}", .{ layer, suffix }) catch return null;
    return tensorByName(tensors, name);
}

fn expectLayerVector(tensors: []const TensorInfo, name_buf: []u8, layer: usize, suffix: []const u8, expected: u32) ModelError!void {
    try expectVector(try layerTensorBySuffix(tensors, name_buf, layer, suffix), expected);
}

fn expectLayerMatrixContaining(tensors: []const TensorInfo, name_buf: []u8, layer: usize, suffix: []const u8, a: u32, b: u32) ModelError!void {
    try expectMatrixContaining(try layerTensorBySuffix(tensors, name_buf, layer, suffix), a, b);
}

fn expectVector(tensor: TensorInfo, len: u32) ModelError!void {
    if (tensor.dims.len != 1 or tensor.dims[0] != len) return error.ShapeMismatch;
}

fn expectMatrixContaining(tensor: TensorInfo, a: u32, b: u32) ModelError!void {
    if (tensor.dims.len != 2) return error.ShapeMismatch;
    if (!dimsContain(tensor.dims, a) or !dimsContain(tensor.dims, b)) return error.ShapeMismatch;
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
        .q4_0 => quantizedTensorByteLen(element_count, 32, 18),
        .q6_k => quantizedTensorByteLen(element_count, 256, 210),
        .unknown => error.UnsupportedQuantization,
    };
}

fn quantizedTensorByteLen(element_count: u64, comptime values_per_block: u64, comptime bytes_per_block: u64) ModelError!u64 {
    if (element_count == 0) return error.MalformedModel;
    const blocks = (element_count + values_per_block - 1) / values_per_block;
    return std.math.mul(u64, blocks, bytes_per_block) catch error.MalformedModel;
}

test "quantization labels and support" {
    try std.testing.expectEqual(QuantizationType.f32, QuantizationType.fromGgmlType(0));
    try std.testing.expectEqual(QuantizationType.f16, QuantizationType.fromGgmlType(1));
    try std.testing.expectEqual(QuantizationType.q4_0, QuantizationType.fromGgmlType(2));
    try std.testing.expectEqual(QuantizationType.q6_k, QuantizationType.fromGgmlType(14));
    try std.testing.expectEqual(QuantizationType.unknown, QuantizationType.fromGgmlType(999));
    try std.testing.expect(QuantizationType.q4_0.isSupported());
    try std.testing.expect(QuantizationType.q6_k.isSupported());
    try std.testing.expect(!QuantizationType.unknown.isSupported());
    try std.testing.expectEqualStrings("Q6_K", QuantizationType.q6_k.label());
}

test "tensor byte length supports quantized blocks" {
    try std.testing.expectEqual(@as(u64, 18), try tensorByteLen(.q4_0, 32));
    try std.testing.expectEqual(@as(u64, 36), try tensorByteLen(.q4_0, 33));
    try std.testing.expectEqual(@as(u64, 210), try tensorByteLen(.q6_k, 256));
    try std.testing.expectEqual(@as(u64, 420), try tensorByteLen(.q6_k, 257));
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
        tensorInfo("blk.0.attn_q_norm.weight"),
        tensorInfo("blk.0.attn_k.weight"),
        tensorInfo("blk.0.attn_k_norm.weight"),
        tensorInfo("blk.0.attn_v.weight"),
        tensorInfo("blk.0.attn_output.weight"),
        tensorInfo("blk.0.ffn_norm.weight"),
        tensorInfo("blk.0.ffn_gate.weight"),
        tensorInfo("blk.0.ffn_up.weight"),
        tensorInfo("blk.0.ffn_down.weight"),
        tensorInfo("blk.0.post_attention_norm.weight"),
        tensorInfo("blk.0.post_ffw_norm.weight"),
        tensorInfo("blk.0.layer_output_scale.weight"),
    };
    try validateGemma4TextTensors(&tensors);

    try std.testing.expectError(error.MissingTensor, validateGemma4TextTensors(tensors[0 .. tensors.len - 1]));
}

test "Gemma 4 text model validator checks every layer and shapes" {
    const tensors = topLevelTensorInfos(&.{ 8, 4 }) ++
        slidingLayerTensorInfos(0, .{}) ++
        slidingLayerTensorInfos(1, .{});
    try validateGemma4TextModel(validGemma4Spec(2), &tensors);

    try std.testing.expectError(error.MissingTensor, validateGemma4TextModel(
        validGemma4Spec(2),
        tensors[0 .. tensors.len - 1],
    ));
}

test "Gemma 4 text validator accepts full k-equals-v layers without v projection" {
    const tensors = topLevelTensorInfos(&.{ 8, 4 }) ++
        slidingLayerTensorInfos(0, .{}) ++
        fullAttentionLayerTensorInfos(1);
    try validateGemma4TextModel(validMixedGemma4Spec(), &tensors);
}

test "Gemma 4 text model validator rejects bad top-level and layer shapes" {
    const bad_embedding = topLevelTensorInfos(&.{ 8, 5 }) ++ slidingLayerTensorInfos(0, .{});
    try std.testing.expectError(error.ShapeMismatch, validateGemma4TextModel(validGemma4Spec(1), &bad_embedding));

    const bad_norm = topLevelTensorInfos(&.{ 8, 4 }) ++
        slidingLayerTensorInfos(0, .{ .attn_norm = &.{ 4, 1 } });
    try std.testing.expectError(error.ShapeMismatch, validateGemma4TextModel(validGemma4Spec(1), &bad_norm));

    const bad_ffn = topLevelTensorInfos(&.{ 8, 4 }) ++
        slidingLayerTensorInfos(0, .{ .ffn_gate = &.{ 7, 4 } });
    try std.testing.expectError(error.ShapeMismatch, validateGemma4TextModel(validGemma4Spec(1), &bad_ffn));
}

const SlidingLayerDims = struct {
    attn_norm: []const u64 = &.{4},
    attn_q: []const u64 = &.{ 4, 4 },
    attn_q_norm: []const u64 = &.{2},
    attn_k: []const u64 = &.{ 2, 4 },
    attn_k_norm: []const u64 = &.{2},
    attn_v: []const u64 = &.{ 2, 4 },
    attn_output: []const u64 = &.{ 4, 4 },
    ffn_norm: []const u64 = &.{4},
    ffn_gate: []const u64 = &.{ 8, 4 },
    ffn_up: []const u64 = &.{ 8, 4 },
    ffn_down: []const u64 = &.{ 4, 8 },
    post_attention_norm: []const u64 = &.{4},
    post_ffw_norm: []const u64 = &.{4},
    layer_output_scale: []const u64 = &.{1},
};

fn topLevelTensorInfos(comptime token_embedding_dims: []const u64) [2]TensorInfo {
    return .{
        tensorInfoDims("token_embd.weight", token_embedding_dims),
        tensorInfoDims("output_norm.weight", &.{4}),
    };
}

fn slidingLayerTensorInfos(comptime layer: usize, comptime dims: SlidingLayerDims) [14]TensorInfo {
    return .{
        layerTensorInfo(layer, "attn_norm.weight", dims.attn_norm),
        layerTensorInfo(layer, "attn_q.weight", dims.attn_q),
        layerTensorInfo(layer, "attn_q_norm.weight", dims.attn_q_norm),
        layerTensorInfo(layer, "attn_k.weight", dims.attn_k),
        layerTensorInfo(layer, "attn_k_norm.weight", dims.attn_k_norm),
        layerTensorInfo(layer, "attn_v.weight", dims.attn_v),
        layerTensorInfo(layer, "attn_output.weight", dims.attn_output),
        layerTensorInfo(layer, "ffn_norm.weight", dims.ffn_norm),
        layerTensorInfo(layer, "ffn_gate.weight", dims.ffn_gate),
        layerTensorInfo(layer, "ffn_up.weight", dims.ffn_up),
        layerTensorInfo(layer, "ffn_down.weight", dims.ffn_down),
        layerTensorInfo(layer, "post_attention_norm.weight", dims.post_attention_norm),
        layerTensorInfo(layer, "post_ffw_norm.weight", dims.post_ffw_norm),
        layerTensorInfo(layer, "layer_output_scale.weight", dims.layer_output_scale),
    };
}

fn fullAttentionLayerTensorInfos(comptime layer: usize) [13]TensorInfo {
    return .{
        layerTensorInfo(layer, "attn_norm.weight", &.{4}),
        layerTensorInfo(layer, "attn_q.weight", &.{ 8, 4 }),
        layerTensorInfo(layer, "attn_q_norm.weight", &.{4}),
        layerTensorInfo(layer, "attn_k.weight", &.{ 4, 4 }),
        layerTensorInfo(layer, "attn_k_norm.weight", &.{4}),
        layerTensorInfo(layer, "attn_output.weight", &.{ 8, 4 }),
        layerTensorInfo(layer, "ffn_norm.weight", &.{4}),
        layerTensorInfo(layer, "ffn_gate.weight", &.{ 8, 4 }),
        layerTensorInfo(layer, "ffn_up.weight", &.{ 8, 4 }),
        layerTensorInfo(layer, "ffn_down.weight", &.{ 4, 8 }),
        layerTensorInfo(layer, "post_attention_norm.weight", &.{4}),
        layerTensorInfo(layer, "post_ffw_norm.weight", &.{4}),
        layerTensorInfo(layer, "layer_output_scale.weight", &.{1}),
    };
}

fn layerTensorInfo(comptime layer: usize, comptime suffix: []const u8, comptime dims: []const u64) TensorInfo {
    return tensorInfoDims(std.fmt.comptimePrint("blk.{d}.{s}", .{ layer, suffix }), dims);
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

fn validMixedGemma4Spec() ModelSpec {
    return .{
        .architecture = "gemma4",
        .context_length = 16,
        .embedding_length = 4,
        .block_count = 2,
        .feed_forward_length = 8,
        .attention_head_count = 2,
        .attention_head_count_kv_per_layer = &.{ 1, 1 },
        .rope_dimension_count = 4,
        .rope_dimension_count_swa = 2,
        .attention_key_length = 4,
        .attention_value_length = 4,
        .attention_key_length_swa = 2,
        .attention_value_length_swa = 2,
        .sliding_window_pattern = &.{ true, false },
        .vocab_size = 8,
        .tensor_count = 29,
        .quantization = .{ .has_q4_0 = true },
    };
}
