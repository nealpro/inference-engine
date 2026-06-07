const std = @import("std");

pub const CpuError = error{
    ContextOverflow,
    DimensionMismatch,
    InvalidToken,
};

pub fn dequantizeQ4_0Block(out: *[32]f32, block: *const [18]u8) void {
    const scale_bits = std.mem.readInt(u16, block[0..2], .little);
    const scale: f16 = @bitCast(scale_bits);
    const scale32: f32 = @floatCast(scale);

    for (0..16) |i| {
        const byte = block[2 + i];
        const low = @as(i8, @intCast(byte & 0x0f)) - 8;
        const high = @as(i8, @intCast((byte >> 4) & 0x0f)) - 8;
        out[i] = @as(f32, @floatFromInt(low)) * scale32;
        out[i + 16] = @as(f32, @floatFromInt(high)) * scale32;
    }
}

pub fn matVec(out: []f32, matrix: []const f32, rows: usize, cols: usize, input: []const f32) CpuError!void {
    if (out.len != rows or input.len != cols or matrix.len != rows * cols) return error.DimensionMismatch;
    for (0..rows) |row| {
        var sum: f32 = 0.0;
        for (0..cols) |col| {
            sum += matrix[row * cols + col] * input[col];
        }
        out[row] = sum;
    }
}

pub fn rmsNorm(out: []f32, input: []const f32, weight: []const f32, eps: f32) CpuError!void {
    if (out.len != input.len or weight.len != input.len) return error.DimensionMismatch;
    var mean_square: f32 = 0.0;
    for (input) |value| mean_square += value * value;
    mean_square /= @floatFromInt(input.len);
    const scale = 1.0 / @sqrt(mean_square + eps);
    for (input, 0..) |value, i| {
        out[i] = value * scale * weight[i];
    }
}

pub fn applyRoPE(values: []f32, position: usize, theta: f32) CpuError!void {
    if (values.len % 2 != 0) return error.DimensionMismatch;
    for (0..values.len / 2) |i| {
        const pair = i * 2;
        const freq = std.math.pow(f32, theta, -@as(f32, @floatFromInt(pair)) / @as(f32, @floatFromInt(values.len)));
        const angle = @as(f32, @floatFromInt(position)) * freq;
        const c = @cos(angle);
        const s = @sin(angle);
        const x0 = values[pair];
        const x1 = values[pair + 1];
        values[pair] = x0 * c - x1 * s;
        values[pair + 1] = x0 * s + x1 * c;
    }
}

pub fn softmax(values: []f32) void {
    if (values.len == 0) return;
    var max_value = values[0];
    for (values[1..]) |value| max_value = @max(max_value, value);
    var sum: f32 = 0.0;
    for (values) |*value| {
        value.* = @exp(value.* - max_value);
        sum += value.*;
    }
    for (values) |*value| value.* /= sum;
}

pub fn argmax(values: []const f32) u32 {
    var best_index: usize = 0;
    var best_value = values[0];
    for (values[1..], 1..) |value, index| {
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    return @intCast(best_index);
}

pub const KVCache = struct {
    layer_count: usize,
    context_length: usize,
    hidden_size: usize,
    keys: []f32,
    values: []f32,
    used: usize = 0,

    pub fn init(allocator: std.mem.Allocator, layer_count: usize, context_length: usize, hidden_size: usize) !KVCache {
        const total = try std.math.mul(usize, try std.math.mul(usize, layer_count, context_length), hidden_size);
        const keys = try allocator.alloc(f32, total);
        errdefer allocator.free(keys);
        const values = try allocator.alloc(f32, total);
        @memset(keys, 0.0);
        @memset(values, 0.0);
        return .{
            .layer_count = layer_count,
            .context_length = context_length,
            .hidden_size = hidden_size,
            .keys = keys,
            .values = values,
        };
    }

    pub fn deinit(self: KVCache, allocator: std.mem.Allocator) void {
        allocator.free(self.keys);
        allocator.free(self.values);
    }

    pub fn append(self: *KVCache, layer: usize, position: usize, key: []const f32, value: []const f32) CpuError!void {
        if (layer >= self.layer_count or position >= self.context_length) return error.ContextOverflow;
        if (key.len != self.hidden_size or value.len != self.hidden_size) return error.DimensionMismatch;
        const start = self.index(layer, position);
        @memcpy(self.keys[start .. start + self.hidden_size], key);
        @memcpy(self.values[start .. start + self.hidden_size], value);
        self.used = @max(self.used, position + 1);
    }

    pub fn keyAt(self: KVCache, layer: usize, position: usize) []const f32 {
        const start = self.index(layer, position);
        return self.keys[start .. start + self.hidden_size];
    }

    pub fn valueAt(self: KVCache, layer: usize, position: usize) []const f32 {
        const start = self.index(layer, position);
        return self.values[start .. start + self.hidden_size];
    }

    fn index(self: KVCache, layer: usize, position: usize) usize {
        return ((layer * self.context_length) + position) * self.hidden_size;
    }
};

pub const ReferenceConfig = struct {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    layer_count: usize,
    context_length: usize,
    rope_theta: f32 = 10000.0,
    rms_eps: f32 = 1e-5,
};

pub const ReferenceWeights = struct {
    token_embedding: []const f32,
    attn_norm: []const f32,
    wq: []const f32,
    wk: []const f32,
    wv: []const f32,
    wo: []const f32,
    ffn_norm: []const f32,
    w1: []const f32,
    w2: []const f32,
    w3: []const f32,
    lm_head: []const f32,
};

pub const ReferenceModel = struct {
    config: ReferenceConfig,
    weights: ReferenceWeights,

    pub fn generate(
        self: ReferenceModel,
        allocator: std.mem.Allocator,
        prompt: []const u32,
        max_new_tokens: usize,
    ) ![]u32 {
        if (prompt.len == 0) return error.InvalidToken;
        var cache = try KVCache.init(allocator, self.config.layer_count, self.config.context_length, self.config.hidden_size);
        defer cache.deinit(allocator);

        var generated: std.ArrayList(u32) = .empty;
        errdefer generated.deinit(allocator);

        var pos: usize = 0;
        const logits = try allocator.alloc(f32, self.config.vocab_size);
        defer allocator.free(logits);

        for (prompt) |token| {
            try self.forwardOne(allocator, &cache, token, pos, logits);
            pos += 1;
        }

        var next = argmax(logits);
        for (0..max_new_tokens) |_| {
            try generated.append(allocator, next);
            try self.forwardOne(allocator, &cache, next, pos, logits);
            pos += 1;
            next = argmax(logits);
        }

        return generated.toOwnedSlice(allocator);
    }

    fn forwardOne(
        self: ReferenceModel,
        allocator: std.mem.Allocator,
        cache: *KVCache,
        token: u32,
        position: usize,
        logits: []f32,
    ) !void {
        const cfg = self.config;
        if (token >= cfg.vocab_size) return error.InvalidToken;
        if (position >= cfg.context_length) return error.ContextOverflow;

        const x = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(x);
        @memcpy(x, self.weights.token_embedding[token * cfg.hidden_size .. (token + 1) * cfg.hidden_size]);

        const norm = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(norm);
        const q = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(v);
        const attn = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(attn);
        const projected = try allocator.alloc(f32, cfg.hidden_size);
        defer allocator.free(projected);
        var scores = try allocator.alloc(f32, position + 1);
        defer allocator.free(scores);
        const gate = try allocator.alloc(f32, cfg.intermediate_size);
        defer allocator.free(gate);
        const up = try allocator.alloc(f32, cfg.intermediate_size);
        defer allocator.free(up);
        const hidden = try allocator.alloc(f32, cfg.intermediate_size);
        defer allocator.free(hidden);

        for (0..cfg.layer_count) |layer| {
            try rmsNorm(norm, x, layerSlice(self.weights.attn_norm, layer, cfg.hidden_size), cfg.rms_eps);
            try matVec(q, layerMatrix(self.weights.wq, layer, cfg.hidden_size, cfg.hidden_size), cfg.hidden_size, cfg.hidden_size, norm);
            try matVec(k, layerMatrix(self.weights.wk, layer, cfg.hidden_size, cfg.hidden_size), cfg.hidden_size, cfg.hidden_size, norm);
            try matVec(v, layerMatrix(self.weights.wv, layer, cfg.hidden_size, cfg.hidden_size), cfg.hidden_size, cfg.hidden_size, norm);
            try applyRoPE(q, position, cfg.rope_theta);
            try applyRoPE(k, position, cfg.rope_theta);
            try cache.append(layer, position, k, v);

            const inv_sqrt = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)));
            for (0..position + 1) |past| {
                var dot: f32 = 0.0;
                const past_key = cache.keyAt(layer, past);
                for (q, 0..) |value, i| dot += value * past_key[i];
                scores[past] = dot * inv_sqrt;
            }
            softmax(scores);
            @memset(attn, 0.0);
            for (0..position + 1) |past| {
                const past_value = cache.valueAt(layer, past);
                for (attn, 0..) |*value, i| value.* += scores[past] * past_value[i];
            }
            try matVec(projected, layerMatrix(self.weights.wo, layer, cfg.hidden_size, cfg.hidden_size), cfg.hidden_size, cfg.hidden_size, attn);
            for (x, 0..) |*value, i| value.* += projected[i];

            try rmsNorm(norm, x, layerSlice(self.weights.ffn_norm, layer, cfg.hidden_size), cfg.rms_eps);
            try matVec(gate, layerMatrix(self.weights.w1, layer, cfg.intermediate_size, cfg.hidden_size), cfg.intermediate_size, cfg.hidden_size, norm);
            try matVec(up, layerMatrix(self.weights.w3, layer, cfg.intermediate_size, cfg.hidden_size), cfg.intermediate_size, cfg.hidden_size, norm);
            for (hidden, 0..) |*value, i| value.* = silu(gate[i]) * up[i];
            try matVec(projected, layerMatrix(self.weights.w2, layer, cfg.hidden_size, cfg.intermediate_size), cfg.hidden_size, cfg.intermediate_size, hidden);
            for (x, 0..) |*value, i| value.* += projected[i];
        }

        try matVec(logits, self.weights.lm_head, cfg.vocab_size, cfg.hidden_size, x);
    }
};

fn layerSlice(values: []const f32, layer: usize, len: usize) []const f32 {
    return values[layer * len .. (layer + 1) * len];
}

fn layerMatrix(values: []const f32, layer: usize, rows: usize, cols: usize) []const f32 {
    const len = rows * cols;
    return values[layer * len .. (layer + 1) * len];
}

fn silu(value: f32) f32 {
    return value / (1.0 + @exp(-value));
}

test "q4_0 dequantizes packed nibbles" {
    var block = [_]u8{0} ** 18;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 2.0)), .little);
    block[2] = 0x8f;

    var out: [32]f32 = undefined;
    dequantizeQ4_0Block(&out, &block);

    try std.testing.expectEqual(@as(f32, 14.0), out[0]);
    try std.testing.expectEqual(@as(f32, 0.0), out[16]);
}

test "rms norm and matvec produce expected values" {
    var norm_out: [2]f32 = undefined;
    try rmsNorm(&norm_out, &.{ 3.0, 4.0 }, &.{ 1.0, 1.0 }, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.84852815), norm_out[0], 0.00001);

    var mat_out: [2]f32 = undefined;
    try matVec(&mat_out, &.{ 1.0, 2.0, 3.0, 4.0 }, 2, 2, &.{ 1.0, 1.0 });
    try std.testing.expectEqualSlices(f32, &.{ 3.0, 7.0 }, &mat_out);
}

test "kv cache appends and rejects overflow" {
    var cache = try KVCache.init(std.testing.allocator, 1, 2, 2);
    defer cache.deinit(std.testing.allocator);

    try cache.append(0, 0, &.{ 1.0, 2.0 }, &.{ 3.0, 4.0 });
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0 }, cache.keyAt(0, 0));
    try std.testing.expectError(error.ContextOverflow, cache.append(0, 2, &.{ 0.0, 0.0 }, &.{ 0.0, 0.0 }));
}

test "tiny reference model generates deterministic token" {
    const cfg = ReferenceConfig{
        .vocab_size = 3,
        .hidden_size = 2,
        .intermediate_size = 2,
        .layer_count = 1,
        .context_length = 8,
    };
    const zeros4 = [_]f32{0.0} ** 4;
    const ones2 = [_]f32{ 1.0, 1.0 };
    const embeddings = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    };
    const lm_head = [_]f32{
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
    };
    const weights = ReferenceWeights{
        .token_embedding = &embeddings,
        .attn_norm = &ones2,
        .wq = &zeros4,
        .wk = &zeros4,
        .wv = &zeros4,
        .wo = &zeros4,
        .ffn_norm = &ones2,
        .w1 = &zeros4,
        .w2 = &zeros4,
        .w3 = &zeros4,
        .lm_head = &lm_head,
    };
    const runtime = ReferenceModel{ .config = cfg, .weights = weights };
    const generated = try runtime.generate(std.testing.allocator, &.{0}, 1);
    defer std.testing.allocator.free(generated);
    try std.testing.expectEqualSlices(u32, &.{1}, generated);
}
