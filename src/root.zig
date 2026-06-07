const std = @import("std");

pub const benchmark = @import("benchmark.zig");
pub const cpu = @import("cpu.zig");
pub const engine = @import("engine.zig");
pub const gguf = @import("gguf.zig");
pub const model = @import("model.zig");
pub const resolver = @import("resolver.zig");
pub const sampler = @import("sampler.zig");
pub const tokenizer = @import("tokenizer.zig");

pub const DecodeMode = engine.DecodeMode;
pub const Engine = engine.Engine;
pub const RunOptions = engine.RunOptions;

test {
    _ = benchmark;
    _ = cpu;
    _ = engine;
    _ = gguf;
    _ = model;
    _ = resolver;
    _ = sampler;
    _ = tokenizer;
}

test "public model types avoid hardcoded runtime identity" {
    try std.testing.expectEqualStrings("Q4_0", model.QuantizationType.q4_0.label());
}

test "public CLI parsing" {
    const args = &[_][]const u8{ "--model", "toy", "--prompt", "hello", "--decode", "ar" };
    const opts = try RunOptions.parse(args);

    try std.testing.expectEqualStrings("toy", opts.model_path.?);
    try std.testing.expectEqualStrings("hello", opts.prompt);
    try std.testing.expectEqual(DecodeMode.ar, opts.decode_mode);
}

test "public benchmark contract" {
    try std.testing.expect(benchmark.prompt_suite.len >= 5);
}
