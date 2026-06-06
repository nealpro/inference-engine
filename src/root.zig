const std = @import("std");

pub const engine = @import("engine.zig");
pub const model = @import("model.zig");
pub const sampler = @import("sampler.zig");
pub const tokenizer = @import("tokenizer.zig");

pub const DecodeMode = engine.DecodeMode;
pub const Engine = engine.Engine;
pub const RunOptions = engine.RunOptions;

test {
    _ = engine;
    _ = model;
    _ = sampler;
    _ = tokenizer;
}

test "public model defaults" {
    const cfg = model.ModelConfig.default();
    try std.testing.expectEqualStrings("gemma-4-12b-q4", cfg.name);
    try std.testing.expectEqual(@as(u32, 8192), cfg.context_length);
}

test "public CLI parsing" {
    const args = &[_][]const u8{ "--prompt", "hello", "--decode", "ar" };
    const opts = try RunOptions.parse(args);

    try std.testing.expectEqualStrings("hello", opts.prompt);
    try std.testing.expectEqual(DecodeMode.ar, opts.decode_mode);
}

test "public placeholder generation" {
    const args = &[_][]const u8{ "--prompt", "hello" };
    const opts = try RunOptions.parse(args);
    const runner = Engine.init(opts);

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);
    try runner.generate(&writer, opts);

    try std.testing.expect(std.mem.indexOf(
        u8,
        out[0..writer.end],
        "placeholder response for prompt (5 bytes): hello",
    ) != null);
}
