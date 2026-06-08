//! Public library surface for the local inference engine scaffold.

const std = @import("std");

/// Artifact identity helpers.
pub const artifact = @import("artifact.zig");
/// Benchmark contract and report writers.
pub const benchmark = @import("benchmark.zig");
/// Fixture-backed CPU reference kernels.
pub const cpu = @import("cpu.zig");
/// CLI options and top-level engine orchestration.
pub const engine = @import("engine.zig");
/// GGUF metadata and tensor directory parser.
pub const gguf = @import("gguf.zig");
/// Model metadata and Gemma 4 validation helpers.
pub const model = @import("model.zig");
/// Model path and cache alias resolution.
pub const resolver = @import("resolver.zig");
/// Sampling configuration and placeholder generation.
pub const sampler = @import("sampler.zig");
/// Tokenizer adapters used by validation and tests.
pub const tokenizer = @import("tokenizer.zig");

/// Decode mode selected by the CLI.
pub const DecodeMode = engine.DecodeMode;
/// Top-level engine facade.
pub const Engine = engine.Engine;
/// Parsed CLI options for a run.
pub const RunOptions = engine.RunOptions;

test {
    _ = artifact;
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
