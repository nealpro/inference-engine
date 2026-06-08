//! Command-line entry point for the inference engine scaffold.

const std = @import("std");
const Io = std.Io;

const inference = @import("inference_engine");

/// Parses CLI options, dispatches benchmark output, or runs validation.
pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    const options = inference.RunOptions.parse(args[1..]) catch |err| {
        try printUsage(init.io, err);
        std.process.exit(1);
    };

    const runner = inference.Engine.init();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const stdout = &stdout_file_writer.interface;

    if (options.show_benchmark_contract) {
        try inference.benchmark.writeContract(stdout);
        try stdout.flush();
        return;
    }
    if (options.show_benchmark_contract_json) {
        try inference.benchmark.writeManifestJson(stdout);
        try stdout.flush();
        return;
    }

    const env = inference.resolver.Env{
        .model_cache = init.environ_map.get("INFERENCE_ENGINE_MODEL_CACHE"),
        .home = init.environ_map.get("HOME"),
    };
    runner.run(init.gpa, init.io, env, stdout, options) catch |err| {
        try stdout.flush();
        try printRuntimeError(init.io, err);
        std.process.exit(1);
    };
    try stdout.flush();
}

fn printUsage(io: Io, err: anyerror) !void {
    var stderr_buffer: [1024]u8 = undefined;
    var stderr_file_writer: Io.File.Writer = .init(.stderr(), io, &stderr_buffer);
    const stderr = &stderr_file_writer.interface;

    try stderr.print("error: {s}\n\n", .{@errorName(err)});
    try stderr.writeAll(
        \\usage:
        \\  inference_engine --model <path-or-cache-alias> --prompt <text> [--tokenizer <path-or-dir>] [--ctx <tokens>] [--max-new-tokens <n>] [--decode ar|mtp|ssd-sim]
        \\  inference_engine --validate-model --model <path-or-cache-alias> [--tokenizer <path-or-dir>] [--artifact-revision <rev>]
        \\  inference_engine --benchmark-contract
        \\  inference_engine --benchmark-contract-json
        \\  inference_engine --benchmark-report-json --model <path-or-cache-alias> [--tokenizer <path-or-dir>] [--artifact-revision <rev>]
        \\
        \\example:
        \\  inference_engine --model /path/to/model.gguf --ctx 8192 --decode ar --prompt "hello"
        \\
    );
    try stderr.flush();
}

fn printRuntimeError(io: Io, err: anyerror) !void {
    var stderr_buffer: [1024]u8 = undefined;
    var stderr_file_writer: Io.File.Writer = .init(.stderr(), io, &stderr_buffer);
    const stderr = &stderr_file_writer.interface;

    try stderr.print("error: {s}\n", .{@errorName(err)});
    switch (err) {
        error.MissingModelPath => try stderr.writeAll("provide --model <path-or-cache-alias>\n"),
        error.ModelNotFound => try stderr.writeAll("model was not found at the provided path or model cache alias\n"),
        error.AmbiguousModelDirectory => try stderr.writeAll("model directory contains multiple text GGUF files; pass the exact file path\n"),
        error.ProjectorModelNotSupported => try stderr.writeAll("mmproj projector GGUF files are out of scope for text-only Gemma 4 inference\n"),
        error.DecodeModeNotImplemented => try stderr.writeAll("only --decode ar is implemented for real model execution in this milestone\n"),
        error.RealModelCpuExecutionNotImplemented => try stderr.writeAll("metadata ingest and CPU reference kernels are present, but external GGUF tensor execution is not complete\n"),
        error.UnsupportedArchitecture => try stderr.writeAll("only Gemma 4 text GGUF artifacts are supported in this milestone\n"),
        error.MissingTensor => try stderr.writeAll("model is missing one or more required Gemma 4 text tensor families\n"),
        error.MissingMetadata => try stderr.writeAll("model is missing required Gemma 4 text metadata\n"),
        else => {},
    }
    try stderr.flush();
}

test "executable imports library scaffold" {
    try std.testing.expectEqual(inference.DecodeMode.ar, try inference.DecodeMode.parse("ar"));
}
