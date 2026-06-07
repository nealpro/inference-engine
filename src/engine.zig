const std = @import("std");

const benchmark = @import("benchmark.zig");
const gguf = @import("gguf.zig");
const model = @import("model.zig");
const resolver = @import("resolver.zig");
const tokenizer = @import("tokenizer.zig");

pub const CliError = error{
    MissingPrompt,
    MissingModelPath,
    MissingValue,
    InvalidArgument,
    InvalidContextLength,
    InvalidMaxNewTokens,
    UnknownDecodeMode,
    ConflictingBenchmarkOutput,
};

pub const RuntimeError = error{
    MissingModelPath,
    ModelNotFound,
    DecodeModeNotImplemented,
    RealModelCpuExecutionNotImplemented,
    MissingTokenizer,
    UnsupportedTokenizer,
    InvalidModelPath,
    AmbiguousModelDirectory,
    ProjectorModelNotSupported,
} || gguf.GgufError || error{
    OutOfMemory,
    PermissionDenied,
    AccessDenied,
    FileNotFound,
    IsDir,
    NotDir,
    SymLinkLoop,
    ProcessFdQuotaExceeded,
    SystemFdQuotaExceeded,
    NoDevice,
    SystemResources,
    NetworkNotFound,
    NameTooLong,
    BadPathName,
    InvalidUtf8,
    BadMagic,
    UnsupportedVersion,
    Unexpected,
    InputOutput,
    LockViolation,
    WouldBlock,
    NotOpenForReading,
    Unseekable,
};

pub const DecodeMode = enum {
    ar,
    mtp,
    ssd_sim,

    pub fn parse(value: []const u8) CliError!DecodeMode {
        if (std.mem.eql(u8, value, "ar")) return .ar;
        if (std.mem.eql(u8, value, "mtp")) return .mtp;
        if (std.mem.eql(u8, value, "ssd-sim")) return .ssd_sim;
        return error.UnknownDecodeMode;
    }

    pub fn label(self: DecodeMode) []const u8 {
        return switch (self) {
            .ar => "ar",
            .mtp => "mtp",
            .ssd_sim => "ssd-sim",
        };
    }
};

pub const RunOptions = struct {
    model_path: ?[]const u8 = null,
    tokenizer_path: ?[]const u8 = null,
    prompt: []const u8 = "",
    context_length: u32 = 0,
    max_new_tokens: u32 = 128,
    decode_mode: DecodeMode = .ar,
    show_benchmark_contract: bool = false,
    show_benchmark_contract_json: bool = false,
    validate_model: bool = false,

    pub fn parse(args: []const []const u8) CliError!RunOptions {
        var opts = RunOptions{};
        var saw_prompt = false;
        var index: usize = 0;

        while (index < args.len) {
            const arg = args[index];

            if (std.mem.eql(u8, arg, "--model")) {
                opts.model_path = try nextValue(args, &index);
            } else if (std.mem.eql(u8, arg, "--tokenizer")) {
                opts.tokenizer_path = try nextValue(args, &index);
            } else if (std.mem.eql(u8, arg, "--prompt")) {
                opts.prompt = try nextValue(args, &index);
                saw_prompt = true;
            } else if (std.mem.eql(u8, arg, "--ctx")) {
                const raw = try nextValue(args, &index);
                opts.context_length = std.fmt.parseUnsigned(u32, raw, 10) catch {
                    return error.InvalidContextLength;
                };
            } else if (std.mem.eql(u8, arg, "--max-new-tokens")) {
                const raw = try nextValue(args, &index);
                opts.max_new_tokens = std.fmt.parseUnsigned(u32, raw, 10) catch {
                    return error.InvalidMaxNewTokens;
                };
            } else if (std.mem.eql(u8, arg, "--decode")) {
                opts.decode_mode = try DecodeMode.parse(try nextValue(args, &index));
            } else if (std.mem.eql(u8, arg, "--benchmark-contract")) {
                opts.show_benchmark_contract = true;
            } else if (std.mem.eql(u8, arg, "--benchmark-contract-json")) {
                opts.show_benchmark_contract_json = true;
            } else if (std.mem.eql(u8, arg, "--validate-model")) {
                opts.validate_model = true;
            } else {
                return error.InvalidArgument;
            }

            index += 1;
        }

        if (opts.show_benchmark_contract and opts.show_benchmark_contract_json) {
            return error.ConflictingBenchmarkOutput;
        }
        if (opts.show_benchmark_contract or opts.show_benchmark_contract_json) return opts;
        if (opts.model_path == null) return error.MissingModelPath;
        if (!opts.validate_model and (!saw_prompt or opts.prompt.len == 0)) return error.MissingPrompt;
        if (opts.context_length == 0) opts.context_length = 0;
        if (opts.max_new_tokens == 0) return error.InvalidMaxNewTokens;

        return opts;
    }
};

fn nextValue(args: []const []const u8, index: *usize) CliError![]const u8 {
    index.* += 1;
    if (index.* >= args.len) return error.MissingValue;

    const value = args[index.*];
    if (std.mem.startsWith(u8, value, "--")) return error.MissingValue;

    return value;
}

pub const Engine = struct {
    pub fn init() Engine {
        return .{};
    }

    pub fn run(
        self: Engine,
        allocator: std.mem.Allocator,
        io: std.Io,
        env: resolver.Env,
        writer: anytype,
        options: RunOptions,
    ) !void {
        if (options.validate_model) {
            return self.validateModel(allocator, io, env, writer, options);
        }

        if (options.decode_mode != .ar) return error.DecodeModeNotImplemented;

        try self.validateModel(allocator, io, env, writer, options);
        try writer.writeAll(
            "\nreal-weight CPU generation is not wired to external GGUF tensors yet; CPU reference kernels are fixture-backed in tests\n",
        );
        return error.RealModelCpuExecutionNotImplemented;
    }

    pub fn validateModel(
        self: Engine,
        allocator: std.mem.Allocator,
        io: std.Io,
        env: resolver.Env,
        writer: anytype,
        options: RunOptions,
    ) !void {
        _ = self;
        var resolved = try resolver.resolve(allocator, io, options.model_path, options.tokenizer_path, env);
        defer resolved.deinit(allocator);

        var parsed = try gguf.parseFromPath(allocator, io, resolved.model_path);
        defer parsed.deinit(allocator);

        var spec = try parsed.buildModelSpec(allocator);
        defer spec.deinit(allocator);
        try spec.validateForTextInference();
        try model.validateGemma4TextModel(spec, parsed.tensors);

        var tok = if (resolved.tokenizer_path) |path|
            try loadExternalTokenizer(allocator, io, path)
        else
            try tokenizer.Tokenizer.fromGguf(allocator, parsed);
        defer tok.deinit(allocator);
        const formatted_prompt = try tok.applySingleUserChatTemplate(allocator, options.prompt);
        defer allocator.free(formatted_prompt);

        try writer.print(
            \\model validation: ok
            \\contract: {s}
            \\source: {s}
            \\model-path: {s}
            \\target-repo: {s}
            \\target-file: {s}
            \\architecture: {s}
            \\
        ,
            .{
                benchmark.contract_version,
                @tagName(resolved.source),
                resolved.model_path,
                benchmark.target_artifact.hf_repo,
                benchmark.target_artifact.filename,
                spec.architecture,
            },
        );
        if (spec.name) |name| try writer.print("name: {s}\n", .{name});
        if (spec.context_length) |value| try writer.print("context-length: {d}\n", .{value});
        if (spec.embedding_length) |value| try writer.print("embedding-length: {d}\n", .{value});
        if (spec.block_count) |value| try writer.print("block-count: {d}\n", .{value});
        if (spec.feed_forward_length) |value| try writer.print("feed-forward-length: {d}\n", .{value});
        if (spec.attention_head_count) |value| try writer.print("attention-head-count: {d}\n", .{value});
        if (spec.attention_head_count_kv) |value| try writer.print("attention-head-count-kv: {d}\n", .{value});
        if (spec.rope_dimension_count) |value| try writer.print("rope-dimension-count: {d}\n", .{value});
        if (spec.vocab_size) |value| try writer.print("vocab-size: {d}\n", .{value});
        try writer.print("tensor-count: {d}\nquantization: ", .{spec.tensor_count});
        try spec.quantization.write(writer);
        try writer.print(
            "\ntokenizer: {s}\ntoken-count: {d}\nformatted-prompt-bytes: {d}\n",
            .{ @tagName(tok.kind), tok.tokens.len, formatted_prompt.len },
        );
        if (resolved.tokenizer_path) |path| try writer.print("tokenizer-path: {s}\n", .{path});
    }
};

fn loadExternalTokenizer(
    allocator: std.mem.Allocator,
    io: std.Io,
    raw_path: []const u8,
) !tokenizer.Tokenizer {
    const path = if (std.mem.endsWith(u8, raw_path, ".json"))
        try allocator.dupe(u8, raw_path)
    else
        try std.Io.Dir.path.join(allocator, &.{ raw_path, "tokenizer.json" });
    defer allocator.free(path);

    const bytes = try readSmallFile(allocator, io, path, 64 * 1024 * 1024);
    defer allocator.free(bytes);
    return tokenizer.Tokenizer.fromJsonSlice(allocator, bytes);
}

fn readSmallFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    max_bytes: usize,
) ![]u8 {
    var file = if (std.Io.Dir.path.isAbsolute(path))
        try std.Io.Dir.openFileAbsolute(io, path, .{})
    else
        try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    if (stat.size > max_bytes) return error.StreamTooLong;
    const size = std.math.cast(usize, stat.size) orelse return error.StreamTooLong;
    const bytes = try allocator.alloc(u8, size);
    errdefer allocator.free(bytes);
    const read = try file.readPositionalAll(io, bytes, 0);
    if (read != bytes.len) return error.MalformedGguf;
    return bytes;
}

test "CLI requires model for real inference" {
    const args = &[_][]const u8{ "--prompt", "hello" };
    try std.testing.expectError(error.MissingModelPath, RunOptions.parse(args));
}

test "CLI validation requires model but not prompt" {
    const args = &[_][]const u8{ "--validate-model", "--model", "toy" };
    const opts = try RunOptions.parse(args);
    try std.testing.expect(opts.validate_model);
    try std.testing.expectEqualStrings("toy", opts.model_path.?);
}

test "CLI argument parsing accepts scaffold flags" {
    const args = &[_][]const u8{
        "--model",
        "toy",
        "--ctx",
        "8192",
        "--decode",
        "ar",
        "--prompt",
        "hello",
    };

    const opts = try RunOptions.parse(args);

    try std.testing.expectEqualStrings("toy", opts.model_path.?);
    try std.testing.expectEqualStrings("hello", opts.prompt);
    try std.testing.expectEqual(@as(u32, 8192), opts.context_length);
    try std.testing.expectEqual(DecodeMode.ar, opts.decode_mode);
}

test "CLI argument parsing accepts benchmark contract without model" {
    const args = &[_][]const u8{"--benchmark-contract"};
    const opts = try RunOptions.parse(args);

    try std.testing.expect(opts.show_benchmark_contract);
}

test "CLI argument parsing accepts benchmark manifest json without model" {
    const args = &[_][]const u8{"--benchmark-contract-json"};
    const opts = try RunOptions.parse(args);

    try std.testing.expect(opts.show_benchmark_contract_json);
}

test "CLI rejects conflicting benchmark output formats" {
    const args = &[_][]const u8{ "--benchmark-contract", "--benchmark-contract-json" };
    try std.testing.expectError(error.ConflictingBenchmarkOutput, RunOptions.parse(args));
}
