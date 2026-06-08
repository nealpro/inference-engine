//! CLI parsing and top-level orchestration for validation and reports.

const std = @import("std");

const artifact = @import("artifact.zig");
const benchmark = @import("benchmark.zig");
const gguf = @import("gguf.zig");
const model = @import("model.zig");
const resolver = @import("resolver.zig");
const tokenizer = @import("tokenizer.zig");

/// Errors returned while parsing command-line options.
pub const CliError = error{
    MissingPrompt,
    MissingModelPath,
    MissingValue,
    InvalidArgument,
    InvalidContextLength,
    InvalidMaxNewTokens,
    UnknownDecodeMode,
    ConflictingBenchmarkOutput,
    InvalidArtifactRevision,
};

/// Runtime errors that can be surfaced by the engine entry points.
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

/// Decode mode requested for generation.
pub const DecodeMode = enum {
    ar,
    mtp,
    ssd_sim,

    /// Parses the CLI spelling for a decode mode.
    pub fn parse(value: []const u8) CliError!DecodeMode {
        if (std.mem.eql(u8, value, "ar")) return .ar;
        if (std.mem.eql(u8, value, "mtp")) return .mtp;
        if (std.mem.eql(u8, value, "ssd-sim")) return .ssd_sim;
        return error.UnknownDecodeMode;
    }

    /// Returns the CLI label for this decode mode.
    pub fn label(self: DecodeMode) []const u8 {
        return switch (self) {
            .ar => "ar",
            .mtp => "mtp",
            .ssd_sim => "ssd-sim",
        };
    }
};

/// Parsed options for one CLI run.
pub const RunOptions = struct {
    model_path: ?[]const u8 = null,
    tokenizer_path: ?[]const u8 = null,
    prompt: []const u8 = "",
    context_length: u32 = 0,
    max_new_tokens: u32 = 128,
    artifact_revision: ?[]const u8 = null,
    decode_mode: DecodeMode = .ar,
    show_benchmark_contract: bool = false,
    show_benchmark_contract_json: bool = false,
    show_benchmark_report_json: bool = false,
    validate_model: bool = false,

    /// Parses CLI arguments after the executable name.
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
            } else if (std.mem.eql(u8, arg, "--artifact-revision")) {
                opts.artifact_revision = try parseArtifactRevision(try nextValue(args, &index));
            } else if (std.mem.eql(u8, arg, "--decode")) {
                opts.decode_mode = try DecodeMode.parse(try nextValue(args, &index));
            } else if (std.mem.eql(u8, arg, "--benchmark-contract")) {
                opts.show_benchmark_contract = true;
            } else if (std.mem.eql(u8, arg, "--benchmark-contract-json")) {
                opts.show_benchmark_contract_json = true;
            } else if (std.mem.eql(u8, arg, "--benchmark-report-json")) {
                opts.show_benchmark_report_json = true;
            } else if (std.mem.eql(u8, arg, "--validate-model")) {
                opts.validate_model = true;
            } else {
                return error.InvalidArgument;
            }

            index += 1;
        }

        const benchmark_output_count =
            @as(u8, if (opts.show_benchmark_contract) 1 else 0) +
            @as(u8, if (opts.show_benchmark_contract_json) 1 else 0) +
            @as(u8, if (opts.show_benchmark_report_json) 1 else 0);
        if (benchmark_output_count > 1) {
            return error.ConflictingBenchmarkOutput;
        }
        if (opts.show_benchmark_contract or opts.show_benchmark_contract_json) return opts;
        if (opts.model_path == null) return error.MissingModelPath;
        if (!opts.validate_model and !opts.show_benchmark_report_json and (!saw_prompt or opts.prompt.len == 0)) return error.MissingPrompt;
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

fn parseArtifactRevision(value: []const u8) CliError![]const u8 {
    if (value.len == 0 or std.mem.eql(u8, value, "main")) return error.InvalidArtifactRevision;
    return value;
}

/// Facade for validation, report generation, and future model execution.
pub const Engine = struct {
    /// Creates a stateless engine value.
    pub fn init() Engine {
        return .{};
    }

    /// Runs the requested CLI action.
    pub fn run(
        self: Engine,
        allocator: std.mem.Allocator,
        io: std.Io,
        env: resolver.Env,
        writer: anytype,
        options: RunOptions,
    ) !void {
        if (options.show_benchmark_report_json) {
            return self.writeBenchmarkReportJson(allocator, io, env, writer, options);
        }
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

    /// Validates model artifact identity, metadata, tensors, and tokenizer data.
    pub fn validateModel(
        self: Engine,
        allocator: std.mem.Allocator,
        io: std.Io,
        env: resolver.Env,
        writer: anytype,
        options: RunOptions,
    ) !void {
        _ = self;
        var ctx = try loadValidationContext(allocator, io, env, options);
        defer ctx.deinit(allocator);

        const formatted_prompt = try ctx.tokenizer.applySingleUserChatTemplate(allocator, options.prompt);
        defer allocator.free(formatted_prompt);

        try writer.print(
            \\model validation: ok
            \\contract: {s}
            \\source: {s}
            \\model-path: {s}
            \\target-repo: {s}
            \\target-file: {s}
            \\artifact-sha256: {s}
            \\artifact-revision-status: {s}
            \\architecture: {s}
            \\
        ,
            .{
                benchmark.contract_version,
                @tagName(ctx.resolved.source),
                ctx.resolved.model_path,
                benchmark.target_artifact.hf_repo,
                benchmark.target_artifact.filename,
                &ctx.artifact_sha256,
                artifactRevisionStatus(options),
                ctx.spec.architecture,
            },
        );
        if (options.artifact_revision) |revision| try writer.print("artifact-revision: {s}\n", .{revision});
        if (ctx.spec.name) |name| try writer.print("name: {s}\n", .{name});
        if (ctx.spec.context_length) |value| try writer.print("context-length: {d}\n", .{value});
        if (ctx.spec.embedding_length) |value| try writer.print("embedding-length: {d}\n", .{value});
        if (ctx.spec.block_count) |value| try writer.print("block-count: {d}\n", .{value});
        if (ctx.spec.feed_forward_length) |value| try writer.print("feed-forward-length: {d}\n", .{value});
        if (ctx.spec.attention_head_count) |value| try writer.print("attention-head-count: {d}\n", .{value});
        if (ctx.spec.attention_head_count_kv) |value| try writer.print("attention-head-count-kv: {d}\n", .{value});
        if (ctx.spec.rope_dimension_count) |value| try writer.print("rope-dimension-count: {d}\n", .{value});
        if (ctx.spec.vocab_size) |value| try writer.print("vocab-size: {d}\n", .{value});
        try writer.print("tensor-count: {d}\nquantization: ", .{ctx.spec.tensor_count});
        try ctx.spec.quantization.write(writer);
        try writer.print(
            "\ntokenizer: {s}\ntoken-count: {d}\nchat-template-source: {s}\nformatted-prompt-bytes: {d}\n",
            .{ @tagName(ctx.tokenizer.kind), ctx.tokenizer.tokens.len, ctx.chat_template_source, formatted_prompt.len },
        );
        if (ctx.resolved.tokenizer_path) |path| try writer.print("tokenizer-path: {s}\n", .{path});
    }

    /// Writes the validation-only benchmark report JSON skeleton.
    pub fn writeBenchmarkReportJson(
        self: Engine,
        allocator: std.mem.Allocator,
        io: std.Io,
        env: resolver.Env,
        writer: anytype,
        options: RunOptions,
    ) !void {
        _ = self;
        var ctx = try loadValidationContext(allocator, io, env, options);
        defer ctx.deinit(allocator);

        const prompt_bytes = try allocator.alloc(usize, benchmark.prompt_suite.len);
        defer allocator.free(prompt_bytes);
        for (benchmark.prompt_suite, 0..) |case, index| {
            const formatted = try ctx.tokenizer.applySingleUserChatTemplate(allocator, case.prompt);
            defer allocator.free(formatted);
            prompt_bytes[index] = formatted.len;
        }

        try benchmark.writeReportJson(writer, .{
            .model_path = ctx.resolved.model_path,
            .model_source = @tagName(ctx.resolved.source),
            .artifact_sha256 = &ctx.artifact_sha256,
            .artifact_revision = options.artifact_revision,
            .artifact_revision_status = artifactRevisionStatus(options),
            .spec = ctx.spec,
            .tokenizer_kind = @tagName(ctx.tokenizer.kind),
            .tokenizer_count = ctx.tokenizer.tokens.len,
            .tokenizer_bos_id = ctx.tokenizer.bos_id,
            .tokenizer_eos_id = ctx.tokenizer.eos_id,
            .tokenizer_pad_id = ctx.tokenizer.pad_id,
            .chat_template_source = ctx.chat_template_source,
            .formatted_prompt_bytes = prompt_bytes,
        });
    }
};

const ValidationContext = struct {
    resolved: resolver.ResolvedModel,
    parsed: gguf.ParsedGguf,
    spec: model.ModelSpec,
    tokenizer: tokenizer.Tokenizer,
    artifact_sha256: artifact.Sha256Hex,
    chat_template_source: []const u8,

    fn deinit(self: *ValidationContext, allocator: std.mem.Allocator) void {
        self.tokenizer.deinit(allocator);
        self.spec.deinit(allocator);
        self.parsed.deinit(allocator);
        self.resolved.deinit(allocator);
    }
};

fn loadValidationContext(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: resolver.Env,
    options: RunOptions,
) !ValidationContext {
    var resolved = try resolver.resolve(allocator, io, options.model_path, options.tokenizer_path, env);
    errdefer resolved.deinit(allocator);

    const artifact_sha256 = try artifact.sha256FileHex(io, resolved.model_path);

    var parsed = try gguf.parseFromPath(allocator, io, resolved.model_path);
    errdefer parsed.deinit(allocator);

    var spec = try parsed.buildModelSpec(allocator);
    errdefer spec.deinit(allocator);
    try spec.validateForTextInference();
    try model.validateGemma4TextModel(spec, parsed.tensors);

    var tok = if (resolved.tokenizer_path) |path|
        try loadExternalTokenizer(allocator, io, path)
    else
        try tokenizer.Tokenizer.fromGguf(allocator, parsed);
    errdefer tok.deinit(allocator);

    return .{
        .resolved = resolved,
        .parsed = parsed,
        .spec = spec,
        .tokenizer = tok,
        .artifact_sha256 = artifact_sha256,
        .chat_template_source = chatTemplateSource(resolved, tok),
    };
}

fn artifactRevisionStatus(options: RunOptions) []const u8 {
    return if (options.artifact_revision == null) "missing" else "provided";
}

fn chatTemplateSource(resolved: resolver.ResolvedModel, tok: tokenizer.Tokenizer) []const u8 {
    if (tok.chat_template == null) return "fallback";
    return if (resolved.tokenizer_path == null) "metadata" else "external";
}

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

test "CLI argument parsing accepts benchmark report json with model and revision" {
    const args = &[_][]const u8{
        "--benchmark-report-json",
        "--model",
        "toy",
        "--artifact-revision",
        "abc123",
    };
    const opts = try RunOptions.parse(args);
    try std.testing.expect(opts.show_benchmark_report_json);
    try std.testing.expectEqualStrings("toy", opts.model_path.?);
    try std.testing.expectEqualStrings("abc123", opts.artifact_revision.?);
}

test "CLI benchmark report json still requires model" {
    const args = &[_][]const u8{"--benchmark-report-json"};
    try std.testing.expectError(error.MissingModelPath, RunOptions.parse(args));
}

test "CLI rejects benchmark report conflicts" {
    const args = &[_][]const u8{ "--benchmark-contract", "--benchmark-report-json", "--model", "toy" };
    try std.testing.expectError(error.ConflictingBenchmarkOutput, RunOptions.parse(args));
}

test "CLI rejects floating artifact revision" {
    const args = &[_][]const u8{ "--validate-model", "--model", "toy", "--artifact-revision", "main" };
    try std.testing.expectError(error.InvalidArtifactRevision, RunOptions.parse(args));
}

test "CLI rejects empty artifact revision" {
    const args = &[_][]const u8{ "--validate-model", "--model", "toy", "--artifact-revision", "" };
    try std.testing.expectError(error.InvalidArtifactRevision, RunOptions.parse(args));
}

test "benchmark report json validates fixture GGUF without generation" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const bytes = try fixtureGemma4Gguf(std.testing.allocator);
    defer std.testing.allocator.free(bytes);

    var file = try tmp.dir.createFile(io, benchmark.target_artifact.filename, .{ .read = true });
    try file.writePositionalAll(io, bytes, 0);
    file.close(io);

    const path = try std.Io.Dir.path.join(
        std.testing.allocator,
        &.{ ".zig-cache", "tmp", &tmp.sub_path, benchmark.target_artifact.filename },
    );
    defer std.testing.allocator.free(path);

    const expected_sha = try artifact.sha256FileHex(io, path);
    var out: [100000]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);
    try Engine.init().run(std.testing.allocator, io, .{}, &writer, .{
        .model_path = path,
        .show_benchmark_report_json = true,
    });

    const json = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, json, "\"version\": \"benchmark-contract-v1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"status\": \"validation_only\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"revision\": null") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"revision_status\": \"missing\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, &expected_sha) != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"architecture\": \"gemma4\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"block_count\": 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"token_count\": 8") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"bos_id\": 0") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"chat_template_source\": \"metadata\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"id\": \"short_latency\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"formatted_prompt_bytes\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"status\": \"not_run\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"value\": null") != null);
}

fn fixtureGemma4Gguf(allocator: std.mem.Allocator) ![]u8 {
    var bytes: std.ArrayList(u8) = .empty;
    errdefer bytes.deinit(allocator);

    const tensor_names = [_][]const u8{
        "token_embd.weight",
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    };
    const tensor_dims = [_][]const u64{
        &.{ 8, 4 },
        &.{4},
        &.{4},
        &.{ 4, 4 },
        &.{ 2, 4 },
        &.{ 2, 4 },
        &.{ 4, 4 },
        &.{4},
        &.{ 8, 4 },
        &.{ 8, 4 },
        &.{ 4, 8 },
    };

    try bytes.appendSlice(allocator, "GGUF");
    try appendFixtureInt(&bytes, allocator, u32, 3);
    try appendFixtureInt(&bytes, allocator, u64, tensor_names.len);
    try appendFixtureInt(&bytes, allocator, u64, 15);

    try appendFixtureStringMeta(&bytes, allocator, "general.architecture", "gemma4");
    try appendFixtureStringMeta(&bytes, allocator, "general.name", "fixture gemma4");
    try appendFixtureU32Meta(&bytes, allocator, "general.alignment", 32);
    try appendFixtureU32Meta(&bytes, allocator, "gemma4.context_length", 16);
    try appendFixtureU32Meta(&bytes, allocator, "gemma4.embedding_length", 4);
    try appendFixtureU32Meta(&bytes, allocator, "gemma4.block_count", 1);
    try appendFixtureU32Meta(&bytes, allocator, "gemma4.feed_forward_length", 8);
    try appendFixtureU32Meta(&bytes, allocator, "gemma4.attention.head_count", 2);
    try appendFixtureU32Meta(&bytes, allocator, "gemma4.attention.head_count_kv", 1);
    try appendFixtureU32Meta(&bytes, allocator, "gemma4.rope.dimension_count", 2);
    try appendFixtureStringArrayMeta(
        &bytes,
        allocator,
        "tokenizer.ggml.tokens",
        &.{ "<bos>", "<eos>", "<pad>", "<start_of_turn>user", "<end_of_turn>", "<start_of_turn>model", "hello", "world" },
    );
    try appendFixtureU32Meta(&bytes, allocator, "tokenizer.ggml.bos_token_id", 0);
    try appendFixtureU32Meta(&bytes, allocator, "tokenizer.ggml.eos_token_id", 1);
    try appendFixtureU32Meta(&bytes, allocator, "tokenizer.ggml.padding_token_id", 2);
    try appendFixtureStringMeta(&bytes, allocator, "tokenizer.chat_template", "<start_of_turn>user\n{{ .Prompt }}<end_of_turn>\n<start_of_turn>model\n");

    var offset: u64 = 0;
    for (tensor_names, 0..) |name, index| {
        try appendFixtureString(&bytes, allocator, name);
        try appendFixtureInt(&bytes, allocator, u32, tensor_dims[index].len);
        for (tensor_dims[index]) |dim| try appendFixtureInt(&bytes, allocator, u64, dim);
        try appendFixtureInt(&bytes, allocator, u32, 2);
        try appendFixtureInt(&bytes, allocator, u64, offset);
        offset += 18;
    }

    while (bytes.items.len % 32 != 0) try bytes.append(allocator, 0);
    try bytes.appendNTimes(allocator, 0, offset);

    return bytes.toOwnedSlice(allocator);
}

fn appendFixtureStringMeta(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
    try appendFixtureString(bytes, allocator, key);
    try appendFixtureInt(bytes, allocator, u32, 8);
    try appendFixtureString(bytes, allocator, value);
}

fn appendFixtureStringArrayMeta(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, values: []const []const u8) !void {
    try appendFixtureString(bytes, allocator, key);
    try appendFixtureInt(bytes, allocator, u32, 9);
    try appendFixtureInt(bytes, allocator, u32, 8);
    try appendFixtureInt(bytes, allocator, u64, values.len);
    for (values) |value| try appendFixtureString(bytes, allocator, value);
}

fn appendFixtureU32Meta(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, value: u32) !void {
    try appendFixtureString(bytes, allocator, key);
    try appendFixtureInt(bytes, allocator, u32, 4);
    try appendFixtureInt(bytes, allocator, u32, value);
}

fn appendFixtureString(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try appendFixtureInt(bytes, allocator, u64, value.len);
    try bytes.appendSlice(allocator, value);
}

fn appendFixtureInt(bytes: *std.ArrayList(u8), allocator: std.mem.Allocator, comptime T: type, value: anytype) !void {
    var buf: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buf, @intCast(value), .little);
    try bytes.appendSlice(allocator, &buf);
}
