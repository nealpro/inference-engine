const std = @import("std");

const model = @import("model.zig");
const sampler = @import("sampler.zig");
const tokenizer = @import("tokenizer.zig");

pub const CliError = error{
    MissingPrompt,
    MissingValue,
    InvalidArgument,
    InvalidContextLength,
    UnknownDecodeMode,
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
    model_name: []const u8 = "gemma-4-12b-q4",
    prompt: []const u8 = "",
    context_length: u32 = 8192,
    decode_mode: DecodeMode = .ar,

    pub fn parse(args: []const []const u8) CliError!RunOptions {
        var opts = RunOptions{};
        var saw_prompt = false;
        var index: usize = 0;

        while (index < args.len) {
            const arg = args[index];

            if (std.mem.eql(u8, arg, "--model")) {
                opts.model_name = try nextValue(args, &index);
            } else if (std.mem.eql(u8, arg, "--prompt")) {
                opts.prompt = try nextValue(args, &index);
                saw_prompt = true;
            } else if (std.mem.eql(u8, arg, "--ctx")) {
                const raw = try nextValue(args, &index);
                opts.context_length = std.fmt.parseUnsigned(u32, raw, 10) catch {
                    return error.InvalidContextLength;
                };
            } else if (std.mem.eql(u8, arg, "--decode")) {
                opts.decode_mode = try DecodeMode.parse(try nextValue(args, &index));
            } else {
                return error.InvalidArgument;
            }

            index += 1;
        }

        if (!saw_prompt or opts.prompt.len == 0) return error.MissingPrompt;
        if (opts.context_length == 0) return error.InvalidContextLength;

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
    config: model.ModelConfig,
    tok: tokenizer.Tokenizer,
    sample: sampler.Sampler,

    pub fn init(options: RunOptions) Engine {
        var cfg = model.ModelConfig.default();
        cfg.name = options.model_name;
        cfg.context_length = options.context_length;

        return .{
            .config = cfg,
            .tok = .{},
            .sample = .{},
        };
    }

    pub fn generate(self: Engine, writer: anytype, options: RunOptions) !void {
        try writer.print(
            "inference-engine scaffold\nmodel: {s}\nfamily: {s}\nquantization: {s}\ncontext: {d}\ndecode: {s}\ntokenizer: {s}\nprompt-bytes: {d}\n\n",
            .{
                self.config.name,
                self.config.family,
                self.config.quantization,
                self.config.context_length,
                options.decode_mode.label(),
                self.tok.name,
                self.tok.countPromptBytes(options.prompt),
            },
        );
        try self.sample.deterministicPlaceholder(writer, options.prompt);
        try writer.writeAll("\n");
    }
};

test "CLI argument parsing accepts scaffold flags" {
    const args = &[_][]const u8{
        "--model",
        "gemma-4-12b-q4",
        "--ctx",
        "8192",
        "--decode",
        "ar",
        "--prompt",
        "hello",
    };

    const opts = try RunOptions.parse(args);

    try std.testing.expectEqualStrings("gemma-4-12b-q4", opts.model_name);
    try std.testing.expectEqualStrings("hello", opts.prompt);
    try std.testing.expectEqual(@as(u32, 8192), opts.context_length);
    try std.testing.expectEqual(DecodeMode.ar, opts.decode_mode);
}

test "CLI argument parsing requires prompt" {
    const args = &[_][]const u8{"--model", "gemma-4-12b-q4"};
    try std.testing.expectError(error.MissingPrompt, RunOptions.parse(args));
}

test "engine generates deterministic scaffold output" {
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
