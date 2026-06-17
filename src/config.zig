//! User configuration loading for reusable CLI defaults.

const std = @import("std");

const backend = @import("backend.zig");
const engine = @import("engine.zig");

/// Errors returned while loading or parsing the user config file.
pub const ConfigError = error{
    ConfigTooLarge,
    InvalidConfigLine,
    InvalidConfigKey,
    InvalidConfigValue,
    InvalidConfigString,
    InvalidContextLength,
    InvalidMaxNewTokens,
    UnknownDecodeMode,
    UnknownBackend,
    InvalidArtifactRevision,
};

/// Environment inputs used to locate the default config file.
pub const Env = struct {
    home: ?[]const u8 = null,
};

const max_config_bytes = 64 * 1024;

/// Returns the default user config path under HOME.
pub fn defaultPath(allocator: std.mem.Allocator, env: Env) !?[]const u8 {
    const home = env.home orelse return null;
    return try std.Io.Dir.path.join(allocator, &.{ home, ".config", "inference-engine", "config.toml" });
}

/// Loads reusable run defaults from ~/.config/inference-engine/config.toml.
pub fn loadRunDefaults(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: Env,
) !engine.RunOptions {
    const path = (try defaultPath(allocator, env)) orelse return .{};
    defer allocator.free(path);

    var file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return .{},
        else => return err,
    };
    defer file.close(io);

    const stat = try file.stat(io);
    if (stat.size > max_config_bytes) return error.ConfigTooLarge;
    const size = std.math.cast(usize, stat.size) orelse return error.ConfigTooLarge;
    const bytes = try allocator.alloc(u8, size);
    errdefer allocator.free(bytes);
    defer allocator.free(bytes);
    const read = try file.readPositionalAll(io, bytes, 0);
    if (read != bytes.len) return error.Unexpected;

    return parseRunDefaults(allocator, bytes);
}

/// Parses reusable run defaults from a flat TOML config slice.
pub fn parseRunDefaults(allocator: std.mem.Allocator, bytes: []const u8) !engine.RunOptions {
    var opts = engine.RunOptions{};

    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, stripComment(raw_line), " \t\r");
        if (line.len == 0) continue;

        const eq_index = std.mem.indexOfScalar(u8, line, '=') orelse return error.InvalidConfigLine;
        const key = std.mem.trim(u8, line[0..eq_index], " \t");
        const value = std.mem.trim(u8, line[eq_index + 1 ..], " \t");
        if (key.len == 0 or value.len == 0) return error.InvalidConfigLine;

        if (std.mem.eql(u8, key, "model")) {
            opts.model_path = try parseString(allocator, value);
        } else if (std.mem.eql(u8, key, "tokenizer")) {
            opts.tokenizer_path = try parseString(allocator, value);
        } else if (std.mem.eql(u8, key, "prompt")) {
            opts.prompt = try parseString(allocator, value);
        } else if (std.mem.eql(u8, key, "ctx")) {
            opts.context_length = parseU32(value) catch return error.InvalidContextLength;
        } else if (std.mem.eql(u8, key, "max_new_tokens")) {
            opts.max_new_tokens = parseU32(value) catch return error.InvalidMaxNewTokens;
        } else if (std.mem.eql(u8, key, "artifact_revision")) {
            const decoded = try parseString(allocator, value);
            opts.artifact_revision = engine.parseArtifactRevision(decoded) catch return error.InvalidArtifactRevision;
        } else if (std.mem.eql(u8, key, "decode")) {
            const decoded = try parseString(allocator, value);
            defer allocator.free(decoded);
            opts.decode_mode = engine.DecodeMode.parse(decoded) catch return error.UnknownDecodeMode;
        } else if (std.mem.eql(u8, key, "backend")) {
            const decoded = try parseString(allocator, value);
            defer allocator.free(decoded);
            opts.backend = backend.Backend.parse(decoded) catch return error.UnknownBackend;
        } else {
            return error.InvalidConfigKey;
        }
    }

    if (opts.max_new_tokens == 0) return error.InvalidMaxNewTokens;
    return opts;
}

fn stripComment(line: []const u8) []const u8 {
    var in_string = false;
    var escaped = false;
    for (line, 0..) |byte, index| {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (in_string and byte == '\\') {
            escaped = true;
            continue;
        }
        if (byte == '"') {
            in_string = !in_string;
            continue;
        }
        if (!in_string and byte == '#') return line[0..index];
    }
    return line;
}

fn parseString(allocator: std.mem.Allocator, value: []const u8) ![]const u8 {
    if (value.len < 2 or value[0] != '"' or value[value.len - 1] != '"') return error.InvalidConfigString;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var index: usize = 1;
    while (index + 1 < value.len) {
        const byte = value[index];
        if (byte == '\\') {
            index += 1;
            if (index + 1 >= value.len) return error.InvalidConfigString;
            try out.append(allocator, switch (value[index]) {
                '\\' => '\\',
                '"' => '"',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                else => return error.InvalidConfigString,
            });
        } else {
            try out.append(allocator, byte);
        }
        index += 1;
    }

    return out.toOwnedSlice(allocator);
}

fn parseU32(value: []const u8) !u32 {
    if (value.len == 0 or value[0] == '-') return error.InvalidConfigValue;
    return std.fmt.parseUnsigned(u32, value, 10) catch error.InvalidConfigValue;
}

test "config parser applies reusable defaults" {
    const opts = try parseRunDefaults(std.testing.allocator,
        \\model = "data/model.gguf"
        \\tokenizer = "tokenizer.json"
        \\prompt = "hello"
        \\ctx = 2048
        \\max_new_tokens = 8
        \\artifact_revision = "abc123"
        \\decode = "ar"
        \\backend = "cuda"
        \\
    );
    defer {
        std.testing.allocator.free(opts.model_path.?);
        std.testing.allocator.free(opts.tokenizer_path.?);
        std.testing.allocator.free(opts.prompt);
        std.testing.allocator.free(opts.artifact_revision.?);
    }

    try std.testing.expectEqualStrings("data/model.gguf", opts.model_path.?);
    try std.testing.expectEqualStrings("tokenizer.json", opts.tokenizer_path.?);
    try std.testing.expectEqualStrings("hello", opts.prompt);
    try std.testing.expectEqual(@as(u32, 2048), opts.context_length);
    try std.testing.expectEqual(@as(u32, 8), opts.max_new_tokens);
    try std.testing.expectEqualStrings("abc123", opts.artifact_revision.?);
    try std.testing.expectEqual(engine.DecodeMode.ar, opts.decode_mode);
    try std.testing.expectEqual(backend.Backend.cuda, opts.backend);
}

test "config parser rejects action and typo keys" {
    try std.testing.expectError(error.InvalidConfigKey, parseRunDefaults(std.testing.allocator, "validate_model = true\n"));
    try std.testing.expectError(error.InvalidConfigKey, parseRunDefaults(std.testing.allocator, "modle = \"data/model.gguf\"\n"));
}

test "config strings allow comments outside quoted values" {
    const opts = try parseRunDefaults(std.testing.allocator, "prompt = \"hello # not comment\" # comment\n");
    defer std.testing.allocator.free(opts.prompt);
    try std.testing.expectEqualStrings("hello # not comment", opts.prompt);
}
