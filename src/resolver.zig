const std = @import("std");

const benchmark = @import("benchmark.zig");

pub const ResolveError = error{
    MissingModelPath,
    ModelNotFound,
    InvalidModelPath,
    AmbiguousModelDirectory,
    ProjectorModelNotSupported,
};

pub const Env = struct {
    model_cache: ?[]const u8 = null,
    home: ?[]const u8 = null,
};

pub const ResolvedModel = struct {
    model_path: []const u8,
    tokenizer_path: ?[]const u8,
    source: Source,

    pub const Source = enum {
        exact,
        directory,
        cache_alias,
    };

    pub fn deinit(self: ResolvedModel, allocator: std.mem.Allocator) void {
        allocator.free(self.model_path);
        if (self.tokenizer_path) |path| allocator.free(path);
    }
};

pub fn defaultCacheRoot(allocator: std.mem.Allocator, env: Env) ![]const u8 {
    if (env.model_cache) |cache| return allocator.dupe(u8, cache);
    if (env.home) |home| return std.Io.Dir.path.join(allocator, &.{ home, ".cache", "inference-engine", "models" });
    return error.MissingModelPath;
}

pub fn resolve(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_arg: ?[]const u8,
    tokenizer_arg: ?[]const u8,
    env: Env,
) !ResolvedModel {
    const raw_model = model_arg orelse return error.MissingModelPath;
    const tokenizer_path = if (tokenizer_arg) |value| try allocator.dupe(u8, value) else null;
    errdefer if (tokenizer_path) |path| allocator.free(path);

    if (looksLikePath(raw_model)) {
        return try resolvePath(allocator, io, raw_model, tokenizer_path, .exact);
    }

    const cache = try defaultCacheRoot(allocator, env);
    defer allocator.free(cache);
    const alias_dir = try std.Io.Dir.path.join(allocator, &.{ cache, raw_model });
    defer allocator.free(alias_dir);

    if (resolvePath(allocator, io, alias_dir, tokenizer_path, .cache_alias)) |resolved| {
        return resolved;
    } else |err| switch (err) {
        error.ModelNotFound => {},
        else => return err,
    }

    const alias_file = try std.fmt.allocPrint(allocator, "{s}.gguf", .{alias_dir});
    defer allocator.free(alias_file);
    if (fileExists(io, alias_file)) {
        if (isProjectorGguf(alias_file)) return error.ProjectorModelNotSupported;
        return .{
            .model_path = try allocator.dupe(u8, alias_file),
            .tokenizer_path = tokenizer_path,
            .source = .cache_alias,
        };
    }

    return error.ModelNotFound;
}

fn resolvePath(
    allocator: std.mem.Allocator,
    io: std.Io,
    raw_path: []const u8,
    tokenizer_path: ?[]const u8,
    source: ResolvedModel.Source,
) !ResolvedModel {
    if (fileExists(io, raw_path)) {
        if (isProjectorGguf(raw_path)) return error.ProjectorModelNotSupported;
        return .{
            .model_path = try allocator.dupe(u8, raw_path),
            .tokenizer_path = tokenizer_path,
            .source = source,
        };
    }

    if (findGgufInDirectory(allocator, io, raw_path)) |model_path| {
        return .{
            .model_path = model_path,
            .tokenizer_path = tokenizer_path,
            .source = .directory,
        };
    } else |err| switch (err) {
        error.ModelNotFound, error.FileNotFound, error.NotDir => {},
        else => return err,
    }

    return error.ModelNotFound;
}

fn looksLikePath(value: []const u8) bool {
    return std.mem.indexOfScalar(u8, value, '/') != null or
        std.mem.indexOfScalar(u8, value, '\\') != null or
        std.mem.endsWith(u8, value, ".gguf");
}

fn fileExists(io: std.Io, path: []const u8) bool {
    var file = if (std.Io.Dir.path.isAbsolute(path))
        std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false
    else
        std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

fn findGgufInDirectory(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) ![]const u8 {
    var dir = if (std.Io.Dir.path.isAbsolute(dir_path))
        try std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true })
    else
        try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);

    return findTextGgufInOpenDirectory(allocator, io, dir_path, dir);
}

fn findTextGgufInOpenDirectory(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir_path: []const u8,
    dir: std.Io.Dir,
) ![]const u8 {
    var found: ?[]const u8 = null;
    errdefer if (found) |path| allocator.free(path);
    var saw_projector = false;

    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".gguf")) continue;
        if (std.mem.eql(u8, entry.name, benchmark.target_artifact.filename)) {
            if (found) |path| allocator.free(path);
            return try std.Io.Dir.path.join(allocator, &.{ dir_path, entry.name });
        }
        if (isProjectorGguf(entry.name)) {
            saw_projector = true;
            continue;
        }

        if (found != null) return error.AmbiguousModelDirectory;
        found = try std.Io.Dir.path.join(allocator, &.{ dir_path, entry.name });
    }
    if (found) |path| return path;
    return if (saw_projector) error.ProjectorModelNotSupported else error.ModelNotFound;
}

fn isProjectorGguf(path: []const u8) bool {
    const name = basename(path);
    return std.mem.startsWith(u8, name, "mmproj-") and std.mem.endsWith(u8, name, ".gguf");
}

fn basename(path: []const u8) []const u8 {
    var index = path.len;
    while (index > 0) {
        index -= 1;
        if (path[index] == '/' or path[index] == '\\') return path[index + 1 ..];
    }
    return path;
}

test "default cache root uses env override" {
    const root = try defaultCacheRoot(std.testing.allocator, .{ .model_cache = "/tmp/models", .home = "/home/me" });
    defer std.testing.allocator.free(root);
    try std.testing.expectEqualStrings("/tmp/models", root);
}

test "default cache root falls back to home" {
    const root = try defaultCacheRoot(std.testing.allocator, .{ .home = "/home/me" });
    defer std.testing.allocator.free(root);
    try std.testing.expectEqualStrings("/home/me/.cache/inference-engine/models", root);
}

test "directory resolver prefers Gemma 4 text GGUF over projector" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    var text_file = try tmp.dir.createFile(io, benchmark.target_artifact.filename, .{ .read = true });
    text_file.close(io);
    var projector_file = try tmp.dir.createFile(io, "mmproj-gemma-4-12b-it-qat-q4_0.gguf", .{ .read = true });
    projector_file.close(io);

    const path = try findTextGgufInOpenDirectory(std.testing.allocator, io, "fixture", tmp.dir);
    defer std.testing.allocator.free(path);
    try std.testing.expect(std.mem.endsWith(u8, path, benchmark.target_artifact.filename));
}

test "directory resolver rejects ambiguous text GGUF files" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    var first = try tmp.dir.createFile(io, "first.gguf", .{ .read = true });
    first.close(io);
    var second = try tmp.dir.createFile(io, "second.gguf", .{ .read = true });
    second.close(io);

    try std.testing.expectError(error.AmbiguousModelDirectory, findTextGgufInOpenDirectory(std.testing.allocator, io, "fixture", tmp.dir));
}

test "projector path is rejected for text-only inference" {
    try std.testing.expect(isProjectorGguf("/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf"));
    try std.testing.expect(!isProjectorGguf("/models/gemma-4-12b-it-qat-q4_0.gguf"));
}
