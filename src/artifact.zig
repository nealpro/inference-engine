//! Artifact identity helpers for benchmarkable model runs.

const std = @import("std");

/// Number of lowercase hexadecimal bytes in a SHA-256 digest.
pub const sha256_hex_len = std.crypto.hash.sha2.Sha256.digest_length * 2;
/// Fixed-size lowercase SHA-256 digest string.
pub const Sha256Hex = [sha256_hex_len]u8;

/// Computes the lowercase SHA-256 digest for a file path.
pub fn sha256FileHex(io: std.Io, path: []const u8) !Sha256Hex {
    var file = if (std.Io.Dir.path.isAbsolute(path))
        try std.Io.Dir.openFileAbsolute(io, path, .{})
    else
        try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;
    while (offset < stat.size) {
        const read = try file.readPositionalAll(io, &buffer, offset);
        if (read == 0) return error.Unexpected;
        hasher.update(buffer[0..read]);
        offset += read;
    }

    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

test "sha256 file identity uses lowercase hex" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile(io, "artifact.bin", .{ .read = true });
    try file.writePositionalAll(io, "abc", 0);
    file.close(io);

    const path = try std.Io.Dir.path.join(
        std.testing.allocator,
        &.{ ".zig-cache", "tmp", &tmp.sub_path, "artifact.bin" },
    );
    defer std.testing.allocator.free(path);

    const digest = try sha256FileHex(io, path);
    try std.testing.expectEqualStrings(
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        &digest,
    );
}
