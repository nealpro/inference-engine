//! Minimal sampling scaffold used before real model execution is wired.

const std = @import("std");

/// Generation settings shared by the placeholder sampler.
pub const SamplingConfig = struct {
    temperature: f32 = 0.0,
    max_new_tokens: u32 = 64,
};

/// Deterministic placeholder sampler for the current scaffold.
pub const Sampler = struct {
    config: SamplingConfig = .{},

    /// Writes a stable placeholder response for tests and CLI plumbing.
    pub fn deterministicPlaceholder(
        _: Sampler,
        writer: anytype,
        prompt: []const u8,
    ) !void {
        try writer.print(
            "placeholder response for prompt ({d} bytes): {s}",
            .{ prompt.len, prompt },
        );
    }
};

test "deterministic placeholder generation" {
    var out: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try (Sampler{}).deterministicPlaceholder(&writer, "hello");

    try std.testing.expectEqualStrings(
        "placeholder response for prompt (5 bytes): hello",
        out[0..writer.end],
    );
}
