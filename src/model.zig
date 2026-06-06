const std = @import("std");

pub const ModelConfig = struct {
    name: []const u8,
    family: []const u8,
    quantization: []const u8,
    context_length: u32,

    pub fn default() ModelConfig {
        return .{
            .name = "gemma-4-12b-q4",
            .family = "gemma",
            .quantization = "q4_0",
            .context_length = 8192,
        };
    }
};

test "model config defaults target Gemma scaffold" {
    const cfg = ModelConfig.default();

    try std.testing.expectEqualStrings("gemma-4-12b-q4", cfg.name);
    try std.testing.expectEqualStrings("gemma", cfg.family);
    try std.testing.expectEqualStrings("q4_0", cfg.quantization);
    try std.testing.expectEqual(@as(u32, 8192), cfg.context_length);
}
