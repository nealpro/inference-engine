const std = @import("std");

pub const Tokenizer = struct {
    name: []const u8 = "placeholder-tokenizer",

    pub fn countPromptBytes(_: Tokenizer, prompt: []const u8) usize {
        return prompt.len;
    }
};

test "placeholder tokenizer counts prompt bytes" {
    const tok = Tokenizer{};
    try std.testing.expectEqual(@as(usize, 5), tok.countPromptBytes("hello"));
}
