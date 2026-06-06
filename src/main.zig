const std = @import("std");
const Io = std.Io;

const inference = @import("inference_engine");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    const options = inference.RunOptions.parse(args[1..]) catch |err| {
        try printUsage(init.io, err);
        return err;
    };

    const runner = inference.Engine.init(options);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const stdout = &stdout_file_writer.interface;

    try runner.generate(stdout, options);
    try stdout.flush();
}

fn printUsage(io: Io, err: anyerror) !void {
    var stderr_buffer: [1024]u8 = undefined;
    var stderr_file_writer: Io.File.Writer = .init(.stderr(), io, &stderr_buffer);
    const stderr = &stderr_file_writer.interface;

    try stderr.print("error: {s}\n\n", .{@errorName(err)});
    try stderr.writeAll(
        \\usage:
        \\  inference_engine --prompt <text> [--model <name-or-path>] [--ctx <tokens>] [--decode ar|mtp|ssd-sim]
        \\
        \\example:
        \\  inference_engine --model gemma-4-12b-q4 --ctx 8192 --decode ar --prompt "hello"
        \\
    );
    try stderr.flush();
}

test "executable imports library scaffold" {
    try std.testing.expectEqual(inference.DecodeMode.ar, try inference.DecodeMode.parse("ar"));
}
