const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const cuda_enabled = b.option(bool, "cuda", "Build CUDA kernel objects with nvcc") orelse false;
    const build_options = b.addOptions();
    build_options.addOption(bool, "cuda_enabled", cuda_enabled);

    const lib_mod = b.addModule("inference_engine", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
        },
    });

    const exe = b.addExecutable(.{
        .name = "inference_engine",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "inference_engine", .module = lib_mod },
                .{ .name = "build_options", .module = build_options.createModule() },
            },
        }),
    });
    var cuda_obj: ?std.Build.LazyPath = null;
    if (cuda_enabled) {
        const nvcc = b.findProgram(&.{"nvcc"}, &.{
            "/usr/local/cuda/bin",
            "/usr/local/cuda-13/bin",
            "/usr/local/cuda-13.3/bin",
        }) catch {
            std.log.err("CUDA build requested with -Dcuda=true, but nvcc was not found on PATH", .{});
            std.process.exit(1);
        };
        const cuda_compile = b.addSystemCommand(&.{
            nvcc,
            "-c",
        });
        cuda_compile.addFileArg(b.path("src/cuda/kernels.cu"));
        cuda_compile.addArgs(&.{
            "-arch=sm_89",
            "-allow-unsupported-compiler",
            "-o",
        });
        cuda_obj = cuda_compile.addOutputFileArg("inference_engine_cuda_kernels.o");
        addCudaLinkage(lib_mod, cuda_obj.?);
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the inference engine scaffold");
    run_step.dependOn(&run_cmd.step);

    const docs_obj = b.addObject(.{
        .name = "inference_engine_docs",
        .root_module = lib_mod,
    });
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs_obj.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Generate API docs into zig-out/docs");
    docs_step.dependOn(&install_docs.step);

    const lib_tests = b.addTest(.{
        .root_module = lib_mod,
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}

fn addCudaLinkage(module: *std.Build.Module, cuda_obj: std.Build.LazyPath) void {
    module.addObjectFile(cuda_obj);
    module.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/targets/x86_64-linux/lib" });
    module.addRPath(.{ .cwd_relative = "/usr/local/cuda/targets/x86_64-linux/lib" });
    module.linkSystemLibrary("c", .{});
    module.linkSystemLibrary("stdc++", .{});
    module.linkSystemLibrary("cudart", .{});
}
