//! Runtime backend selection and CUDA availability probing.

const std = @import("std");
const build_options = @import("build_options");

/// Errors returned while selecting or preparing an inference backend.
pub const BackendError = error{
    UnknownBackend,
    CudaUnavailable,
    CudaDeviceUnavailable,
    CudaKernelUnavailable,
    CudaProbeFailed,
    GpuInferenceNotImplemented,
};

/// User-selectable execution backend.
pub const Backend = enum {
    auto,
    cuda,

    /// Parses the CLI spelling for a backend.
    pub fn parse(value: []const u8) BackendError!Backend {
        if (std.mem.eql(u8, value, "auto")) return .auto;
        if (std.mem.eql(u8, value, "cuda")) return .cuda;
        return error.UnknownBackend;
    }

    /// Returns the CLI label for this backend.
    pub fn label(self: Backend) []const u8 {
        return switch (self) {
            .auto => "auto",
            .cuda => "cuda",
        };
    }
};

/// Resolved backend ready for inference setup.
pub const SelectedBackend = enum {
    cuda,

    /// Returns the CLI label for this selected backend.
    pub fn label(self: SelectedBackend) []const u8 {
        return switch (self) {
            .cuda => "cuda",
        };
    }
};

/// Selects the runtime backend. CPU transformer execution is intentionally not a fallback.
pub fn select(requested: Backend) BackendError!SelectedBackend {
    return switch (requested) {
        .auto, .cuda => blk: {
            try cuda.ensureAvailable();
            break :blk .cuda;
        },
    };
}

/// Minimal CUDA Driver API probing without requiring CUDA headers at Zig compile time.
pub const cuda = struct {
    const CudaSuccess = 0;
    const CudaProbeSuccess = 0;

    const CuInit = *const fn (flags: c_uint) callconv(.c) c_int;
    const CuDeviceGetCount = *const fn (count: *c_int) callconv(.c) c_int;

    extern fn inference_engine_cuda_probe() c_int;

    /// Verifies that the CUDA driver library can be opened and at least one device is visible.
    pub fn ensureAvailable() BackendError!void {
        var lib = std.DynLib.open("libcuda.so.1") catch
            std.DynLib.open("libcuda.so") catch
            return error.CudaUnavailable;
        defer lib.close();

        const cu_init = lib.lookup(CuInit, "cuInit") orelse return error.CudaUnavailable;
        const cu_device_get_count = lib.lookup(CuDeviceGetCount, "cuDeviceGetCount") orelse return error.CudaUnavailable;

        if (cu_init(0) != CudaSuccess) return error.CudaUnavailable;

        var count: c_int = 0;
        if (cu_device_get_count(&count) != CudaSuccess) return error.CudaDeviceUnavailable;
        if (count <= 0) return error.CudaDeviceUnavailable;
    }

    /// Verifies that the CUDA runtime can launch a kernel compiled into this binary.
    pub fn ensureKernelLaunchWorks() BackendError!void {
        if (!build_options.cuda_enabled) return error.CudaKernelUnavailable;
        if (inference_engine_cuda_probe() != CudaProbeSuccess) return error.CudaProbeFailed;
    }
};

test "backend parser accepts auto and cuda" {
    try std.testing.expectEqual(Backend.auto, try Backend.parse("auto"));
    try std.testing.expectEqual(Backend.cuda, try Backend.parse("cuda"));
    try std.testing.expectError(error.UnknownBackend, Backend.parse("cpu"));
    try std.testing.expectEqualStrings("cuda", Backend.cuda.label());
}

test "selected backend labels are stable" {
    try std.testing.expectEqualStrings("cuda", SelectedBackend.cuda.label());
}

test "cuda probe kernel launches when CUDA build is enabled" {
    if (!build_options.cuda_enabled) return;
    try cuda.ensureAvailable();
    try cuda.ensureKernelLaunchWorks();
}
