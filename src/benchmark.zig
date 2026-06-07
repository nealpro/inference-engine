const std = @import("std");

pub const contract_version = "benchmark-contract-v1";
pub const gemma4_integration_env = "INFERENCE_ENGINE_GEMMA4_DIR";

pub const TargetArtifact = struct {
    hf_repo: []const u8,
    filename: []const u8,
    tokenizer_source: []const u8,
    revision_policy: []const u8,
    artifact_hash_policy: []const u8,
    text_only_policy: []const u8,
};

pub const target_artifact = TargetArtifact{
    .hf_repo = "google/gemma-4-12B-it-qat-q4_0-gguf",
    .filename = "gemma-4-12b-it-qat-q4_0.gguf",
    .tokenizer_source = "GGUF tokenizer metadata or explicit Gemma-compatible tokenizer path",
    .revision_policy = "benchmark runs must record an immutable Hugging Face snapshot revision; do not record main",
    .artifact_hash_policy = "benchmark runs must record a SHA256, Xet hash, or equivalent immutable file digest",
    .text_only_policy = "text-only inference target; reject mmproj projector GGUF files",
};

pub const ModelLineage = struct {
    model_source_policy: []const u8,
    engine_artifact_policy: []const u8,
    reference_artifact_policy: []const u8,
    revision_policy: []const u8,
    artifact_format_policy: []const u8,
    tokenizer_policy: []const u8,
    context_length_policy: []const u8,
    chat_template_policy: []const u8,
};

pub const model_lineage = ModelLineage{
    .model_source_policy = "official Gemma 4 12B instruction-tuned QAT Q4_0 GGUF target",
    .engine_artifact_policy = "local GGUF path supplied by --model; default target is google/gemma-4-12B-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf",
    .reference_artifact_policy = "MLX reference source and conversion output must be recorded by the benchmark run, including snapshot hash",
    .revision_policy = "record immutable snapshot hashes for both engine and reference artifacts; never benchmark a floating alias",
    .artifact_format_policy = "engine artifact is GGUF Q4_0 QAT; reference artifact is MLX native conversion or explicitly recorded MLX-compatible source",
    .tokenizer_policy = "load tokenizer from the GGUF artifact or explicit Gemma-compatible --tokenizer path",
    .context_length_policy = "record requested context and discovered model context in the run output",
    .chat_template_policy = "single user message, no system message, reference chat template, add generation prompt",
};

pub const OutputPolicy = struct {
    max_new_tokens: u32,
    min_required_new_tokens: u32,
    stop_on_eos: bool,
    early_eos_is_failure: bool,
};

pub const output_policy = OutputPolicy{
    .max_new_tokens = 128,
    .min_required_new_tokens = 16,
    .stop_on_eos = true,
    .early_eos_is_failure = true,
};

pub const GenerationSettings = struct {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    seed: u64,
    sampling_policy: []const u8,
};

pub const generation_settings = GenerationSettings{
    .temperature = 0.0,
    .top_p = 1.0,
    .top_k = 1,
    .repeat_penalty = 1.0,
    .seed = 0,
    .sampling_policy = "greedy argmax; no random sampling",
};

pub const PromptCase = struct {
    id: []const u8,
    category: []const u8,
    prompt: []const u8,
};

pub const prompt_suite = [_]PromptCase{
    .{
        .id = "short_latency",
        .category = "short answer",
        .prompt = "In one sentence, explain why low latency matters for local inference.",
    },
    .{
        .id = "zig_api",
        .category = "coding",
        .prompt = "In Zig, sketch a function signature for multiplying two f32 matrices and list the shape checks it must perform.",
    },
    .{
        .id = "latency_math",
        .category = "reasoning",
        .prompt = "A model spends 180 ms before its first token, then emits 8 tokens over the next 96 ms. Compute TTFT, average inter-token latency after the first token, and generation tokens per second.",
    },
    .{
        .id = "summarize_contract",
        .category = "summarization",
        .prompt =
        \\Summarize this benchmark rule in three concise bullets: compare engines only when they use the same model lineage, prompt suite, output length policy, deterministic decoding settings, and hardware environment capture.
        ,
    },
    .{
        .id = "json_shape",
        .category = "format following",
        .prompt = "Return JSON with exactly these keys: model, primary_metric, secondary_metrics. Use short string values or arrays of strings.",
    },
    .{
        .id = "long_context_probe",
        .category = "longer prompt",
        .prompt =
        \\You are evaluating an inference engine for interactive local use. The benchmark should value time to first token, steady inter-token latency, and total wall-clock latency over peak throughput. Explain the benchmark in a short memo for an engineer who will reproduce the run later.
        ,
    },
};

pub const MetricPriority = enum {
    primary,
    secondary,
};

pub const Metric = struct {
    name: []const u8,
    unit: []const u8,
    priority: MetricPriority,
    description: []const u8,
};

pub const metrics = [_]Metric{
    .{
        .name = "time_to_first_token",
        .unit = "milliseconds",
        .priority = .primary,
        .description = "Wall-clock time from generation start to first emitted token.",
    },
    .{
        .name = "inter_token_latency",
        .unit = "milliseconds per token",
        .priority = .primary,
        .description = "Mean, p50, and p95 latency between emitted tokens after the first token.",
    },
    .{
        .name = "end_to_end_latency",
        .unit = "milliseconds",
        .priority = .primary,
        .description = "Wall-clock time from generation start to completion.",
    },
    .{
        .name = "tokens_per_second",
        .unit = "tokens per second",
        .priority = .secondary,
        .description = "Generated tokens divided by decode wall time.",
    },
    .{
        .name = "peak_memory",
        .unit = "bytes",
        .priority = .secondary,
        .description = "Peak resident memory observed during the benchmark run.",
    },
    .{
        .name = "generated_token_correctness",
        .unit = "pass/fail",
        .priority = .secondary,
        .description = "Deterministic token IDs match the MLX reference for the prompt case.",
    },
};

pub const MlxBaseline = struct {
    model_dir_env: []const u8,
    prompt_file_env: []const u8,
    command: []const u8,
    timing_status: []const u8,
    conversion_policy: []const u8,
    environment_commands: []const []const u8,
};

pub const mlx_baseline = MlxBaseline{
    .model_dir_env = "MLX_MODEL_DIR",
    .prompt_file_env = "PROMPT_FILE",
    .command = "python -m mlx_lm.generate --model \"$MLX_MODEL_DIR\" --prompt \"$(cat \"$PROMPT_FILE\")\" --max-tokens 128 --temp 0 --top-p 1.0 --seed 0",
    .timing_status = "baseline command is fixed, but timing comparison is deferred until this engine emits real Gemma 4 12B tokens",
    .conversion_policy = "record the MLX source model, conversion command, output directory, quantization settings, and immutable snapshot revision",
    .environment_commands = &[_][]const u8{
        "sw_vers",
        "sysctl -n machdep.cpu.brand_string",
        "system_profiler SPHardwareDataType",
        "python -V",
        "python -m pip show mlx mlx-lm",
        "git rev-parse HEAD",
        "zig version",
    },
};

pub fn promptById(id: []const u8) ?PromptCase {
    for (prompt_suite) |case| {
        if (std.mem.eql(u8, case.id, id)) return case;
    }
    return null;
}

pub fn writeManifestJson(writer: anytype) !void {
    try writer.writeAll("{\n");
    try writeJsonField(writer, 1, "version", contract_version, true);
    try writeJsonField(writer, 1, "target_hf_repo", target_artifact.hf_repo, true);
    try writeJsonField(writer, 1, "target_filename", target_artifact.filename, true);
    try writeJsonField(writer, 1, "target_tokenizer_source", target_artifact.tokenizer_source, true);
    try writeJsonField(writer, 1, "target_revision_policy", target_artifact.revision_policy, true);
    try writeJsonField(writer, 1, "target_artifact_hash_policy", target_artifact.artifact_hash_policy, true);
    try writeJsonField(writer, 1, "target_text_only_policy", target_artifact.text_only_policy, true);
    try writer.writeAll("  \"generation\": {\n");
    try writeJsonNumberField(writer, 2, "max_new_tokens", output_policy.max_new_tokens, true);
    try writeJsonNumberField(writer, 2, "min_required_new_tokens", output_policy.min_required_new_tokens, true);
    try writeJsonBoolField(writer, 2, "stop_on_eos", output_policy.stop_on_eos, true);
    try writeJsonBoolField(writer, 2, "early_eos_is_failure", output_policy.early_eos_is_failure, true);
    try writeJsonFloatField(writer, 2, "temperature", generation_settings.temperature, true);
    try writeJsonFloatField(writer, 2, "top_p", generation_settings.top_p, true);
    try writeJsonNumberField(writer, 2, "top_k", generation_settings.top_k, true);
    try writeJsonFloatField(writer, 2, "repeat_penalty", generation_settings.repeat_penalty, true);
    try writeJsonNumberField(writer, 2, "seed", generation_settings.seed, true);
    try writeJsonField(writer, 2, "sampling_policy", generation_settings.sampling_policy, false);
    try writer.writeAll("  },\n");

    try writer.writeAll("  \"prompts\": [\n");
    for (prompt_suite, 0..) |case, index| {
        try writer.writeAll("    {");
        try writeJsonInlineField(writer, "id", case.id, true);
        try writeJsonInlineField(writer, "category", case.category, true);
        try writeJsonInlineField(writer, "prompt", case.prompt, false);
        try writer.writeAll(if (index + 1 == prompt_suite.len) "}\n" else "},\n");
    }
    try writer.writeAll("  ],\n");

    try writer.writeAll("  \"metrics\": [\n");
    for (metrics, 0..) |metric, index| {
        try writer.writeAll("    {");
        try writeJsonInlineField(writer, "name", metric.name, true);
        try writeJsonInlineField(writer, "unit", metric.unit, true);
        try writeJsonInlineField(writer, "priority", @tagName(metric.priority), true);
        try writeJsonInlineField(writer, "description", metric.description, false);
        try writer.writeAll(if (index + 1 == metrics.len) "}\n" else "},\n");
    }
    try writer.writeAll("  ],\n");

    try writer.writeAll("  \"mlx_baseline\": {\n");
    try writeJsonField(writer, 2, "model_dir_env", mlx_baseline.model_dir_env, true);
    try writeJsonField(writer, 2, "prompt_file_env", mlx_baseline.prompt_file_env, true);
    try writeJsonField(writer, 2, "command", mlx_baseline.command, true);
    try writeJsonField(writer, 2, "timing_status", mlx_baseline.timing_status, true);
    try writeJsonField(writer, 2, "conversion_policy", mlx_baseline.conversion_policy, true);
    try writer.writeAll("    \"environment_commands\": [");
    for (mlx_baseline.environment_commands, 0..) |command, index| {
        if (index != 0) try writer.writeAll(", ");
        try writeJsonString(writer, command);
    }
    try writer.writeAll("]\n");
    try writer.writeAll("  }\n");
    try writer.writeAll("}\n");
}

pub fn writeContract(writer: anytype) !void {
    try writer.print(
        \\benchmark contract
        \\version: {s}
        \\target repo: {s}
        \\target file: {s}
        \\target tokenizer: {s}
        \\target revision: {s}
        \\target artifact hash: {s}
        \\target modality: {s}
        \\model source: {s}
        \\engine artifact: {s}
        \\reference artifact: {s}
        \\revision: {s}
        \\format: {s}
        \\tokenizer: {s}
        \\context: {s}
        \\chat template: {s}
        \\
        \\generation:
        \\  max_new_tokens: {d}
        \\  min_required_new_tokens: {d}
        \\  stop_on_eos: {s}
        \\  early_eos_is_failure: {s}
        \\  temperature: {d:.1}
        \\  top_p: {d:.1}
        \\  top_k: {d}
        \\  repeat_penalty: {d:.1}
        \\  seed: {d}
        \\  sampling: {s}
        \\
    ,
        .{
            contract_version,
            target_artifact.hf_repo,
            target_artifact.filename,
            target_artifact.tokenizer_source,
            target_artifact.revision_policy,
            target_artifact.artifact_hash_policy,
            target_artifact.text_only_policy,
            model_lineage.model_source_policy,
            model_lineage.engine_artifact_policy,
            model_lineage.reference_artifact_policy,
            model_lineage.revision_policy,
            model_lineage.artifact_format_policy,
            model_lineage.tokenizer_policy,
            model_lineage.context_length_policy,
            model_lineage.chat_template_policy,
            output_policy.max_new_tokens,
            output_policy.min_required_new_tokens,
            if (output_policy.stop_on_eos) "true" else "false",
            if (output_policy.early_eos_is_failure) "true" else "false",
            generation_settings.temperature,
            generation_settings.top_p,
            generation_settings.top_k,
            generation_settings.repeat_penalty,
            generation_settings.seed,
            generation_settings.sampling_policy,
        },
    );

    try writer.writeAll("prompts:\n");
    for (prompt_suite, 1..) |case, index| {
        try writer.print("  {d}. {s} [{s}]\n     {s}\n", .{
            index,
            case.id,
            case.category,
            case.prompt,
        });
    }

    try writer.writeAll("\nmetrics:\n");
    for (metrics) |metric| {
        try writer.print("  - {s} ({s}, {s}): {s}\n", .{
            metric.name,
            @tagName(metric.priority),
            metric.unit,
            metric.description,
        });
    }

    try writer.print(
        \\
        \\mlx baseline:
        \\  {s}
        \\mlx timing status:
        \\  {s}
        \\mlx conversion:
        \\  {s}
        \\environment capture:
        \\
    ,
        .{
            mlx_baseline.command,
            mlx_baseline.timing_status,
            mlx_baseline.conversion_policy,
        },
    );
    for (mlx_baseline.environment_commands) |command| {
        try writer.print("  - {s}\n", .{command});
    }
}

fn writeJsonField(writer: anytype, indent: usize, key: []const u8, value: []const u8, comma: bool) !void {
    try writeIndent(writer, indent);
    try writeJsonString(writer, key);
    try writer.writeAll(": ");
    try writeJsonString(writer, value);
    try writer.writeAll(if (comma) ",\n" else "\n");
}

fn writeJsonNumberField(writer: anytype, indent: usize, key: []const u8, value: anytype, comma: bool) !void {
    try writeIndent(writer, indent);
    try writeJsonString(writer, key);
    try writer.print(": {d}", .{value});
    try writer.writeAll(if (comma) ",\n" else "\n");
}

fn writeJsonFloatField(writer: anytype, indent: usize, key: []const u8, value: f32, comma: bool) !void {
    try writeIndent(writer, indent);
    try writeJsonString(writer, key);
    try writer.print(": {d:.1}", .{value});
    try writer.writeAll(if (comma) ",\n" else "\n");
}

fn writeJsonBoolField(writer: anytype, indent: usize, key: []const u8, value: bool, comma: bool) !void {
    try writeIndent(writer, indent);
    try writeJsonString(writer, key);
    try writer.writeAll(if (value) ": true" else ": false");
    try writer.writeAll(if (comma) ",\n" else "\n");
}

fn writeJsonInlineField(writer: anytype, key: []const u8, value: []const u8, comma: bool) !void {
    try writeJsonString(writer, key);
    try writer.writeAll(": ");
    try writeJsonString(writer, value);
    try writer.writeAll(if (comma) ", " else "");
}

fn writeIndent(writer: anytype, indent: usize) !void {
    for (0..indent) |_| try writer.writeAll("  ");
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(byte),
        }
    }
    try writer.writeByte('"');
}

test "benchmark manifest pins Gemma 4 QAT GGUF target" {
    try std.testing.expectEqualStrings("benchmark-contract-v1", contract_version);
    try std.testing.expectEqualStrings("google/gemma-4-12B-it-qat-q4_0-gguf", target_artifact.hf_repo);
    try std.testing.expectEqualStrings("gemma-4-12b-it-qat-q4_0.gguf", target_artifact.filename);
    try std.testing.expect(std.mem.indexOf(u8, target_artifact.revision_policy, "immutable") != null);
    try std.testing.expect(std.mem.indexOf(u8, target_artifact.artifact_hash_policy, "hash") != null);
}

test "benchmark manifest json is stable and includes MLX deferral" {
    var out: [20000]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try writeManifestJson(&writer);
    const json = out[0..writer.end];

    try std.testing.expect(std.mem.indexOf(u8, json, "\"version\": \"benchmark-contract-v1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"target_filename\": \"gemma-4-12b-it-qat-q4_0.gguf\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"timing_status\"") != null);
}

test "benchmark contract pins model lineage" {
    try std.testing.expect(std.mem.indexOf(u8, model_lineage.revision_policy, "snapshot") != null);
    try std.testing.expect(std.mem.indexOf(u8, model_lineage.engine_artifact_policy, "--model") != null);
}

test "benchmark contract fixes deterministic generation settings" {
    try std.testing.expectEqual(@as(u32, 128), output_policy.max_new_tokens);
    try std.testing.expect(output_policy.stop_on_eos);
    try std.testing.expectEqual(@as(f32, 0.0), generation_settings.temperature);
    try std.testing.expectEqual(@as(u64, 0), generation_settings.seed);
}

test "benchmark prompt ids are unique and discoverable" {
    try std.testing.expect(prompt_suite.len >= 5);
    try std.testing.expect(promptById("latency_math") != null);
    try std.testing.expect(promptById("missing") == null);

    for (prompt_suite, 0..) |left, index| {
        for (prompt_suite[index + 1 ..]) |right| {
            try std.testing.expect(!std.mem.eql(u8, left.id, right.id));
        }
    }
}

test "benchmark contract includes primary and secondary metrics" {
    var primary_count: usize = 0;
    var secondary_count: usize = 0;

    for (metrics) |metric| {
        switch (metric.priority) {
            .primary => primary_count += 1,
            .secondary => secondary_count += 1,
        }
    }

    try std.testing.expectEqual(@as(usize, 3), primary_count);
    try std.testing.expect(secondary_count >= 3);
}

test "benchmark contract writer prints MLX baseline" {
    var out: [20000]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try writeContract(&writer);

    try std.testing.expect(std.mem.indexOf(
        u8,
        out[0..writer.end],
        "python -m mlx_lm.generate",
    ) != null);
}
