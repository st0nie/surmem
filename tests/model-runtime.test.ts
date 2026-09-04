import { describe, expect, test } from "bun:test";

import {
  defaultEmbeddingModelUri,
  defaultGgufGpu,
  defaultJudgeModelUri,
  resolveModelSource,
} from "../src/model-runtime";

describe("model runtime defaults", () => {
  test("uses Metal on macOS when GPU is not configured", () => {
    // Given
    const platform = "darwin";

    // When
    const gpu = defaultGgufGpu(platform);

    // Then
    expect(gpu).toBe("metal");
  });

  test("keeps CPU default on non-macOS platforms", () => {
    // Given
    const platform = "linux";

    // When
    const gpu = defaultGgufGpu(platform);

    // Then
    expect(gpu).toBe(false);
  });

  test("uses ModelScope for a mainland China timezone", () => {
    // Given
    const timeZone = "Asia/Shanghai";

    // When
    const source = resolveModelSource(undefined, timeZone);

    // Then
    expect(source).toBe("modelscope");
  });

  test("uses Hugging Face outside China", () => {
    // Given
    const timeZone = "Europe/London";

    // When
    const source = resolveModelSource(undefined, timeZone);

    // Then
    expect(source).toBe("huggingface");
  });

  test("explicit model source overrides timezone detection", () => {
    // Given
    const timeZone = "Asia/Shanghai";

    // When
    const source = resolveModelSource("huggingface", timeZone);

    // Then
    expect(source).toBe("huggingface");
  });

  test("selects working ModelScope model URLs", () => {
    // Given
    const source = "modelscope";

    // When
    const embedding = defaultEmbeddingModelUri(source);
    const judge = defaultJudgeModelUri(source);

    // Then
    expect(new URL(embedding).hostname).toBe("modelscope.cn");
    expect(new URL(embedding).pathname).toEndWith(".gguf");
    expect(new URL(judge).hostname).toBe("modelscope.cn");
    expect(new URL(judge).pathname).toEndWith("Qwen3-4B-Q4_K_M.gguf");
  });
});
