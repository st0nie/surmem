import { describe, expect, test } from "bun:test";

import { DaemonMemoryJudge, DEFAULT_GGUF_MODEL_URI, DEFAULT_JUDGE_MODEL_URI } from "../src/index";
import {
  defaultEmbeddingModelUri,
  defaultGgufGpu,
  defaultJudgeModelUri,
  HUGGING_FACE_EMBEDDING_MODEL_URI,
  HUGGING_FACE_JUDGE_MODEL_URI,
  MODELSCOPE_EMBEDDING_MODEL_URI,
  MODELSCOPE_JUDGE_MODEL_URI,
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
    const source = resolveModelSource("auto", timeZone);

    // Then
    expect(source).toBe("modelscope");
  });

  test("uses Hugging Face outside China", () => {
    // Given
    const timeZone = "Europe/London";

    // When
    const source = resolveModelSource("auto", timeZone);

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

  test.each([
    [
      "huggingface",
      HUGGING_FACE_JUDGE_MODEL_URI,
      "hf:unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf",
    ],
    [
      "modelscope",
      MODELSCOPE_JUDGE_MODEL_URI,
      "https://modelscope.cn/models/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/master/Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf",
    ],
  ] as const)("selects the exact Unsloth Instruct UD-Q4_K_XL judge from %s", (source, uri, expected) => {
    expect(uri).toBe(expected);
    expect(defaultJudgeModelUri(source)).toBe(expected);

    const previousSource = process.env.SURMEM_MODEL_SOURCE;
    try {
      process.env.SURMEM_MODEL_SOURCE = source;
      expect(defaultJudgeModelUri()).toBe(expected);
      expect(new DaemonMemoryJudge().diagnostics().model).toBe(expected);
    } finally {
      if (previousSource === undefined) delete process.env.SURMEM_MODEL_SOURCE;
      else process.env.SURMEM_MODEL_SOURCE = previousSource;
    }
  });

  test("exports the Hugging Face judge default through the public facade", () => {
    expect(DEFAULT_JUDGE_MODEL_URI).toBe(HUGGING_FACE_JUDGE_MODEL_URI);
  });

  test("preserves explicit judge URI and local path overrides", () => {
    const modelUri = "hf:test/custom-judge.gguf";
    const modelPath = "/models/custom-judge.gguf";

    expect(new DaemonMemoryJudge({ modelUri }).diagnostics().model).toBe(modelUri);
    expect(new DaemonMemoryJudge({ modelUri, modelPath }).diagnostics().model).toBe(modelPath);
  });

  test("keeps both EmbeddingGemma sources and the public default unchanged", () => {
    expect(HUGGING_FACE_EMBEDDING_MODEL_URI).toBe(
      "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
    );
    expect(MODELSCOPE_EMBEDDING_MODEL_URI).toBe(
      "https://modelscope.cn/models/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/master/embeddinggemma-300m-qat-Q8_0.gguf",
    );
    expect(defaultEmbeddingModelUri("huggingface")).toBe(HUGGING_FACE_EMBEDDING_MODEL_URI);
    expect(defaultEmbeddingModelUri("modelscope")).toBe(MODELSCOPE_EMBEDDING_MODEL_URI);
    expect(DEFAULT_GGUF_MODEL_URI).toBe(HUGGING_FACE_EMBEDDING_MODEL_URI);
  });
});
