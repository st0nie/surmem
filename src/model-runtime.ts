import { ValidationError } from "./errors";

export type GgufGpu = "auto" | "metal" | "cuda" | "vulkan" | false;
export type ModelSource = "huggingface" | "modelscope";

export const HUGGING_FACE_EMBEDDING_MODEL_URI =
  "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
export const HUGGING_FACE_JUDGE_MODEL_URI = "hf:ggml-org/Qwen3-4B-GGUF:Q4_K_M";
export const MODELSCOPE_EMBEDDING_MODEL_URI =
  "https://modelscope.cn/models/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/resolve/master/embeddinggemma-300m-qat-Q8_0.gguf";
export const MODELSCOPE_JUDGE_MODEL_URI =
  "https://modelscope.cn/models/unsloth/Qwen3-4B-GGUF/resolve/master/Qwen3-4B-Q4_K_M.gguf";

const CHINA_TIME_ZONES = new Set([
  "Asia/Chongqing",
  "Asia/Harbin",
  "Asia/Hong_Kong",
  "Asia/Kashgar",
  "Asia/Macau",
  "Asia/Shanghai",
  "Asia/Urumqi",
]);

export function defaultGgufGpu(platform: NodeJS.Platform = process.platform): GgufGpu {
  return platform === "darwin" ? "metal" : false;
}

export function resolveModelSource(
  configured: string | undefined = process.env.SURMEM_MODEL_SOURCE,
  timeZone: string = Intl.DateTimeFormat().resolvedOptions().timeZone,
): ModelSource {
  const value = configured?.trim().toLowerCase();
  if (value === "huggingface" || value === "modelscope") return value;
  if (value && value !== "auto") {
    throw new ValidationError(
      `SURMEM_MODEL_SOURCE must be auto, huggingface, or modelscope; received ${configured}.`,
    );
  }
  return CHINA_TIME_ZONES.has(timeZone) ? "modelscope" : "huggingface";
}

export function defaultEmbeddingModelUri(source: ModelSource = resolveModelSource()): string {
  return source === "modelscope" ? MODELSCOPE_EMBEDDING_MODEL_URI : HUGGING_FACE_EMBEDDING_MODEL_URI;
}

export function defaultJudgeModelUri(source: ModelSource = resolveModelSource()): string {
  return source === "modelscope" ? MODELSCOPE_JUDGE_MODEL_URI : HUGGING_FACE_JUDGE_MODEL_URI;
}
