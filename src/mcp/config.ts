/** Host-independent configuration, matching Pi's storage and embedding identities. */
import { createHash } from "node:crypto";
import { existsSync, realpathSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { basename, isAbsolute, join, resolve } from "node:path";
import { DaemonGgufEmbedder } from "../daemon-embedder";
import { DaemonMemoryJudge } from "../daemon-judge";
import { type Embedder, HashEmbedder, OpenAIEmbedder } from "../embeddings";
import { ValidationError } from "../errors";
import type { LLMJudge } from "../gate";
import { OpenAIJudge } from "../judge";
import { defaultGgufGpu } from "../model-runtime";

export interface McpConfig {
  projectPath: string;
  projectKey: string;
  projectName: string;
  storageDir: string;
  configPath: string;
}

export function parseArgs(args: string[], env: NodeJS.ProcessEnv = process.env): McpConfig {
  let project: string | undefined;
  let storage: string | undefined;
  const seen = new Set<string>();
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index];
    const value = args[index + 1];
    if (
      !value ||
      value.startsWith("--") ||
      !["--project", "--storage-dir"].includes(flag) ||
      seen.has(flag)
    ) {
      throw new ValidationError("Usage: surmem-mcp --project /absolute/project [--storage-dir /path]");
    }
    seen.add(flag);
    if (flag === "--project") project = value;
    else storage = value;
  }
  if (!project || !isAbsolute(project)) {
    throw new ValidationError("--project must name an existing absolute project directory.");
  }
  const projectPath = realpathSync(project);
  if (!statSync(projectPath).isDirectory()) throw new ValidationError(`Not a project directory: ${project}`);
  const storageDir = resolve(
    storage ?? env.SURMEM_DIR ?? join(env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent"), "surmem"),
  );
  return {
    projectPath,
    projectKey: createHash("sha256").update(projectPath).digest("hex").slice(0, 20),
    projectName: basename(projectPath) || "project",
    storageDir,
    configPath: resolve(env.SURMEM_CONFIG_PATH ?? join(storageDir, "config.json")),
  };
}

function integer(env: NodeJS.ProcessEnv, key: string, fallback: number, min: number, max: number): number {
  const value = env[key] === undefined ? fallback : Number(env[key]);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new ValidationError(`${key} must be an integer from ${min} to ${max}.`);
  }
  return value;
}

function gpu(value: string | undefined): "auto" | "metal" | "cuda" | "vulkan" | false {
  if (value === undefined) return defaultGgufGpu();
  if (value === "false" || value === "cpu") return false;
  if (value === "auto" || value === "metal" || value === "cuda" || value === "vulkan") return value;
  throw new ValidationError("GGUF GPU must be cpu, false, auto, metal, cuda, or vulkan.");
}

function modelPath(env: NodeJS.ProcessEnv, key: string): string | undefined {
  const path = env[key];
  if (path && !existsSync(path)) throw new ValidationError(`${key} does not exist: ${path}`);
  return path;
}

export function createEmbedders(
  storageDir: string,
  env: NodeJS.ProcessEnv = process.env,
): { document: Embedder; query: Embedder } {
  const backend = (env.SURMEM_EMBEDDER ?? "").trim().toLowerCase();
  if (backend === "hash") {
    const document = new HashEmbedder();
    return { document, query: document };
  }
  if (backend === "api" || (!backend && env.SURMEM_EMBEDDING_API_KEY)) {
    if (!env.SURMEM_EMBEDDING_API_KEY)
      throw new ValidationError("SURMEM_EMBEDDER=api requires SURMEM_EMBEDDING_API_KEY.");
    const document = new OpenAIEmbedder({
      apiKey: env.SURMEM_EMBEDDING_API_KEY,
      baseUrl: env.SURMEM_EMBEDDING_BASE_URL,
      model: env.SURMEM_EMBEDDING_MODEL,
      dim: integer(env, "SURMEM_EMBEDDING_DIM", 1536, 1, 65_536),
      timeoutMs: integer(env, "SURMEM_HTTP_TIMEOUT_MS", 30_000, 1000, 300_000),
    });
    return { document, query: document };
  }
  if (backend && backend !== "gguf") throw new ValidationError("SURMEM_EMBEDDER must be hash, api, or gguf.");
  return DaemonGgufEmbedder.createPair({
    daemonDir: join(storageDir, "embedding-daemon"),
    modelPath: modelPath(env, "SURMEM_GGUF_MODEL_PATH"),
    modelUri: env.SURMEM_GGUF_MODEL_URI,
    dim: integer(env, "SURMEM_GGUF_DIM", 768, 1, 65_536),
    gpu: gpu(env.SURMEM_GGUF_GPU),
    startupTimeoutMs: integer(env, "SURMEM_GGUF_STARTUP_TIMEOUT_MS", 15 * 60_000, 1000, 60 * 60_000),
    requestTimeoutMs: integer(env, "SURMEM_GGUF_REQUEST_TIMEOUT_MS", 120_000, 1000, 30 * 60_000),
    idleMs: integer(env, "SURMEM_GGUF_DAEMON_IDLE_MS", 30 * 60_000, 60_000, 24 * 60 * 60_000),
  });
}

export function createArbiter(
  storageDir: string,
  env: NodeJS.ProcessEnv = process.env,
): (LLMJudge & { dispose?(): void }) | undefined {
  if (env.SURMEM_JUDGE_MODE?.toLowerCase() === "heuristic") return undefined;
  const prefix = env.SURMEM_ARBITER_API_KEY && env.SURMEM_ARBITER_MODEL ? "SURMEM_ARBITER" : "SURMEM_JUDGE";
  const apiKey = env[`${prefix}_API_KEY`];
  const model = env[`${prefix}_MODEL`];
  if (apiKey && model) return new OpenAIJudge({ apiKey, model, baseUrl: env[`${prefix}_BASE_URL`] });
  return new DaemonMemoryJudge({
    daemonDir: join(storageDir, "judgment-daemon"),
    modelPath: modelPath(env, "SURMEM_JUDGE_GGUF"),
    modelUri: env.SURMEM_JUDGE_GGUF_URI,
    gpu: gpu(env.SURMEM_JUDGE_GGUF_GPU ?? env.SURMEM_GGUF_GPU),
    startupTimeoutMs: integer(env, "SURMEM_JUDGE_STARTUP_TIMEOUT_MS", 30 * 60_000, 1000, 60 * 60_000),
    requestTimeoutMs: integer(env, "SURMEM_JUDGE_TIMEOUT_MS", 180_000, 1000, 30 * 60_000),
    idleMs: integer(env, "SURMEM_JUDGE_DAEMON_IDLE_MS", 30 * 60_000, 60_000, 24 * 60 * 60_000),
  });
}
