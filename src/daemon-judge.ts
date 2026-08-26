/** Shared local GGUF model for durable-memory judgment and contradiction arbitration. */

import { spawn } from "node:child_process";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { closeSync, constants, openSync } from "node:fs";
import { chmod, mkdir, open, readFile, stat, unlink, writeFile } from "node:fs/promises";
import { request as httpRequest } from "node:http";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { ValidationError } from "./errors";
import type { LLMJudge, LLMJudgeDecision } from "./gate";
import type { MemorabilityJudge } from "./judge";
import { sanitizeForPrompt } from "./safety";
import { WriteVerdict } from "./types";

export const DEFAULT_JUDGE_MODEL_URI = "hf:ggml-org/Qwen3-4B-GGUF:Q4_K_M";

export interface DaemonMemoryJudgeOptions {
  daemonDir?: string;
  modelUri?: string;
  modelPath?: string;
  gpu?: "auto" | "metal" | "cuda" | "vulkan" | false;
  startupTimeoutMs?: number;
  requestTimeoutMs?: number;
  idleMs?: number;
  contextSize?: number;
}

export interface JudgmentDaemonProgress {
  phase: "resolving" | "downloading" | "loading" | "starting" | "ready" | "error";
  pid: number;
  fingerprint: string;
  model: string;
  modelPath?: string;
  downloadedSize?: number;
  totalSize?: number;
  percent?: number;
  averageSpeed?: number;
  estimatedTimeLeft?: number | null;
  host?: string;
  port?: number;
  error?: string;
  updatedAt: string;
  alive: boolean;
}

export interface JudgmentDaemonStatus {
  ok: boolean;
  protocol: number;
  pid: number;
  fingerprint: string;
  modelPath: string;
  lastUsedAt: number;
  host: string;
  port: number;
}

type Endpoint = {
  protocol: number;
  pid: number;
  host: string;
  port: number;
  fingerprint: string;
  modelPath: string;
};

function defaultDaemonDir(): string {
  const agentDir = process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent");
  return join(agentDir, "surmem", "judgment-daemon");
}

function integer(value: number, name: string, min: number, max: number): number {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new ValidationError(`${name} must be an integer from ${min} to ${max}.`);
  }
  return value;
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code !== "ESRCH";
  }
}

async function endpoint(path: string): Promise<Endpoint | null> {
  try {
    const value = JSON.parse(await readFile(path, "utf8")) as Partial<Endpoint>;
    if (
      value.protocol !== 1 ||
      !Number.isSafeInteger(value.pid) ||
      typeof value.host !== "string" ||
      !Number.isInteger(value.port) ||
      typeof value.fingerprint !== "string" ||
      typeof value.modelPath !== "string"
    ) {
      return null;
    }
    return value as Endpoint;
  } catch {
    return null;
  }
}

async function token(path: string): Promise<string> {
  try {
    const value = (await readFile(path, "utf8")).trim();
    if (value.length >= 32) return value;
    throw new Error(`Invalid judgment daemon token: ${path}`);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const value = randomBytes(32).toString("hex");
  try {
    await writeFile(path, `${value}\n`, { encoding: "utf8", mode: 0o600, flag: "wx" });
    await chmod(path, 0o600);
    return value;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    return (await readFile(path, "utf8")).trim();
  }
}

function requestJson<T>(options: {
  endpoint: Endpoint;
  token: string;
  path: string;
  method: "GET" | "POST";
  body?: unknown;
  timeoutMs: number;
  signal?: AbortSignal;
}): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const body = options.body === undefined ? undefined : JSON.stringify(options.body);
    const request = httpRequest(
      {
        host: options.endpoint.host,
        port: options.endpoint.port,
        path: options.path,
        method: options.method,
        agent: false,
        headers: {
          authorization: `Bearer ${options.token}`,
          connection: "close",
          ...(body ? { "content-type": "application/json", "content-length": Buffer.byteLength(body) } : {}),
        },
      },
      (response) => {
        const chunks: Buffer[] = [];
        response.on("data", (chunk: Buffer) => chunks.push(chunk));
        response.on("end", () => {
          try {
            const payload = JSON.parse(Buffer.concat(chunks).toString("utf8")) as T & {
              error?: string;
            };
            if ((response.statusCode ?? 500) >= 400)
              reject(new Error(payload.error ?? "Judgment daemon error."));
            else resolve(payload);
          } catch (error) {
            reject(error);
          }
        });
      },
    );
    const timer = setTimeout(
      () => request.destroy(new Error("Judgment daemon request timed out.")),
      options.timeoutMs,
    );
    timer.unref?.();
    const onAbort = () => request.destroy(new Error("Judgment request aborted."));
    options.signal?.addEventListener("abort", onAbort, { once: true });
    request.on("error", reject);
    request.on("close", () => {
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", onAbort);
    });
    if (body) request.write(body);
    request.end();
  });
}

async function health(value: Endpoint | null, auth: string): Promise<JudgmentDaemonStatus | null> {
  if (!value || !alive(value.pid)) return null;
  try {
    const result = await requestJson<JudgmentDaemonStatus>({
      endpoint: value,
      token: auth,
      path: "/health",
      method: "GET",
      timeoutMs: 1500,
    });
    return result.ok ? { ...result, host: value.host, port: value.port } : null;
  } catch {
    return null;
  }
}

function parse(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  const withoutThinking = trimmed.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  for (const candidate of [withoutThinking, withoutThinking.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1]]) {
    if (!candidate) continue;
    try {
      const result = JSON.parse(candidate.trim());
      if (result && typeof result === "object" && !Array.isArray(result)) return result;
    } catch {}
  }
  const start = withoutThinking.indexOf("{");
  const end = withoutThinking.lastIndexOf("}");
  if (start >= 0 && end > start) {
    try {
      return JSON.parse(withoutThinking.slice(start, end + 1)) as Record<string, unknown>;
    } catch {}
  }
  return null;
}

export class DaemonMemoryJudge implements MemorabilityJudge, LLMJudge {
  readonly fingerprint: string;
  private readonly daemonDir: string;
  private readonly modelUri: string;
  private readonly modelPath?: string;
  private readonly gpu: DaemonMemoryJudgeOptions["gpu"];
  private readonly startupTimeoutMs: number;
  private readonly requestTimeoutMs: number;
  private readonly idleMs: number;
  private readonly contextSize: number;
  private connectionPromise: Promise<{ endpoint: Endpoint; token: string }> | null = null;
  private calls = 0;
  private failures = 0;
  private lastError: string | null = null;

  constructor(options: DaemonMemoryJudgeOptions = {}) {
    this.daemonDir = options.daemonDir ?? defaultDaemonDir();
    this.modelUri = options.modelUri ?? DEFAULT_JUDGE_MODEL_URI;
    this.modelPath = options.modelPath;
    this.gpu = options.gpu ?? false;
    this.startupTimeoutMs = integer(
      options.startupTimeoutMs ?? 30 * 60_000,
      "startupTimeoutMs",
      1000,
      60 * 60_000,
    );
    this.requestTimeoutMs = integer(
      options.requestTimeoutMs ?? 180_000,
      "requestTimeoutMs",
      1000,
      30 * 60_000,
    );
    this.idleMs = integer(options.idleMs ?? 30 * 60_000, "idleMs", 60_000, 24 * 60 * 60_000);
    this.contextSize = integer(options.contextSize ?? 4096, "contextSize", 1024, 32_768);
    this.fingerprint = `gguf-judge:${createHash("sha256")
      .update(`${this.modelPath ?? this.modelUri}\0${this.contextSize}`)
      .digest("hex")
      .slice(0, 24)}`;
  }

  private async connect(): Promise<{ endpoint: Endpoint; token: string }> {
    if (!this.connectionPromise) {
      this.connectionPromise = this.ensure().catch((error) => {
        this.connectionPromise = null;
        throw error;
      });
    }
    const result = await this.connectionPromise;
    const status = await health(result.endpoint, result.token);
    if (!status) {
      this.connectionPromise = null;
      return this.connect();
    }
    if (status.fingerprint !== this.fingerprint) {
      throw new Error(`Judgment daemon serves ${status.fingerprint}; expected ${this.fingerprint}.`);
    }
    return result;
  }

  private async ensure(): Promise<{ endpoint: Endpoint; token: string }> {
    const endpointPath = join(this.daemonDir, "endpoint.json");
    const tokenPath = join(this.daemonDir, "token");
    const lockPath = join(this.daemonDir, "startup.lock");
    await mkdir(this.daemonDir, { recursive: true, mode: 0o700 });
    const auth = await token(tokenPath);
    const current = await endpoint(endpointPath);
    const currentHealth = await health(current, auth);
    if (currentHealth) {
      if (currentHealth.fingerprint !== this.fingerprint)
        throw new Error("A different judgment model is already running.");
      return { endpoint: current as Endpoint, token: auth };
    }

    const owner = `${process.pid}:${randomUUID()}`;
    const deadline = Date.now() + this.startupTimeoutMs;
    let locked = false;
    while (!locked) {
      try {
        const handle = await open(lockPath, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY, 0o600);
        await handle.writeFile(owner);
        await handle.close();
        locked = true;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
        const readyEndpoint = await endpoint(endpointPath);
        const ready = await health(readyEndpoint, auth);
        if (ready) return { endpoint: readyEndpoint as Endpoint, token: auth };
        try {
          const info = await stat(lockPath);
          if (Date.now() - info.mtimeMs > this.startupTimeoutMs) await unlink(lockPath);
        } catch {}
        if (Date.now() >= deadline)
          throw new Error(`Timed out waiting for judgment daemon: ${this.daemonDir}`);
        await new Promise((resolve) => setTimeout(resolve, 250));
      }
    }

    try {
      await unlink(endpointPath).catch(() => {});
      const entry = fileURLToPath(new URL("./judgment-daemon-entry.mjs", import.meta.url));
      const logPath = join(this.daemonDir, "daemon.log");
      const logFd = openSync(logPath, "a", 0o600);
      let child: ReturnType<typeof spawn>;
      try {
        child = spawn(process.execPath, [entry], {
          detached: true,
          stdio: ["ignore", logFd, logFd],
          env: {
            ...process.env,
            NODE_USE_ENV_PROXY: process.env.NODE_USE_ENV_PROXY ?? "1",
            SURMEM_JUDGE_DAEMON_DIR: this.daemonDir,
            SURMEM_JUDGE_DAEMON_TOKEN_FILE: tokenPath,
            SURMEM_JUDGE_DAEMON_FINGERPRINT: this.fingerprint,
            SURMEM_JUDGE_DAEMON_MODEL_URI: this.modelUri,
            ...(this.modelPath ? { SURMEM_JUDGE_DAEMON_MODEL_PATH: this.modelPath } : {}),
            SURMEM_JUDGE_DAEMON_GPU: this.gpu === false ? "false" : String(this.gpu),
            SURMEM_JUDGE_DAEMON_IDLE_MS: String(this.idleMs),
            SURMEM_JUDGE_CONTEXT_SIZE: String(this.contextSize),
          },
        });
        child.unref();
      } finally {
        closeSync(logFd);
      }
      while (Date.now() < deadline) {
        if (child.exitCode !== null)
          throw new Error(`Judgment daemon exited with code ${child.exitCode}. See ${logPath}`);
        const readyEndpoint = await endpoint(endpointPath);
        const ready = await health(readyEndpoint, auth);
        if (ready) return { endpoint: readyEndpoint as Endpoint, token: auth };
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      throw new Error(`Timed out starting judgment daemon. See ${logPath}`);
    } finally {
      try {
        if ((await readFile(lockPath, "utf8")) === owner) await unlink(lockPath);
      } catch {}
    }
  }

  private async call(
    path: "/assess" | "/arbitrate",
    body: unknown,
    signal?: AbortSignal,
  ): Promise<Record<string, unknown> | null> {
    const connection = await this.connect();
    this.calls++;
    try {
      const result = await requestJson<{ output: string }>({
        endpoint: connection.endpoint,
        token: connection.token,
        path,
        method: "POST",
        body,
        timeoutMs: this.requestTimeoutMs,
        signal,
      });
      this.lastError = null;
      return parse(result.output);
    } catch (error) {
      this.failures++;
      this.lastError = error instanceof Error ? error.message : String(error);
      this.connectionPromise = null;
      throw error;
    }
  }

  async assess(text: string, signal?: AbortSignal): Promise<string | null> {
    const result = await this.call("/assess", { text: sanitizeForPrompt(text, 6000) }, signal);
    const confidence = Number(result?.confidence);
    if (result?.memorable !== true || !Number.isFinite(confidence) || confidence < 0.72) return null;
    const canonical =
      typeof result.canonicalText === "string" ? sanitizeForPrompt(result.canonicalText, 2000) : "";
    return canonical || null;
  }

  async arbitrate(newText: string, nearestText: string, signal?: AbortSignal): Promise<LLMJudgeDecision> {
    try {
      const result = await this.call(
        "/arbitrate",
        { oldText: sanitizeForPrompt(nearestText, 4000), newText: sanitizeForPrompt(newText, 4000) },
        signal,
      );
      const verdict = result?.verdict;
      const confidence = Number(result?.confidence);
      return {
        verdict: Object.values(WriteVerdict).includes(verdict as WriteVerdict)
          ? (verdict as WriteVerdict)
          : WriteVerdict.NOOP,
        confidence: Number.isFinite(confidence) ? Math.max(0, Math.min(1, confidence)) : 0,
        reason: typeof result?.reason === "string" ? result.reason.slice(0, 300) : "invalid-output",
      };
    } catch (error) {
      return {
        verdict: WriteVerdict.NOOP,
        confidence: 0,
        reason: `judge-failed: ${error instanceof Error ? error.message : String(error)}`.slice(0, 300),
      };
    }
  }

  async progress(): Promise<JudgmentDaemonProgress | null> {
    try {
      const value = JSON.parse(await readFile(join(this.daemonDir, "state.json"), "utf8")) as Omit<
        JudgmentDaemonProgress,
        "alive"
      >;
      return { ...value, alive: alive(value.pid) };
    } catch {
      return null;
    }
  }

  async status(): Promise<JudgmentDaemonStatus | null> {
    try {
      const auth = (await readFile(join(this.daemonDir, "token"), "utf8")).trim();
      return health(await endpoint(join(this.daemonDir, "endpoint.json")), auth);
    } catch {
      return null;
    }
  }

  diagnostics() {
    return {
      backend: "gguf-daemon",
      model: this.modelPath ?? this.modelUri,
      calls: this.calls,
      failures: this.failures,
      lastError: this.lastError,
    };
  }

  dispose(): void {
    this.connectionPromise = null;
  }
}
