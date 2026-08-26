/** Cross-process GGUF embedder backed by one shared local daemon. */

import { spawn } from "node:child_process";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { closeSync, constants, openSync } from "node:fs";
import { chmod, mkdir, open, readFile, stat, unlink, writeFile } from "node:fs/promises";
import { request as httpRequest } from "node:http";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { Embedder } from "./embeddings";
import { validateVector } from "./embeddings";
import { PersistenceError, ValidationError } from "./errors";

export const DEFAULT_GGUF_MODEL_URI = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";

export interface DaemonGgufEmbedderOptions {
  daemonDir?: string;
  modelUri?: string;
  modelPath?: string;
  dim?: number;
  gpu?: "auto" | "metal" | "cuda" | "vulkan" | false;
  documentTemplate?: string;
  queryTemplate?: string;
  startupTimeoutMs?: number;
  requestTimeoutMs?: number;
  idleMs?: number;
}

export interface EmbeddingDaemonProgress {
  phase: "resolving" | "downloading" | "loading" | "starting" | "ready" | "error";
  pid: number;
  fingerprint: string;
  dim: number;
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

export interface EmbeddingDaemonStatus {
  ok: boolean;
  protocol: number;
  pid: number;
  fingerprint: string;
  dim: number;
  modelPath: string;
  lastUsedAt: number;
  host: string;
  port: number;
}

type Role = "document" | "query";
type Endpoint = {
  protocol: number;
  pid: number;
  host: string;
  port: number;
  fingerprint: string;
  dim: number;
  modelPath: string;
  startedAt?: string;
};

function defaultDaemonDir(): string {
  const agentDir = process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent");
  return join(agentDir, "surmem", "embedding-daemon");
}

function positiveInteger(value: number, name: string, min: number, max: number): number {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new ValidationError(`${name} must be an integer from ${min} to ${max}.`);
  }
  return value;
}

function processAlive(pid: number): boolean {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code !== "ESRCH";
  }
}

async function readEndpoint(path: string): Promise<Endpoint | null> {
  try {
    const raw = JSON.parse(await readFile(path, "utf8")) as Partial<Endpoint>;
    if (
      raw.protocol !== 1 ||
      !Number.isSafeInteger(raw.pid) ||
      typeof raw.host !== "string" ||
      !Number.isInteger(raw.port) ||
      typeof raw.fingerprint !== "string" ||
      !Number.isInteger(raw.dim) ||
      typeof raw.modelPath !== "string"
    ) {
      return null;
    }
    return raw as Endpoint;
  } catch {
    return null;
  }
}

async function ensureToken(path: string): Promise<string> {
  try {
    const value = (await readFile(path, "utf8")).trim();
    if (value.length >= 32) return value;
    throw new PersistenceError(`Embedding daemon token is invalid: ${path}`);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const token = randomBytes(32).toString("hex");
  try {
    await writeFile(path, `${token}\n`, { encoding: "utf8", mode: 0o600, flag: "wx" });
    await chmod(path, 0o600);
    return token;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    const existing = (await readFile(path, "utf8")).trim();
    if (existing.length < 32) throw new PersistenceError(`Embedding daemon token is invalid: ${path}`);
    return existing;
  }
}

function requestJson<T>(options: {
  endpoint: Endpoint;
  token: string;
  method: "GET" | "POST";
  path: string;
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
        method: options.method,
        path: options.path,
        agent: false,
        headers: {
          authorization: `Bearer ${options.token}`,
          connection: "close",
          ...(body ? { "content-type": "application/json", "content-length": Buffer.byteLength(body) } : {}),
        },
      },
      (response) => {
        const chunks: Buffer[] = [];
        let size = 0;
        response.on("data", (chunk: Buffer) => {
          size += chunk.length;
          if (size > 128 * 1024 * 1024) request.destroy(new Error("Embedding daemon response is too large."));
          else chunks.push(chunk);
        });
        response.on("end", () => {
          try {
            const payload = JSON.parse(Buffer.concat(chunks).toString("utf8")) as T & { error?: string };
            if ((response.statusCode ?? 500) >= 400) {
              reject(new Error(payload.error ?? `Embedding daemon HTTP ${response.statusCode}`));
            } else {
              resolve(payload);
            }
          } catch (error) {
            reject(error);
          }
        });
      },
    );
    const timer = setTimeout(
      () => request.destroy(new Error("Embedding daemon request timed out.")),
      options.timeoutMs,
    );
    timer.unref?.();
    const onAbort = () =>
      request.destroy(
        options.signal?.reason instanceof Error
          ? options.signal.reason
          : new Error("Embedding request aborted."),
      );
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

async function healthy(
  endpoint: Endpoint | null,
  token: string,
  timeoutMs: number,
): Promise<EmbeddingDaemonStatus | null> {
  if (!endpoint || !processAlive(endpoint.pid)) return null;
  try {
    const status = await requestJson<EmbeddingDaemonStatus>({
      endpoint,
      token,
      method: "GET",
      path: "/health",
      timeoutMs,
    });
    return status.ok && status.protocol === 1
      ? { ...status, host: endpoint.host, port: endpoint.port }
      : null;
  } catch {
    return null;
  }
}

interface SharedClientState {
  endpointPromise: Promise<{ endpoint: Endpoint; token: string }> | null;
}

export class DaemonGgufEmbedder implements Embedder {
  readonly dim: number;
  readonly fingerprint: string;
  private readonly daemonDir: string;
  private readonly modelUri: string;
  private readonly modelPath?: string;
  private readonly gpu: DaemonGgufEmbedderOptions["gpu"];
  private readonly documentTemplate: string;
  private readonly queryTemplate: string;
  private readonly startupTimeoutMs: number;
  private readonly requestTimeoutMs: number;
  private readonly idleMs: number;
  private readonly role: Role;
  private readonly shared: SharedClientState;

  constructor(options: DaemonGgufEmbedderOptions = {}, role: Role = "document", shared?: SharedClientState) {
    this.daemonDir = options.daemonDir ?? defaultDaemonDir();
    this.modelUri = options.modelUri ?? DEFAULT_GGUF_MODEL_URI;
    this.modelPath = options.modelPath;
    this.dim = positiveInteger(options.dim ?? 768, "dim", 1, 65_536);
    this.gpu = options.gpu ?? false;
    this.documentTemplate = options.documentTemplate ?? "title: none | text: {text}";
    this.queryTemplate = options.queryTemplate ?? "task: search result | query: {text}";
    if (!this.documentTemplate.includes("{text}") || !this.queryTemplate.includes("{text}")) {
      throw new ValidationError("GGUF embedding templates must contain {text}.");
    }
    this.startupTimeoutMs = positiveInteger(
      options.startupTimeoutMs ?? 15 * 60_000,
      "startupTimeoutMs",
      1000,
      60 * 60_000,
    );
    this.requestTimeoutMs = positiveInteger(
      options.requestTimeoutMs ?? 120_000,
      "requestTimeoutMs",
      1000,
      30 * 60_000,
    );
    this.idleMs = positiveInteger(options.idleMs ?? 30 * 60_000, "idleMs", 60_000, 24 * 60 * 60_000);
    this.role = role;
    this.shared = shared ?? { endpointPromise: null };
    const identity = createHash("sha256")
      .update(
        `${this.modelPath ?? this.modelUri}\0${this.dim}\0${this.documentTemplate}\0${this.queryTemplate}`,
      )
      .digest("hex")
      .slice(0, 24);
    this.fingerprint = `gguf-daemon:${identity}:${this.dim}`;
  }

  static createPair(options: DaemonGgufEmbedderOptions = {}): {
    document: DaemonGgufEmbedder;
    query: DaemonGgufEmbedder;
  } {
    const shared: SharedClientState = { endpointPromise: null };
    return {
      document: new DaemonGgufEmbedder(options, "document", shared),
      query: new DaemonGgufEmbedder(options, "query", shared),
    };
  }

  private async connection(): Promise<{ endpoint: Endpoint; token: string }> {
    if (!this.shared.endpointPromise) {
      this.shared.endpointPromise = this.ensureDaemon().catch((error) => {
        this.shared.endpointPromise = null;
        throw error;
      });
    }
    const connection = await this.shared.endpointPromise;
    const status = await healthy(connection.endpoint, connection.token, 1500);
    if (!status) {
      this.shared.endpointPromise = null;
      return this.connection();
    }
    if (status.fingerprint !== this.fingerprint || status.dim !== this.dim) {
      throw new EmbeddingDaemonMismatchError(
        `The shared embedding daemon already serves ${status.fingerprint}; requested ${this.fingerprint}. Stop it before changing models.`,
      );
    }
    return connection;
  }

  private async ensureDaemon(): Promise<{ endpoint: Endpoint; token: string }> {
    const endpointPath = join(this.daemonDir, "endpoint.json");
    const tokenPath = join(this.daemonDir, "token");
    const lockPath = join(this.daemonDir, "startup.lock");
    await mkdir(this.daemonDir, { recursive: true, mode: 0o700 });
    const token = await ensureToken(tokenPath);
    const existing = await readEndpoint(endpointPath);
    const existingStatus = await healthy(existing, token, 1500);
    if (existingStatus) {
      if (existingStatus.fingerprint !== this.fingerprint || existingStatus.dim !== this.dim) {
        throw new EmbeddingDaemonMismatchError(
          `The shared embedding daemon already serves ${existingStatus.fingerprint}; requested ${this.fingerprint}.`,
        );
      }
      return { endpoint: existing as Endpoint, token };
    }

    const ownerToken = `${process.pid}:${randomUUID()}`;
    const deadline = Date.now() + this.startupTimeoutMs;
    let ownsLock = false;
    while (!ownsLock) {
      try {
        const handle = await open(lockPath, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY, 0o600);
        await handle.writeFile(ownerToken, "utf8");
        await handle.close();
        ownsLock = true;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
        const endpoint = await readEndpoint(endpointPath);
        const status = await healthy(endpoint, token, 1500);
        if (status) {
          if (status.fingerprint !== this.fingerprint || status.dim !== this.dim) {
            throw new EmbeddingDaemonMismatchError(
              `The shared embedding daemon already serves ${status.fingerprint}; requested ${this.fingerprint}.`,
            );
          }
          return { endpoint: endpoint as Endpoint, token };
        }
        try {
          const info = await stat(lockPath);
          if (Date.now() - info.mtimeMs > this.startupTimeoutMs) await unlink(lockPath);
        } catch {}
        if (Date.now() >= deadline)
          throw new Error(
            `Timed out waiting for GGUF embedding daemon. See ${join(this.daemonDir, "daemon.log")}`,
          );
        await new Promise((resolve) => setTimeout(resolve, 250));
      }
    }

    try {
      const endpoint = await readEndpoint(endpointPath);
      const status = await healthy(endpoint, token, 1500);
      if (status) return { endpoint: endpoint as Endpoint, token };
      await unlink(endpointPath).catch(() => {});
      const entry = fileURLToPath(new URL("./embedding-daemon-entry.mjs", import.meta.url));
      const logPath = join(this.daemonDir, "daemon.log");
      const logFd = openSync(logPath, "a", 0o600);
      let child: ReturnType<typeof spawn>;
      try {
        child = spawn(process.execPath, [entry], {
          detached: true,
          stdio: ["ignore", logFd, logFd],
          env: {
            ...process.env,
            // Node does not honor HTTP(S)_PROXY by default. Node 24+ uses this
            // switch to route fetch/undici downloads through the user's proxy.
            NODE_USE_ENV_PROXY: process.env.NODE_USE_ENV_PROXY ?? "1",
            SURMEM_DAEMON_DIR: this.daemonDir,
            SURMEM_DAEMON_TOKEN_FILE: tokenPath,
            SURMEM_DAEMON_FINGERPRINT: this.fingerprint,
            SURMEM_DAEMON_MODEL_URI: this.modelUri,
            ...(this.modelPath ? { SURMEM_DAEMON_MODEL_PATH: this.modelPath } : {}),
            SURMEM_DAEMON_DIM: String(this.dim),
            SURMEM_DAEMON_GPU: this.gpu === false ? "false" : String(this.gpu),
            SURMEM_DAEMON_DOCUMENT_TEMPLATE: this.documentTemplate,
            SURMEM_DAEMON_QUERY_TEMPLATE: this.queryTemplate,
            SURMEM_DAEMON_IDLE_MS: String(this.idleMs),
          },
        });
        child.unref();
      } finally {
        closeSync(logFd);
      }
      while (Date.now() < deadline) {
        if (child.exitCode !== null) {
          throw new Error(`GGUF embedding daemon exited with code ${child.exitCode}. See ${logPath}`);
        }
        const readyEndpoint = await readEndpoint(endpointPath);
        const ready = await healthy(readyEndpoint, token, 1500);
        if (ready) {
          if (ready.fingerprint !== this.fingerprint || ready.dim !== this.dim) {
            throw new EmbeddingDaemonMismatchError(
              `Started daemon serves ${ready.fingerprint}; expected ${this.fingerprint}.`,
            );
          }
          return { endpoint: readyEndpoint as Endpoint, token };
        }
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      throw new Error(`Timed out starting GGUF embedding daemon. See ${logPath}`);
    } finally {
      try {
        if ((await readFile(lockPath, "utf8")) === ownerToken) await unlink(lockPath);
      } catch {}
    }
  }

  async embed(texts: string[], signal?: AbortSignal): Promise<number[][]> {
    if (texts.length === 0) return [];
    const connection = await this.connection();
    try {
      const payload = await requestJson<{ vectors: unknown[] }>({
        endpoint: connection.endpoint,
        token: connection.token,
        method: "POST",
        path: "/embed",
        body: { role: this.role, texts },
        timeoutMs: this.requestTimeoutMs,
        signal,
      });
      if (!Array.isArray(payload.vectors) || payload.vectors.length !== texts.length) {
        throw new ValidationError("Embedding daemon returned the wrong number of vectors.");
      }
      return payload.vectors.map((vector, index) =>
        validateVector(vector, this.dim, `daemon embedding[${index}]`),
      );
    } catch (error) {
      this.shared.endpointPromise = null;
      throw error;
    }
  }

  async progress(): Promise<EmbeddingDaemonProgress | null> {
    try {
      const raw = JSON.parse(await readFile(join(this.daemonDir, "state.json"), "utf8")) as Omit<
        EmbeddingDaemonProgress,
        "alive"
      >;
      if (
        !raw ||
        typeof raw !== "object" ||
        !Number.isSafeInteger(raw.pid) ||
        typeof raw.phase !== "string"
      ) {
        return null;
      }
      return { ...raw, alive: processAlive(raw.pid) };
    } catch {
      return null;
    }
  }

  async status(): Promise<EmbeddingDaemonStatus | null> {
    const endpointPath = join(this.daemonDir, "endpoint.json");
    const tokenPath = join(this.daemonDir, "token");
    try {
      const token = (await readFile(tokenPath, "utf8")).trim();
      return healthy(await readEndpoint(endpointPath), token, 1500);
    } catch {
      return null;
    }
  }

  dispose(): void {
    // Client shutdown must not stop the cross-process shared model.
    this.shared.endpointPromise = null;
  }
}

export class EmbeddingDaemonMismatchError extends Error {}

export async function stopEmbeddingDaemon(daemonDir = defaultDaemonDir()): Promise<boolean> {
  const endpoint = await readEndpoint(join(daemonDir, "endpoint.json"));
  if (!endpoint || !processAlive(endpoint.pid)) return false;
  process.kill(endpoint.pid, "SIGTERM");
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    if (!processAlive(endpoint.pid)) return true;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return false;
}
