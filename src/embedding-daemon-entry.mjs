#!/usr/bin/env node

import { chmodSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { chmod, mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { homedir } from "node:os";
import { join } from "node:path";
import { getLlama, resolveModelFile } from "node-llama-cpp";

const daemonDir = process.env.SURMEM_DAEMON_DIR;
const tokenFile = process.env.SURMEM_DAEMON_TOKEN_FILE;
const fingerprint = process.env.SURMEM_DAEMON_FINGERPRINT;
const modelUri = process.env.SURMEM_DAEMON_MODEL_URI;
const configuredModelPath = process.env.SURMEM_DAEMON_MODEL_PATH;
const expectedDim = Number(process.env.SURMEM_DAEMON_DIM ?? 768);
const idleMs = Number(process.env.SURMEM_DAEMON_IDLE_MS ?? 30 * 60_000);
const gpuValue = process.env.SURMEM_DAEMON_GPU ?? "false";
const documentTemplate = process.env.SURMEM_DAEMON_DOCUMENT_TEMPLATE ?? "title: none | text: {text}";
const queryTemplate = process.env.SURMEM_DAEMON_QUERY_TEMPLATE ?? "task: search result | query: {text}";
const endpointPath = daemonDir ? join(daemonDir, "endpoint.json") : "";
const statePath = daemonDir ? join(daemonDir, "state.json") : "";

if (!daemonDir || !tokenFile || !fingerprint || (!modelUri && !configuredModelPath)) {
  console.error("SurMem embedding daemon is missing required environment variables.");
  process.exit(2);
}
if (!Number.isInteger(expectedDim) || expectedDim < 1 || expectedDim > 65_536) {
  console.error(`Invalid embedding dimension: ${expectedDim}`);
  process.exit(2);
}

await mkdir(daemonDir, { recursive: true, mode: 0o700 });

function writeState(state) {
  const temp = `${statePath}.tmp-${process.pid}`;
  try {
    writeFileSync(
      temp,
      `${JSON.stringify(
        {
          protocol: 1,
          pid: process.pid,
          fingerprint,
          dim: expectedDim,
          model: configuredModelPath || modelUri,
          updatedAt: new Date().toISOString(),
          ...state,
        },
        null,
        2,
      )}\n`,
      { encoding: "utf8", mode: 0o600 },
    );
    renameSync(temp, statePath);
    chmodSync(statePath, 0o600);
  } finally {
    try {
      unlinkSync(temp);
    } catch {}
  }
}

const token = (await readFile(tokenFile, "utf8")).trim();
if (!token) {
  console.error("SurMem embedding daemon token is empty.");
  process.exit(2);
}

const gpu = gpuValue === "false" || gpuValue === "cpu" ? false : gpuValue;
const modelDir = process.env.SURMEM_DAEMON_MODEL_DIR ?? join(homedir(), ".cache", "qmd", "models");
await mkdir(modelDir, { recursive: true, mode: 0o700 });
console.error(`[surmem-embedding-daemon] resolving model ${configuredModelPath || modelUri}`);
writeState({ phase: "resolving", downloadedSize: 0, totalSize: 0, percent: 0 });
const downloadStartedAt = Date.now();
let lastProgressWrite = 0;
let modelPath;
let llama;
let model;
let context;
try {
  modelPath =
    configuredModelPath ||
    (await resolveModelFile(modelUri, {
      directory: modelDir,
      cli: false,
      verify: true,
      onProgress: ({ totalSize, downloadedSize }) => {
        const now = Date.now();
        if (downloadedSize < totalSize && now - lastProgressWrite < 500) return;
        lastProgressWrite = now;
        const elapsedSeconds = Math.max(0.001, (now - downloadStartedAt) / 1000);
        const averageSpeed = downloadedSize / elapsedSeconds;
        const remaining = Math.max(0, totalSize - downloadedSize);
        writeState({
          phase: "downloading",
          downloadedSize,
          totalSize,
          percent: totalSize > 0 ? (downloadedSize / totalSize) * 100 : 0,
          averageSpeed,
          estimatedTimeLeft: averageSpeed > 0 ? remaining / averageSpeed : null,
        });
      },
    }));
  console.error(`[surmem-embedding-daemon] loading ${modelPath}`);
  writeState({ phase: "loading", modelPath, percent: 100 });
  llama = await getLlama({ gpu });
  model = await llama.loadModel({ modelPath });
  context = await model.createEmbeddingContext();
  writeState({ phase: "starting", modelPath, percent: 100 });
} catch (error) {
  writeState({ phase: "error", error: error instanceof Error ? error.message : String(error) });
  await unlink(join(daemonDir, "startup.lock")).catch(() => {});
  throw error;
}
let lastUsedAt = Date.now();
let closing = false;
let requestTail = Promise.resolve();

function authorized(request) {
  return request.headers.authorization === `Bearer ${token}`;
}

function json(response, status, value) {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
    connection: "close",
  });
  response.end(body);
}

async function readJson(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > 2 * 1024 * 1024) throw new Error("request body exceeds 2 MiB");
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

async function embed(role, texts) {
  if (role !== "document" && role !== "query") throw new Error("role must be document or query");
  if (!Array.isArray(texts) || texts.length < 1 || texts.length > 64)
    throw new Error("texts must contain 1-64 strings");
  const template = role === "query" ? queryTemplate : documentTemplate;
  const vectors = [];
  for (const text of texts) {
    if (typeof text !== "string" || text.length < 1 || text.length > 20_000)
      throw new Error("embedding text length is invalid");
    const result = await context.getEmbeddingFor(template.replace("{text}", text));
    const vector = Array.from(result.vector);
    if (vector.length !== expectedDim || !vector.every(Number.isFinite)) {
      throw new Error(`model returned ${vector.length} dimensions; expected ${expectedDim}`);
    }
    vectors.push(vector);
  }
  lastUsedAt = Date.now();
  return vectors;
}

const server = createServer(async (request, response) => {
  response.setHeader("connection", "close");
  if (!authorized(request)) {
    json(response, 401, { error: "unauthorized" });
    return;
  }
  if (request.method === "GET" && request.url === "/health") {
    json(response, 200, {
      ok: true,
      protocol: 1,
      pid: process.pid,
      fingerprint,
      dim: expectedDim,
      modelPath,
      lastUsedAt,
    });
    return;
  }
  if (request.method === "POST" && request.url === "/embed") {
    try {
      const body = await readJson(request);
      const operation = requestTail.then(() => embed(body.role, body.texts));
      requestTail = operation.then(
        () => undefined,
        () => undefined,
      );
      json(response, 200, { vectors: await operation });
    } catch (error) {
      json(response, 400, { error: error instanceof Error ? error.message : String(error) });
    }
    return;
  }
  json(response, 404, { error: "not found" });
});

async function atomicJson(path, value) {
  const temp = `${path}.tmp-${process.pid}`;
  try {
    await writeFile(temp, `${JSON.stringify(value, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600,
      flag: "w",
    });
    await rename(temp, path);
    await chmod(path, 0o600);
  } finally {
    await unlink(temp).catch(() => {});
  }
}

async function shutdown(reason) {
  if (closing) return;
  closing = true;
  console.error(`[surmem-embedding-daemon] stopping: ${reason}`);
  await new Promise((resolve) => server.close(() => resolve()));
  await requestTail.catch(() => {});
  await context?.dispose?.();
  await model?.dispose?.();
  await llama?.dispose?.();
  try {
    const endpoint = JSON.parse(await readFile(endpointPath, "utf8"));
    if (endpoint.pid === process.pid) await unlink(endpointPath);
  } catch {}
  process.exit(0);
}

server.listen(0, "127.0.0.1", async () => {
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("failed to bind loopback TCP endpoint");
  await atomicJson(endpointPath, {
    protocol: 1,
    pid: process.pid,
    host: "127.0.0.1",
    port: address.port,
    fingerprint,
    dim: expectedDim,
    modelPath,
    startedAt: new Date().toISOString(),
  });
  writeState({ phase: "ready", modelPath, host: "127.0.0.1", port: address.port, percent: 100 });
  await unlink(join(daemonDir, "startup.lock")).catch(() => {});
  console.error(`[surmem-embedding-daemon] ready pid=${process.pid} port=${address.port}`);
});

const idleTimer = setInterval(
  () => {
    if (idleMs > 0 && Date.now() - lastUsedAt >= idleMs) void shutdown("idle timeout");
  },
  Math.min(60_000, Math.max(1000, Math.floor(idleMs / 4) || 60_000)),
);

process.on("SIGTERM", () => void shutdown("SIGTERM"));
process.on("SIGINT", () => void shutdown("SIGINT"));
process.on("uncaughtException", (error) => {
  console.error(error);
  clearInterval(idleTimer);
  void shutdown("uncaught exception");
});
process.on("unhandledRejection", (error) => {
  console.error(error);
  clearInterval(idleTimer);
  void shutdown("unhandled rejection");
});
