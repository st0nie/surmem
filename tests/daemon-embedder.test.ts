import { describe, expect, test } from "bun:test";
import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { DaemonGgufEmbedder, EmbeddingDaemonMismatchError } from "../src/daemon-embedder";

async function fakeDaemon(options: { dir: string; fingerprint: string; dim: number; token: string }) {
  let embedRequests = 0;
  const server = createServer(async (request, response) => {
    const authorized = request.headers.authorization === `Bearer ${options.token}`;
    if (!authorized) {
      response.writeHead(401, { connection: "close" });
      response.end(JSON.stringify({ error: "unauthorized" }));
      return;
    }
    if (request.method === "GET" && request.url === "/health") {
      response.writeHead(200, { "content-type": "application/json", connection: "close" });
      response.end(
        JSON.stringify({
          ok: true,
          protocol: 1,
          pid: process.pid,
          fingerprint: options.fingerprint,
          dim: options.dim,
          modelPath: "/fake/model.gguf",
          lastUsedAt: Date.now(),
        }),
      );
      return;
    }
    if (request.method === "POST" && request.url === "/embed") {
      embedRequests++;
      const chunks: Buffer[] = [];
      for await (const chunk of request) chunks.push(chunk as Buffer);
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as { texts: string[] };
      response.writeHead(200, { "content-type": "application/json", connection: "close" });
      response.end(
        JSON.stringify({
          vectors: body.texts.map(() =>
            new Array<number>(options.dim).fill(0).map((_, index) => (index === 0 ? 1 : 0)),
          ),
        }),
      );
      return;
    }
    response.writeHead(404, { connection: "close" });
    response.end();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("fake daemon failed to bind");
  await writeFile(join(options.dir, "token"), `${options.token}\n`, { mode: 0o600 });
  await chmod(join(options.dir, "token"), 0o600);
  await writeFile(
    join(options.dir, "endpoint.json"),
    JSON.stringify({
      protocol: 1,
      pid: process.pid,
      host: "127.0.0.1",
      port: address.port,
      fingerprint: options.fingerprint,
      dim: options.dim,
      modelPath: "/fake/model.gguf",
    }),
    { mode: 0o600 },
  );
  return {
    get embedRequests() {
      return embedRequests;
    },
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

describe("DaemonGgufEmbedder", () => {
  test("multiple clients reuse one authenticated daemon endpoint", async () => {
    const dir = await mkdtemp(join(tmpdir(), "surmem-daemon-client-"));
    const options = { daemonDir: dir, modelUri: "hf:test/model.gguf", dim: 4 };
    const first = DaemonGgufEmbedder.createPair(options);
    const second = DaemonGgufEmbedder.createPair(options);
    const daemon = await fakeDaemon({
      dir,
      fingerprint: first.document.fingerprint,
      dim: 4,
      token: "a".repeat(64),
    });
    try {
      const [left, right] = await Promise.all([
        first.document.embed(["document one"]),
        second.query.embed(["query two"]),
      ]);
      expect(left[0]).toEqual([1, 0, 0, 0]);
      expect(right[0]).toEqual([1, 0, 0, 0]);
      expect(daemon.embedRequests).toBe(2);
      expect((await first.document.status())?.pid).toBe(process.pid);
    } finally {
      first.document.dispose();
      first.query.dispose();
      second.document.dispose();
      second.query.dispose();
      await daemon.close();
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("refuses to mix a second model into the singleton daemon", async () => {
    const dir = await mkdtemp(join(tmpdir(), "surmem-daemon-mismatch-"));
    const serving = new DaemonGgufEmbedder({ daemonDir: dir, modelUri: "hf:test/model-a.gguf", dim: 4 });
    const requesting = new DaemonGgufEmbedder({ daemonDir: dir, modelUri: "hf:test/model-b.gguf", dim: 4 });
    const daemon = await fakeDaemon({ dir, fingerprint: serving.fingerprint, dim: 4, token: "b".repeat(64) });
    try {
      await expect(requesting.embed(["query"])).rejects.toBeInstanceOf(EmbeddingDaemonMismatchError);
    } finally {
      await daemon.close();
      await rm(dir, { recursive: true, force: true });
    }
  });
});
