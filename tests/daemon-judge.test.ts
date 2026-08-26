import { describe, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { DaemonMemoryJudge } from "../src/daemon-judge";

async function fakeJudge(dir: string, fingerprint: string, auth: string) {
  let calls = 0;
  const server = createServer(async (request, response) => {
    if (request.headers.authorization !== `Bearer ${auth}`) {
      response.writeHead(401, { connection: "close" });
      response.end(JSON.stringify({ error: "unauthorized" }));
      return;
    }
    if (request.method === "GET") {
      response.writeHead(200, { "content-type": "application/json", connection: "close" });
      response.end(
        JSON.stringify({
          ok: true,
          protocol: 1,
          pid: process.pid,
          fingerprint,
          modelPath: "/fake/qwen.gguf",
          lastUsedAt: Date.now(),
        }),
      );
      return;
    }
    calls++;
    for await (const _chunk of request) {
      // Drain body.
    }
    const output =
      request.url === "/assess"
        ? '<think>ignored</think>\n{"memorable":true,"confidence":0.93,"canonicalText":"用户偏好简洁回答。","scope":"global","reason":"stable preference"}'
        : '{"verdict":"UPDATE","confidence":0.96,"reason":"direct correction"}';
    response.writeHead(200, { "content-type": "application/json", connection: "close" });
    response.end(JSON.stringify({ output }));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("fake judge failed to bind");
  await writeFile(join(dir, "token"), `${auth}\n`, { mode: 0o600 });
  await writeFile(
    join(dir, "endpoint.json"),
    JSON.stringify({
      protocol: 1,
      pid: process.pid,
      host: "127.0.0.1",
      port: address.port,
      fingerprint,
      modelPath: "/fake/qwen.gguf",
    }),
    { mode: 0o600 },
  );
  return {
    get calls() {
      return calls;
    },
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

describe("DaemonMemoryJudge", () => {
  test("one shared local judge handles candidate and contradiction roles", async () => {
    const dir = await mkdtemp(join(tmpdir(), "surmem-judge-client-"));
    const first = new DaemonMemoryJudge({ daemonDir: dir, modelUri: "hf:test/qwen.gguf" });
    const second = new DaemonMemoryJudge({ daemonDir: dir, modelUri: "hf:test/qwen.gguf" });
    const daemon = await fakeJudge(dir, first.fingerprint, "c".repeat(64));
    try {
      expect(await first.assess("我希望回答简洁一些")).toBe("用户偏好简洁回答。");
      const decision = await second.arbitrate("用户现在使用 pnpm。", "用户使用 npm。");
      expect(decision.verdict).toBe("UPDATE");
      expect(decision.confidence).toBe(0.96);
      expect(daemon.calls).toBe(2);
      expect(first.diagnostics().backend).toBe("gguf-daemon");
    } finally {
      first.dispose();
      second.dispose();
      await daemon.close();
      await rm(dir, { recursive: true, force: true });
    }
  });
});
