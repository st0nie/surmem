import { expect, test } from "bun:test";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { HashEmbedder, SqlitePersister, SurpriseMemory } from "../src/index";

const cli = resolve("src/mcp/cli.ts");

for (const termination of ["EOF", "SIGTERM", "SIGINT"] as const) {
  test(`stdio ${termination} closes stores and exits cleanly after durable writes`, async () => {
    const root = await mkdtemp(join(tmpdir(), "surmem-mcp-exit-"));
    const storage = join(root, "storage");
    const child = spawn(process.execPath, [cli, "--project", root, "--storage-dir", storage], {
      env: { PATH: process.env.PATH, SURMEM_EMBEDDER: "hash", SURMEM_JUDGE_MODE: "heuristic" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    const client = new Client({ name: "lifecycle-test", version: "1" });
    // SDK framing over our own subprocess streams exposes the exact OS exit event.
    const transport = new StdioServerTransport(child.stdout, child.stdin);
    try {
      await client.connect(transport, { timeout: 5000 });
      const response = await client.callTool(
        {
          name: "surmem_remember",
          arguments: { text: "Global shutdown durability requires saved memory records", scope: "global" },
        },
        undefined,
        { timeout: 5000 },
      );
      expect(response.isError).not.toBe(true);
      const exited = once(child, "exit", { signal: AbortSignal.timeout(5000) });
      if (termination === "EOF") child.stdin.end();
      else child.kill(termination);
      expect(await exited).toEqual([0, null]);
      expect(stderr).toBe("");
      const path = join(storage, "global.sqlite");
      const wal = await stat(`${path}-wal`).catch((error: NodeJS.ErrnoException) => {
        if (error.code === "ENOENT") return null;
        throw error;
      });
      expect(wal?.size ?? 0).toBe(0);
      const memory = new SurpriseMemory({
        embedder: new HashEmbedder(),
        store: { persister: new SqlitePersister(path) },
      });
      try {
        await memory.load();
        expect(memory.stats.active).toBe(1);
      } finally {
        await memory.close();
      }
    } finally {
      if (child.exitCode === null && child.signalCode === null) {
        const exited = once(child, "exit", { signal: AbortSignal.timeout(5000) });
        child.kill("SIGKILL");
        await exited;
      }
      await client.close();
      await rm(root, { recursive: true, force: true });
    }
  });
}

test("packaged CLI has a Bun shebang and invalid arguments never emit stdout", async () => {
  expect((await readFile(cli, "utf8")).split("\n")[0]).toBe("#!/usr/bin/env bun");
  const child = Bun.spawn([process.execPath, cli, "--project", "."], { stdout: "pipe", stderr: "pipe" });
  const [code, stdout, stderr] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ]);
  expect(code).toBe(1);
  expect(stdout).toBe("");
  expect(stderr.length).toBeGreaterThan(0);
});
