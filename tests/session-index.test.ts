import { describe, expect, test } from "bun:test";
import { mkdtemp, rm, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SessionIndex } from "../src/session-index";

async function fixture() {
  const dir = await mkdtemp(join(tmpdir(), "surmem-session-index-"));
  const path = join(dir, "nested", "session.jsonl");
  await Bun.write(path, "");
  const lines = `${[
    JSON.stringify({
      type: "session",
      id: "session-1",
      cwd: "/workspace",
      timestamp: "2025-01-01T00:00:00.000Z",
    }),
    JSON.stringify({
      type: "message",
      id: "u1",
      timestamp: "2025-01-01T00:00:01.000Z",
      message: { role: "user", content: [{ type: "text", text: "Find the deployment script." }] },
    }),
    "not json",
    JSON.stringify({
      type: "message",
      id: "a1",
      timestamp: "2025-01-01T00:00:02.000Z",
      message: {
        role: "assistant",
        content: [{ type: "text", text: "The deployment script is in scripts/deploy.sh." }],
      },
    }),
    JSON.stringify({
      type: "message",
      id: "t1",
      message: { role: "tool", content: [{ type: "text", text: "secret tool result" }] },
    }),
    JSON.stringify({ type: "message", id: "s1", message: { role: "system", content: "System guidance" } }),
  ].join("\n")}\n`;
  await Bun.write(path, lines);
  return { dir, path };
}

describe("SessionIndex", () => {
  test("indexes Pi JSONL messages, skips bad rows and ignores tool results", async () => {
    const { dir, path } = await fixture();
    const index = new SessionIndex(join(dir, "db", "sessions.sqlite"));
    try {
      const result = await index.indexFile(path);
      expect(result.indexed).toBe(true);
      expect(result.messages).toBe(3);
      expect((await index.stats()).messages).toBe(3);
      expect((await index.search("deployment")).some((hit) => hit.role === "assistant")).toBe(true);
      expect(await index.search("secret tool result")).toHaveLength(0);
    } finally {
      await index.close();
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("uses size and mtime for an incremental no-op", async () => {
    const { dir, path } = await fixture();
    const index = new SessionIndex(join(dir, "sessions.sqlite"));
    try {
      const first = await index.indexFile(path);
      const second = await index.indexFile(path);
      expect(first.indexed).toBe(true);
      expect(second.skipped).toBe(true);
      expect((await index.stats()).messages).toBe(3);
    } finally {
      await index.close();
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("replaces a changed file transactionally and supports close/reopen", async () => {
    const { dir, path } = await fixture();
    const dbPath = join(dir, "sessions.sqlite");
    const index = new SessionIndex(dbPath);
    try {
      await index.indexFile(path);
      await Bun.write(
        path,
        JSON.stringify({ type: "session", id: "session-1", cwd: "/workspace" }) +
          "\n" +
          JSON.stringify({
            type: "message",
            message: { role: "user", content: "A completely different topic" },
          }) +
          "\n",
      );
      const now = new Date(Date.now() + 2000);
      await utimes(path, now, now);
      expect((await index.indexFile(path)).indexed).toBe(true);
      expect(await index.search("deployment")).toHaveLength(0);
      expect((await index.stats()).messages).toBe(1);
      await index.close();
      const reopened = new SessionIndex(dbPath);
      expect((await reopened.stats()).messages).toBe(1);
      await reopened.close();
    } finally {
      await index.close().catch(() => {});
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("indexes recursive directories newest first with a limit", async () => {
    const dir = await mkdtemp(join(tmpdir(), "surmem-session-directory-"));
    const oldPath = join(dir, "old.jsonl");
    const newPath = join(dir, "sub", "new.jsonl");
    try {
      await Bun.write(
        oldPath,
        JSON.stringify({ type: "session", id: "old" }) +
          "\n" +
          JSON.stringify({ type: "message", message: { role: "user", content: "old message" } }) +
          "\n",
      );
      await Bun.write(
        newPath,
        JSON.stringify({ type: "session", id: "new" }) +
          "\n" +
          JSON.stringify({ type: "message", message: { role: "user", content: "new message" } }) +
          "\n",
      );
      const later = new Date(Date.now() + 5000);
      await utimes(newPath, later, later);
      const index = new SessionIndex(join(dir, "index.sqlite"));
      try {
        const results = await index.indexDirectory(dir, { limit: 1 });
        expect(results).toHaveLength(1);
        expect(results[0].sessionId).toBe("new");
      } finally {
        await index.close();
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("supports short CJK LIKE search, safe limits, cwd filters and truncation", async () => {
    const { dir, path } = await fixture();
    const longPath = join(dir, "cjk.jsonl");
    const longText = `北京${"x".repeat(1400)}`;
    await Bun.write(
      longPath,
      `${[
        JSON.stringify({ type: "session", id: "cjk", cwd: "/china" }),
        JSON.stringify({ type: "message", message: { role: "user", content: longText } }),
      ].join("\n")}\n`,
    );
    const index = new SessionIndex(join(dir, "sessions.sqlite"));
    try {
      await index.indexFile(path);
      await index.indexFile(longPath);
      const hits = await index.search("北京", { limit: 1, cwd: "/china" });
      expect(hits).toHaveLength(1);
      expect(hits[0].sessionId).toBe("cjk");
      expect(hits[0].content.length).toBe(1200);
      await expect(index.search("deployment", { limit: 0 })).resolves.toHaveLength(1);
      await expect(index.search("deployment", { limit: 100 })).resolves.toBeArray();
    } finally {
      await index.close();
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("reports corrupt database errors with the database path", async () => {
    const dir = await mkdtemp(join(tmpdir(), "surmem-session-corrupt-"));
    const dbPath = join(dir, "bad.sqlite");
    await writeFile(dbPath, "this is not sqlite");
    const index = new SessionIndex(dbPath);
    try {
      await expect(index.stats()).rejects.toThrow(dbPath);
    } finally {
      await index.close().catch(() => {});
      await rm(dir, { recursive: true, force: true });
    }
  });
});
