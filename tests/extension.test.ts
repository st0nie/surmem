import { describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import surmemExtension from "../extensions/surmem/index";
import { HashEmbedder, Kind } from "../src/index";

type Handler = (event: any, context: any) => Promise<any> | any;

describe("Pi extension integration", () => {
  test("registers production tools and completes remember/recall/recovery lifecycle", async () => {
    const root = await mkdtemp(join(tmpdir(), "surmem-extension-"));
    const previousDir = process.env.SURMEM_DIR;
    const previousSessions = process.env.PI_CODING_AGENT_SESSION_DIR;
    const previousLegacyStore = process.env.SURMEM_STORE_PATH;
    const previousEmbedder = process.env.SURMEM_EMBEDDER;
    const previousJudgeMode = process.env.SURMEM_JUDGE_MODE;
    process.env.SURMEM_DIR = join(root, "data");
    process.env.SURMEM_EMBEDDER = "hash";
    process.env.SURMEM_JUDGE_MODE = "heuristic";
    process.env.PI_CODING_AGENT_SESSION_DIR = join(root, "sessions");
    process.env.SURMEM_STORE_PATH = join(root, "legacy-memory.json");
    await mkdir(process.env.PI_CODING_AGENT_SESSION_DIR, { recursive: true });
    const [legacyVector] = new HashEmbedder().embed(["Legacy memory says this repository uses Bun."]);
    await writeFile(
      process.env.SURMEM_STORE_PATH,
      JSON.stringify([
        {
          id: "legacy-extension-id",
          text: "Legacy memory says this repository uses Bun.",
          vector: legacyVector,
          kind: Kind.EPISODIC,
          createdAt: 1,
          lastAccessed: 1,
          baseStrength: 1,
          accessCount: 0,
          surpriseAtWrite: 1,
          supersededBy: null,
          sourceIds: [],
          metadata: {},
        },
      ]),
      "utf8",
    );

    const handlers = new Map<string, Handler[]>();
    const tools = new Map<string, any>();
    const commands = new Map<string, any>();
    const pi = {
      on(name: string, handler: Handler) {
        handlers.set(name, [...(handlers.get(name) ?? []), handler]);
      },
      registerTool(tool: any) {
        tools.set(tool.name, tool);
      },
      registerCommand(name: string, command: any) {
        commands.set(name, command);
      },
    } as any;
    const notifications: string[] = [];
    const context = {
      cwd: "/workspace/surmem",
      mode: "print",
      hasUI: false,
      signal: undefined,
      ui: {
        notify(message: string) {
          notifications.push(message);
        },
        select: async () => undefined,
        input: async () => undefined,
      },
      sessionManager: {
        getSessionId: () => "extension-test-session",
        getSessionFile: () => join(root, "sessions", "empty-session-not-created.jsonl"),
      },
    };

    try {
      surmemExtension(pi);
      expect(tools.size).toBeGreaterThanOrEqual(10);
      expect(commands.has("surmem")).toBe(true);
      for (const handler of handlers.get("session_start") ?? []) {
        await handler({ reason: "startup" }, context);
      }
      for (const handler of handlers.get("resources_discover") ?? []) {
        const discovered = await handler({ cwd: context.cwd, reason: "startup" }, context);
        expect(discovered.skillPaths).toEqual([]);
      }

      const remember = await tools
        .get("surmem_remember")
        .execute(
          "remember-1",
          { text: "The project always uses pnpm for package management.", scope: "project" },
          undefined,
          undefined,
          context,
        );
      expect(remember.details.verdict).toBe("ADD");
      const id = remember.details.id as string;

      const recall = await tools
        .get("surmem_recall")
        .execute("recall-1", { query: "package manager", scope: "all", k: 5 }, undefined, undefined, context);
      expect(recall.content[0].text).toContain("pnpm");
      expect(recall.details.ids).toContain(id);

      const forgot = await tools
        .get("surmem_forget")
        .execute("forget-1", { id, scope: "project" }, undefined, undefined, context);
      expect(forgot.details.recoveryId).toBeString();

      const restored = await tools
        .get("surmem_restore")
        .execute("restore-1", { recoveryId: forgot.details.recoveryId }, undefined, undefined, context);
      expect(restored.details.restored).toBe(true);
      expect(restored.details.id).toBe(id);

      const status = await tools.get("surmem_status").execute("status-1", {}, undefined, undefined, context);
      expect(status.details.project.stats.active).toBe(2);
      expect(status.details.embedder).toContain("hash");

      for (const handler of handlers.get("message_end") ?? []) {
        await handler(
          {
            message: {
              role: "user",
              content: [{ type: "text", text: "Please remember that I always prefer concise answers." }],
            },
          },
          context,
        );
      }
      const originalMessages = [{ role: "user", content: [{ type: "text", text: "hello" }] }];
      const contextResults = [];
      for (const handler of handlers.get("context") ?? []) {
        contextResults.push(await handler({ messages: originalMessages }, context));
      }
      const transformed = contextResults.find((result) => result?.messages)?.messages;
      expect(transformed).toHaveLength(2);
      expect(originalMessages).toHaveLength(1);
      expect(transformed[1].content[0].text).toContain("not yet stored");

      const warnings: string[] = [];
      const originalWarn = console.warn;
      console.warn = (...values: unknown[]) => warnings.push(values.map(String).join(" "));
      try {
        for (const handler of handlers.get("session_shutdown") ?? []) {
          await handler({ reason: "quit" }, context);
        }
      } finally {
        console.warn = originalWarn;
      }
      expect(warnings).toEqual([]);
      expect(notifications.some((message) => message.includes("SurMem ready"))).toBe(true);
    } finally {
      if (previousDir === undefined) delete process.env.SURMEM_DIR;
      else process.env.SURMEM_DIR = previousDir;
      if (previousSessions === undefined) delete process.env.PI_CODING_AGENT_SESSION_DIR;
      else process.env.PI_CODING_AGENT_SESSION_DIR = previousSessions;
      if (previousLegacyStore === undefined) delete process.env.SURMEM_STORE_PATH;
      else process.env.SURMEM_STORE_PATH = previousLegacyStore;
      if (previousEmbedder === undefined) delete process.env.SURMEM_EMBEDDER;
      else process.env.SURMEM_EMBEDDER = previousEmbedder;
      if (previousJudgeMode === undefined) delete process.env.SURMEM_JUDGE_MODE;
      else process.env.SURMEM_JUDGE_MODE = previousJudgeMode;
      await rm(root, { recursive: true, force: true });
    }
  });
});
