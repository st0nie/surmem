import { describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
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

  test("/surmem menu manages project and global memories (add/edit/delete/search/status)", async () => {
    const root = await mkdtemp(join(tmpdir(), "surmem-menu-"));
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
    const selectScript: Array<string | ((title: string, options: string[]) => string)> = [
      "Manage project memories",
      "+ Add memory",
      "episodic",
      (_title, options) => {
        const record = options.find((option) => option.startsWith("1. "));
        expect(record).toBeDefined();
        return record as string;
      },
      "View / edit",
      (_title, options) => {
        const record = options.find((option) => option.startsWith("1. "));
        expect(record).toBeDefined();
        return record as string;
      },
      "Delete",
      "← Back",
      "Status details",
      "Manage global memories",
      (_title, options) => {
        // Global scope stays empty: only Add/Search/Back are offered.
        expect(options.some((option) => option.startsWith("1. "))).toBe(false);
        return "← Back";
      },
      "Close",
    ];
    const editorScript: string[] = [
      "The project uses uv for all Python tooling.",
      "The project uses uv for every Python workflow, including CI.",
    ];
    const confirmScript: boolean[] = [true];
    const context = {
      cwd: "/workspace/surmem",
      mode: "tui",
      hasUI: true,
      signal: undefined,
      ui: {
        notify(message: string) {
          notifications.push(message);
        },
        async select(title: string, options: string[]) {
          const step = selectScript.shift();
          if (step === undefined) throw new Error(`Unexpected select dialog: ${title}`);
          const choice = typeof step === "function" ? step(title, options) : step;
          if (!options.includes(choice))
            throw new Error(`Scripted choice "${choice}" not offered by "${title}": ${options.join(" | ")}`);
          return choice;
        },
        async input() {
          return undefined;
        },
        async editor() {
          const next = editorScript.shift();
          if (next === undefined) throw new Error("Unexpected editor dialog");
          return next;
        },
        async confirm() {
          const next = confirmScript.shift();
          if (next === undefined) throw new Error("Unexpected confirm dialog");
          return next;
        },
      },
      sessionManager: {
        getSessionId: () => "menu-test-session",
        getSessionFile: () => join(root, "sessions", "empty-session-not-created.jsonl"),
      },
    };

    try {
      surmemExtension(pi);
      for (const handler of handlers.get("session_start") ?? []) {
        await handler({ reason: "startup" }, context);
      }
      await commands.get("surmem").handler("", context);

      expect(selectScript).toHaveLength(0);
      expect(editorScript).toHaveLength(0);
      expect(confirmScript).toHaveLength(0);

      // Add path reported the gate verdict.
      expect(notifications.some((message) => message.includes("ADD id="))).toBe(true);
      // Edit kept the record and produced a recovery file for the old text.
      expect(notifications.some((message) => message.includes("Updated"))).toBe(true);
      // Delete produced its own recovery file.
      expect(notifications.some((message) => message.includes("Deleted"))).toBe(true);
      // Status details row now does something visible instead of a noop.
      expect(notifications.some((message) => message.includes("embedder: hash"))).toBe(true);

      const recoveryDir = join(root, "data", "recovery");
      const recoveryFiles = (await readdir(recoveryDir)).filter((name) => name.endsWith(".json"));
      expect(recoveryFiles).toHaveLength(2);
      const recoveries = [];
      for (const name of recoveryFiles) {
        recoveries.push(JSON.parse(await readFile(join(recoveryDir, name), "utf8")));
      }
      const texts = recoveries.map((entry) => entry.record.text).sort();
      expect(texts).toEqual([
        "The project uses uv for all Python tooling.",
        "The project uses uv for every Python workflow, including CI.",
      ]);
      // The edit preserved the record id: both recovery files reference it.
      expect(recoveries[0].record.id).toBe(recoveries[1].record.id);
      expect(recoveries.every((entry) => entry.scope === "project")).toBe(true);

      // Deleted from the active store; scopes stay isolated.
      const projectList = await tools
        .get("surmem_list")
        .execute("list-1", { scope: "project", limit: 20 }, undefined, undefined, context);
      expect(projectList.content[0].text).toBe("No memories stored.");
      const globalList = await tools
        .get("surmem_list")
        .execute("list-2", { scope: "global", limit: 20 }, undefined, undefined, context);
      expect(globalList.content[0].text).toBe("No memories stored.");

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
