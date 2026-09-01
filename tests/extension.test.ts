import { Database } from "bun:sqlite";
import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
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

  test("surmem_remember supports explicit supersedes and reports NOOP nearest hints", async () => {
    const root = await mkdtemp(join(tmpdir(), "surmem-supersede-"));
    const previousDir = process.env.SURMEM_DIR;
    const previousSessions = process.env.PI_CODING_AGENT_SESSION_DIR;
    const previousEmbedder = process.env.SURMEM_EMBEDDER;
    const previousJudgeMode = process.env.SURMEM_JUDGE_MODE;
    process.env.SURMEM_DIR = join(root, "data");
    process.env.SURMEM_EMBEDDER = "hash";
    process.env.SURMEM_JUDGE_MODE = "heuristic";
    process.env.PI_CODING_AGENT_SESSION_DIR = join(root, "sessions");
    await mkdir(process.env.PI_CODING_AGENT_SESSION_DIR, { recursive: true });

    const handlers = new Map<string, Handler[]>();
    const tools = new Map<string, any>();
    const pi = {
      on(name: string, handler: Handler) {
        handlers.set(name, [...(handlers.get(name) ?? []), handler]);
      },
      registerTool(tool: any) {
        tools.set(tool.name, tool);
      },
      registerCommand() {},
    } as any;
    const context = {
      cwd: "/workspace/surmem",
      mode: "print",
      hasUI: false,
      signal: undefined,
      ui: { notify() {}, select: async () => undefined, input: async () => undefined },
      sessionManager: {
        getSessionId: () => "supersede-test-session",
        getSessionFile: () => join(root, "sessions", "session.jsonl"),
      },
    };

    try {
      surmemExtension(pi);
      for (const handler of handlers.get("session_start") ?? []) {
        await handler({ reason: "startup" }, context);
      }
      const remember = tools.get("surmem_remember");

      const first = await remember.execute(
        "remember-1",
        { text: "The user just moved from Beijing to Shanghai.", scope: "project" },
        undefined,
        undefined,
        context,
      );
      expect(first.details.verdict).toBe("ADD");

      const noop = await remember.execute(
        "remember-2",
        { text: "The user moved from Beijing back to Shanghai again.", scope: "project" },
        undefined,
        undefined,
        context,
      );
      expect(noop.details.verdict).toBe("NOOP");
      expect(noop.details.nearestId).toBe(first.details.id);
      expect(noop.content[0].text).toContain(`supersedes="${first.details.id}"`);

      const updated = await remember.execute(
        "remember-3",
        {
          text: "The user moved from Beijing back to Shanghai again.",
          scope: "project",
          supersedes: first.details.id,
        },
        undefined,
        undefined,
        context,
      );
      expect(updated.details.verdict).toBe("UPDATE");
      expect(updated.details.supersededId).toBe(first.details.id);

      const crossScope = remember.execute(
        "remember-4",
        {
          text: "The user moved from Beijing back to Shanghai once more.",
          scope: "global",
          supersedes: first.details.id,
        },
        undefined,
        undefined,
        context,
      );
      await expect(crossScope).rejects.toThrow(/was not found/);

      for (const handler of handlers.get("session_shutdown") ?? []) {
        await handler({ reason: "quit" }, context);
      }
    } finally {
      if (previousDir === undefined) delete process.env.SURMEM_DIR;
      else process.env.SURMEM_DIR = previousDir;
      if (previousSessions === undefined) delete process.env.PI_CODING_AGENT_SESSION_DIR;
      else process.env.PI_CODING_AGENT_SESSION_DIR = previousSessions;
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

  test("search results survive viewing a record and support re-search", async () => {
    const root = await mkdtemp(join(tmpdir(), "surmem-search-"));
    const previousDir = process.env.SURMEM_DIR;
    const previousSessions = process.env.PI_CODING_AGENT_SESSION_DIR;
    const previousEmbedder = process.env.SURMEM_EMBEDDER;
    const previousJudgeMode = process.env.SURMEM_JUDGE_MODE;
    process.env.SURMEM_DIR = join(root, "data");
    process.env.SURMEM_EMBEDDER = "hash";
    process.env.SURMEM_JUDGE_MODE = "heuristic";
    process.env.PI_CODING_AGENT_SESSION_DIR = join(root, "sessions");
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

    const PYTHON_TEXT = "The project uses uv for every Python workflow including CI.";
    const BUN_TEXT = "The project team decided to adopt bun for JavaScript tooling.";

    const notifications: string[] = [];
    const selectScript: Array<string | ((title: string, options: string[]) => string)> = [
      "Manage project memories",
      "? Search memories",
      (title, options) => {
        expect(title).toContain(`match(es) for "Python workflow"`);
        const first = options.find((option) => option.startsWith("1. "));
        expect(first).toBeDefined();
        return first as string;
      },
      "View / edit",
      (title, options) => {
        // Back from the record view must land on the same result list,
        // not reset to the unfiltered scope list.
        expect(title).toContain(`match(es) for "Python workflow"`);
        const second = options.find((option) => option.startsWith("2. "));
        expect(second).toBeDefined();
        return second as string;
      },
      "Delete",
      (title, _options) => {
        // The refresh after deletion reflects that only one record remains.
        expect(title).toContain(`1 match(es) for "Python workflow"`);
        return "? Search memories";
      },
      (title, options) => {
        expect(title).toContain(`match(es) for "uv python"`);
        const first = options.find((option) => option.startsWith("1. "));
        expect(first).toBeDefined();
        return first as string;
      },
      "Back",
      (title, _options) => {
        expect(title).toContain(`1 match(es) for "uv python"`);
        return "← Back";
      },
      "← Back",
      "Close",
    ];
    const inputScript = ["Python workflow", "uv python"];
    const confirmScript = [true];
    const editorScript = [PYTHON_TEXT];
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
          const next = inputScript.shift();
          if (next === undefined) throw new Error("Unexpected input dialog");
          return next;
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
        getSessionId: () => "search-test-session",
        getSessionFile: () => join(root, "sessions", "empty-session-not-created.jsonl"),
      },
    };

    try {
      surmemExtension(pi);
      for (const handler of handlers.get("session_start") ?? []) {
        await handler({ reason: "startup" }, context);
      }
      for (const [index, text] of [PYTHON_TEXT, BUN_TEXT].entries()) {
        const remembered = await tools
          .get("surmem_remember")
          .execute(`remember-${index}`, { text, scope: "project" }, undefined, undefined, context);
        expect(remembered.details.verdict).toBe("ADD");
      }

      await commands.get("surmem").handler("", context);

      expect(selectScript).toHaveLength(0);
      expect(inputScript).toHaveLength(0);
      expect(confirmScript).toHaveLength(0);
      expect(editorScript).toHaveLength(0);
      expect(notifications.some((message) => message.includes("Memory unchanged."))).toBe(true);
      expect(notifications.some((message) => message.includes("Deleted"))).toBe(true);

      for (const handler of handlers.get("session_shutdown") ?? []) {
        await handler({ reason: "quit" }, context);
      }

      // Returning to the result list re-runs recall without reinforcement, so
      // the surviving record only counts the two real searches that matched it.
      const projectsDir = join(root, "data", "projects");
      const dbFile = (await readdir(projectsDir)).find((name) => name.endsWith(".sqlite"));
      expect(dbFile).toBeDefined();
      const db = new Database(join(projectsDir, dbFile as string), { readonly: true });
      try {
        const rows = db.query("SELECT payload FROM memories").all() as Array<{ payload: string }>;
        const payloads = rows.map((row) => JSON.parse(row.payload) as { text: string; accessCount: number });
        const survivor = payloads.find((payload) => payload.text === PYTHON_TEXT);
        expect(survivor).toBeDefined();
        expect(survivor?.accessCount).toBe(2);
      } finally {
        db.close();
      }
    } finally {
      if (previousDir === undefined) delete process.env.SURMEM_DIR;
      else process.env.SURMEM_DIR = previousDir;
      if (previousSessions === undefined) delete process.env.PI_CODING_AGENT_SESSION_DIR;
      else process.env.PI_CODING_AGENT_SESSION_DIR = previousSessions;
      if (previousEmbedder === undefined) delete process.env.SURMEM_EMBEDDER;
      else process.env.SURMEM_EMBEDDER = previousEmbedder;
      if (previousJudgeMode === undefined) delete process.env.SURMEM_JUDGE_MODE;
      else process.env.SURMEM_JUDGE_MODE = previousJudgeMode;
      await rm(root, { recursive: true, force: true });
    }
  });

  test("skill and backing memory record are deleted and restored together", async () => {
    const root = await mkdtemp(join(tmpdir(), "surmem-skill-"));
    const previousDir = process.env.SURMEM_DIR;
    const previousSessions = process.env.PI_CODING_AGENT_SESSION_DIR;
    const previousEmbedder = process.env.SURMEM_EMBEDDER;
    const previousJudgeMode = process.env.SURMEM_JUDGE_MODE;
    process.env.SURMEM_DIR = join(root, "data");
    process.env.SURMEM_EMBEDDER = "hash";
    process.env.SURMEM_JUDGE_MODE = "heuristic";
    process.env.PI_CODING_AGENT_SESSION_DIR = join(root, "sessions");
    await mkdir(process.env.PI_CODING_AGENT_SESSION_DIR, { recursive: true });

    const handlers = new Map<string, Handler[]>();
    const tools = new Map<string, any>();
    const pi = {
      on(name: string, handler: Handler) {
        handlers.set(name, [...(handlers.get(name) ?? []), handler]);
      },
      registerTool(tool: any) {
        tools.set(tool.name, tool);
      },
      registerCommand() {},
    } as any;
    const context = {
      cwd: "/workspace/surmem",
      mode: "print",
      hasUI: false,
      signal: undefined,
      ui: { notify() {}, select: async () => undefined, input: async () => undefined },
      sessionManager: {
        getSessionId: () => "skill-test-session",
        getSessionFile: () => join(root, "sessions", "session.jsonl"),
      },
    };
    const skillFile = join(root, "data", "skills", "global", "test-skill", "SKILL.md");
    const findSkillRecordId = async (): Promise<string | null> => {
      const listed = await tools
        .get("surmem_list")
        .execute("list", { scope: "global", limit: 50 }, undefined, undefined, context);
      const text = listed.content[0].text as string;
      const match = text.match(/^- id=(\S+) .*\n {2}test-skill:/m);
      return match ? match[1] : null;
    };

    try {
      surmemExtension(pi);
      for (const handler of handlers.get("session_start") ?? []) {
        await handler({ reason: "startup" }, context);
      }

      const created = await tools.get("surmem_skill").execute(
        "skill-create",
        {
          action: "create",
          scope: "global",
          name: "test-skill",
          description: "Deterministic test skill procedure.",
          steps: ["do the thing"],
          verification: ["verify the thing"],
        },
        undefined,
        undefined,
        context,
      );
      expect(created.content[0].text).toContain("Created Pi skill test-skill");
      expect(existsSync(skillFile)).toBe(true);
      const recordId = await findSkillRecordId();
      expect(recordId).not.toBeNull();

      // surmem_skill delete must tombstone the backing record and leave a
      // recovery file carrying the SKILL.md body.
      const deleted = await tools
        .get("surmem_skill")
        .execute(
          "skill-delete",
          { action: "delete", scope: "global", name: "test-skill" },
          undefined,
          undefined,
          context,
        );
      expect(deleted.details.recoveryId).toBeString();
      expect(existsSync(skillFile)).toBe(false);
      expect(await findSkillRecordId()).toBeNull();

      // Restore recreates both the record and the on-disk skill files.
      const restored = await tools
        .get("surmem_restore")
        .execute("restore-1", { recoveryId: deleted.details.recoveryId }, undefined, undefined, context);
      expect(restored.details.restored).toBe(true);
      expect(restored.content[0].text).toContain("Skill files recreated");
      expect(existsSync(skillFile)).toBe(true);
      expect(await findSkillRecordId()).toBe(recordId);

      // surmem_forget on a skill-backed record removes the files too.
      const forgot = await tools
        .get("surmem_forget")
        .execute("forget-1", { id: recordId, scope: "global" }, undefined, undefined, context);
      expect(forgot.content[0].text).toContain("Skill files removed");
      expect(existsSync(skillFile)).toBe(false);
      expect(await findSkillRecordId()).toBeNull();

      const restoredAgain = await tools
        .get("surmem_restore")
        .execute("restore-2", { recoveryId: forgot.details.recoveryId }, undefined, undefined, context);
      expect(restoredAgain.details.restored).toBe(true);
      expect(existsSync(skillFile)).toBe(true);
      expect(await findSkillRecordId()).toBe(recordId);
    } finally {
      if (previousDir === undefined) delete process.env.SURMEM_DIR;
      else process.env.SURMEM_DIR = previousDir;
      if (previousSessions === undefined) delete process.env.PI_CODING_AGENT_SESSION_DIR;
      else process.env.PI_CODING_AGENT_SESSION_DIR = previousSessions;
      if (previousEmbedder === undefined) delete process.env.SURMEM_EMBEDDER;
      else process.env.SURMEM_EMBEDDER = previousEmbedder;
      if (previousJudgeMode === undefined) delete process.env.SURMEM_JUDGE_MODE;
      else process.env.SURMEM_JUDGE_MODE = previousJudgeMode;
      await rm(root, { recursive: true, force: true });
    }
  });
});
