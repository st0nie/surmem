/**
 * SurMem pi extension — surprise-gated long-term memory for the pi coding agent.
 *
 * Pipeline (fully automatic):
 *   message_end (user) -> cheap prefilter -> local GGUF memorability judge
 *   -> if memorable: a <system-reminder name="surmem"> steering message nudges
 *      the main agent to store it via `surmem_remember`
 *   -> surmem_remember runs the surprise gate (ADD / UPDATE / REINFORCE / NOOP)
 *      so dedup and contradiction handling stay centralized
 *
 * Also:
 *   - KV-cache-stable snapshot of the strongest memories in the system prompt
 *   - `surmem_recall` tool for prompt-dependent search
 *   - `/surmem` settings-style panel (status + live-tunable parameters,
 *     persisted globally to ~/.pi/agent/surmem.json)
 *   - consolidation (episodic -> semantic) + persistence on session shutdown
 *
 * Load it with:
 *   pi -e ./extensions/surmem/index.ts
 * or register the absolute path in the "extensions" list of
 * ~/.pi/agent/settings.json for permanent auto-loading.
 *
 * Configuration (env):
 *   SURMEM_GGUF_MODEL_PATH     - local embedding model (default: qmd-cached
 *                                embeddinggemma-300M if present, else HashEmbedder)
 *   SURMEM_GGUF_DIM            - embedding dim (default 768; 1024 for Qwen3-Embedding-0.6B)
 *   SURMEM_JUDGE_GGUF          - local judge model (default: qmd-cached
 *                                qmd-query-expansion-1.7B if present, else no judge)
 *   SURMEM_EMBEDDING_API_KEY   - OpenAI-compatible embedder (used when no GGUF path)
 *   SURMEM_EMBEDDING_BASE_URL  - default https://api.openai.com/v1
 *   SURMEM_EMBEDDING_MODEL     - default text-embedding-3-small
 *   SURMEM_EMBEDDING_DIM       - default 1536
 *   SURMEM_STORE_PATH          - default <cwd>/.pi/surmem/memory.json
 *   SURMEM_CONFIG_PATH         - default ~/.pi/agent/surmem.json
 */

import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { CONFIG_DIR_NAME, getSettingsListTheme } from "@earendil-works/pi-coding-agent";
import {
  Container,
  Key,
  matchesKey,
  SettingsList,
  Spacer,
  Text,
  type SettingItem,
} from "@earendil-works/pi-tui";
import { Type } from "typebox";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

import {
  GgufEmbedder,
  GgufMemorabilityJudge,
  HashEmbedder,
  OpenAIEmbedder,
  SurpriseMemory,
  WriteVerdict,
  type Embedder,
  type MemorabilityJudge,
} from "../../src/index";

const SNAPSHOT_SIZE = 8;
const OBSERVE_MAX_CHARS = 2000;

/** Tunable parameters, persisted globally to ~/.pi/agent/surmem.json. */
interface ExtConfig {
  tauAdd?: number; // novelty threshold for ADD (default 0.45)
  dupSim?: number; // similarity threshold for REINFORCE (default 0.85)
  conflictSim?: number; // similarity threshold for the UPDATE zone (default 0.55)
  minTokens?: number; // observations shorter than this are discarded (default 3)
  decayRatePerHour?: number; // episodic decay rate (default 0.02)
  semanticDecayRatePerHour?: number; // semantic decay rate (default 0.002)
  forgetThreshold?: number; // strength below this is forgotten (default 0.1)
  snapshotSize?: number; // memories injected into the system prompt (default 8)
}

interface ConfigRow {
  key: keyof ExtConfig;
  desc: string;
}

const CONFIG_ROWS: ConfigRow[] = [
  { key: "tauAdd", desc: "novelty above this -> ADD" },
  { key: "dupSim", desc: "similarity above this -> REINFORCE" },
  { key: "conflictSim", desc: "similarity above this -> UPDATE zone" },
  { key: "minTokens", desc: "shorter observations -> NOOP" },
  { key: "decayRatePerHour", desc: "episodic decay rate" },
  { key: "semanticDecayRatePerHour", desc: "semantic decay rate" },
  { key: "forgetThreshold", desc: "strength below this -> forgotten" },
  { key: "snapshotSize", desc: "memories injected into system prompt" },
];

interface EmbedderPair {
  embedder: Embedder;
  queryEmbedder?: Embedder;
  name: string;
}

function defaultQmdModel(file: string): string | undefined {
  const p = join(homedir(), ".cache/qmd/models", file);
  return existsSync(p) ? p : undefined;
}

/**
 * Embedder precedence:
 *   1. SURMEM_GGUF_MODEL_PATH, or qmd-cached embeddinggemma if present
 *   2. SURMEM_EMBEDDING_API_KEY (OpenAI-compatible endpoint)
 *   3. HashEmbedder fallback (offline, lower quality)
 */
function embeddersFromEnv(): EmbedderPair {
  const ggufPath =
    process.env.SURMEM_GGUF_MODEL_PATH ??
    defaultQmdModel("hf_ggml-org_embeddinggemma-300M-Q8_0.gguf");
  if (ggufPath) {
    const dim = Number(process.env.SURMEM_GGUF_DIM ?? 768);
    const pair = GgufEmbedder.createPair({ modelPath: ggufPath, dim });
    return { embedder: pair.document, queryEmbedder: pair.query, name: `gguf:${ggufPath.split("/").pop()}` };
  }
  const apiKey = process.env.SURMEM_EMBEDDING_API_KEY;
  if (apiKey) {
    const embedder = new OpenAIEmbedder({
      apiKey,
      baseUrl: process.env.SURMEM_EMBEDDING_BASE_URL,
      model: process.env.SURMEM_EMBEDDING_MODEL,
      dim: process.env.SURMEM_EMBEDDING_DIM
        ? Number(process.env.SURMEM_EMBEDDING_DIM)
        : undefined,
    });
    return { embedder, name: `openai:${process.env.SURMEM_EMBEDDING_MODEL ?? "text-embedding-3-small"}` };
  }
  return { embedder: new HashEmbedder(), name: "hash (offline fallback)" };
}

/** Judge: always the local qmd-query-expansion GGUF (zero API cost). */
function judgeFromEnv(): { judge: MemorabilityJudge | null; name: string } {
  const judgePath =
    process.env.SURMEM_JUDGE_GGUF ??
    defaultQmdModel("hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf");
  if (judgePath) {
    return {
      judge: new GgufMemorabilityJudge({ modelPath: judgePath }),
      name: `gguf:${judgePath.split("/").pop()}`,
    };
  }
  return { judge: null, name: "unavailable (no local judge model found)" };
}

function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter(
        (p): p is { type: string; text: string } =>
          typeof p === "object" && p !== null && (p as { type?: string }).type === "text",
      )
      .map((p) => p.text)
      .join("\n");
  }
  return "";
}

/** Cheap prefilter before spending a judge call. */
function worthJudging(text: string): boolean {
  const t = text.trim();
  if (t.length < 12 || t.length > OBSERVE_MAX_CHARS) return false;
  if (t.startsWith("/")) return false; // commands
  if (t.startsWith("<system-reminder")) return false; // our own reminders (loop guard)
  if (t.startsWith("```")) return false; // pure code blocks
  return true;
}

export default function (pi: ExtensionAPI) {
  let mem: SurpriseMemory | null = null;
  let judge: MemorabilityJudge | null = null;
  let snapshot = "";
  let configPath = "";
  let snapshotSize = SNAPSHOT_SIZE;
  let embedderName = "";
  let judgeName = "";

  function loadConfig(): ExtConfig {
    try {
      return JSON.parse(readFileSync(configPath, "utf8")) as ExtConfig;
    } catch {
      return {};
    }
  }

  function applyConfig(cfg: ExtConfig): void {
    if (!mem) return;
    mem.configure({
      gate: {
        tauAdd: cfg.tauAdd,
        dupSim: cfg.dupSim,
        conflictSim: cfg.conflictSim,
        minTokens: cfg.minTokens,
      },
      store: {
        decayRatePerHour: cfg.decayRatePerHour,
        semanticDecayRatePerHour: cfg.semanticDecayRatePerHour,
        forgetThreshold: cfg.forgetThreshold,
      },
    });
    if (cfg.snapshotSize !== undefined) snapshotSize = cfg.snapshotSize;
  }

  function persistConfig(cfg: ExtConfig): void {
    mkdirSync(dirname(configPath), { recursive: true });
    writeFileSync(configPath, JSON.stringify(cfg, null, 2), "utf8");
  }

  /** Top-N strongest memory bullets, prompt-independent for KV-cache stability. */
  function buildSnapshot(): string {
    if (!mem) return "";
    const top = mem.store
      .active()
      .map((r) => ({ r, s: mem!.store.effectiveStrength(r) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, snapshotSize);
    if (top.length === 0) return "";
    return top.map(({ r }) => `- [${r.kind}] ${r.text}`).join("\n");
  }

  pi.on("session_start", async (_event, ctx) => {
    const persistPath =
      process.env.SURMEM_STORE_PATH ??
      join(ctx.cwd, CONFIG_DIR_NAME, "surmem", "memory.json");
    configPath =
      process.env.SURMEM_CONFIG_PATH ??
      join(homedir(), CONFIG_DIR_NAME, "agent", "surmem.json");
    const embedders = embeddersFromEnv();
    embedderName = embedders.name;
    const j = judgeFromEnv();
    judge = j.judge;
    judgeName = j.name;
    mem = new SurpriseMemory({
      embedder: embedders.embedder,
      queryEmbedder: embedders.queryEmbedder,
      gate: { tauAdd: 0.45, dupSim: 0.85, conflictSim: 0.55 },
      consolidation: { clusterSim: 0.3 },
      store: { persistPath },
    });
    try {
      await mem.load();
    } catch {
      // Corrupt store: start empty rather than crash the session.
    }
    applyConfig(loadConfig());
    snapshot = buildSnapshot();
  });

  // Always inject the SurMem section: instruction linkage gives the
  // <system-reminder> tags their trust level; memories ride along when present.
  pi.on("before_agent_start", async (event) => {
    const instructions = [
      "## Long-term memory (SurMem)",
      "",
      "You may receive <system-reminder name=\"surmem\"> tags containing candidate " +
        "facts detected in the conversation. When one is a durable fact, preference, " +
        "or decision, store it with the `surmem_remember` tool; otherwise ignore the " +
        "reminder. Use `surmem_recall` to search long-term memory for a topic.",
    ];
    const section = snapshot
      ? instructions.join("\n") + "\n\n" + snapshot
      : instructions.join("\n");
    return { systemPrompt: event.systemPrompt + "\n\n" + section };
  });

  // Fully automatic observation: judge every user message locally, then nudge
  // the main agent via a steering reminder instead of writing silently.
  pi.on("message_end", async (event, ctx) => {
    if (!mem || !judge || event.message.role !== "user") return;
    const text = extractText(
      (event.message as { content?: unknown }).content,
    );
    if (!text || !worthJudging(text)) return;
    try {
      const fact = await judge.assess(text.slice(0, OBSERVE_MAX_CHARS));
      if (!fact) return;
      pi.sendUserMessage(
        `<system-reminder name="surmem">\n` +
          `Potential memory detected: "${fact}"\n` +
          `If this is a durable fact, preference, or decision worth remembering ` +
          `across sessions, store it with the surmem_remember tool. Otherwise ` +
          `ignore this reminder.\n` +
          `</system-reminder>`,
        { deliverAs: "steer" },
      );
    } catch {
      // Memory pipeline must never break the agent loop.
    }
  });

  pi.registerTool({
    name: "surmem_remember",
    label: "Remember",
    description:
      "Store an important fact, user preference, or decision in long-term memory. " +
      "The surprise gate decides whether to add, update, reinforce, or discard it.",
    parameters: Type.Object({
      text: Type.String({ description: "The fact to remember, as one self-contained sentence" }),
    }),
    async execute(_toolCallId, params) {
      if (!mem) {
        return {
          content: [{ type: "text" as const, text: "Memory not initialized." }],
          details: { verdict: "NOOP" as WriteVerdict, surprise: 0 },
        };
      }
      const r = await mem.observe(params.text, { origin: "explicit-tool" });
      if (r.verdict === WriteVerdict.ADD || r.verdict === WriteVerdict.UPDATE) {
        snapshot = buildSnapshot();
      }
      return {
        content: [
          {
            type: "text" as const,
            text:
              `${r.verdict} (surprise=${r.surprise.toFixed(2)})` +
              (r.superseded ? ` — superseded: "${r.superseded.text}"` : ""),
          },
        ],
        details: { verdict: r.verdict, surprise: r.surprise },
      };
    },
  });

  pi.registerTool({
    name: "surmem_recall",
    label: "Recall",
    description:
      "Search long-term memory for facts relevant to a query. Returns the top-k " +
      "memories ranked by relevance, recency, and strength.",
    parameters: Type.Object({
      query: Type.String({ description: "What to search memory for" }),
      k: Type.Optional(Type.Number({ description: "Max results (default 5)" })),
    }),
    async execute(_toolCallId, params) {
      if (!mem) {
        return {
          content: [{ type: "text" as const, text: "Memory not initialized." }],
          details: { count: 0 },
        };
      }
      const hits = await mem.recall(params.query, params.k ?? 5);
      const text = hits.length
        ? hits
            .map(
              ({ record, score }) =>
                `- [${record.kind} | score=${score.toFixed(2)}] ${record.text}`,
            )
            .join("\n")
        : "No relevant memories found.";
      return {
        content: [{ type: "text" as const, text }],
        details: { count: hits.length },
      };
    },
  });

  pi.registerCommand("surmem", {
    description:
      "Open the SurMem settings panel (status + parameters, " +
      "persisted to ~/.pi/agent/surmem.json, applied live)",
    handler: async (_args, ctx) => {
      if (!mem) {
        ctx.ui.notify("SurMem not initialized", "warning");
        return;
      }
      await showSurmemSettings(ctx);
    },
  });

  /** Settings UI: pi-tui's SettingsList (same component as pi's /settings).
   *  Enter on a row closes the overlay and opens pi's native input dialog;
   *  valid input is applied live, persisted, then the panel reopens. */
  async function showSurmemSettings(ctx: ExtensionCommandContext): Promise<void> {
    if (!mem) return;

    const all = mem.store.active();
    const statusLines = [
      `memories: ${all.length} (${all.filter((m) => m.kind === "episodic").length} episodic, ${all.filter((m) => m.kind === "semantic").length} semantic)`,
      `embedder: ${embedderName}`,
      `judge:    ${judgeName}`,
      `config:   ${configPath}`,
    ];

    const currentValues = (): Record<string, number> => {
      const c = mem!.config;
      return {
        tauAdd: c.gate.tauAdd,
        dupSim: c.gate.dupSim,
        conflictSim: c.gate.conflictSim,
        minTokens: c.gate.minTokens,
        decayRatePerHour: c.store.decayRatePerHour,
        semanticDecayRatePerHour: c.store.semanticDecayRatePerHour,
        forgetThreshold: c.store.forgetThreshold,
        snapshotSize,
      };
    };

    const commit = (key: keyof ExtConfig, value: number): void => {
      const cfg = { ...loadConfig(), [key]: value };
      persistConfig(cfg);
      applyConfig(cfg);
      snapshot = buildSnapshot(); // snapshotSize may have changed
    };

    const values = currentValues();
    const items: SettingItem[] = CONFIG_ROWS.map((row) => ({
      id: row.key,
      label: row.key,
      description: row.desc,
      currentValue: String(values[row.key]),
    }));

    let list: SettingsList | null = null;
    let currentIndex = 0;

    const picked = await ctx.ui.custom<string | undefined>(
      (_tui, _theme, _kb, done) => {
        const container = new Container();
        container.addChild(new Text("SurMem", 0, 0));
        container.addChild(new Spacer(1));
        for (const line of statusLines) container.addChild(new Text(line, 0, 0));
        container.addChild(new Spacer(1));

        list = new SettingsList(
          items,
          items.length + 2,
          getSettingsListTheme(),
          () => {}, // values edited via the input dialog below, not inline
          () => done(undefined),
        );
        container.addChild(list);

        return {
          render: (w: number) => container.render(w),
          invalidate: () => container.invalidate(),
          handleInput: (data: string) => {
            if (matchesKey(data, "up")) currentIndex = Math.max(0, currentIndex - 1);
            else if (matchesKey(data, "down")) currentIndex = Math.min(items.length - 1, currentIndex + 1);
            if (matchesKey(data, Key.enter)) {
              done(items[currentIndex].id);
              return;
            }
            list?.handleInput?.(data);
          },
        };
      },
      { overlay: true },
    );

    if (!picked) return;

    const row = CONFIG_ROWS.find((r) => r.key === picked);
    const current = String(values[picked as keyof ExtConfig]);

    // Re-prompt on invalid input with the last entry, Esc cancels.
    let input: string | undefined = await ctx.ui.input(
      `${picked} — ${row?.desc ?? ""}`,
      current,
    );
    while (input != null) {
      const trimmed = input.trim();
      const n = Number(trimmed);
      if (trimmed !== "" && Number.isFinite(n)) {
        commit(picked as keyof ExtConfig, n);
        ctx.ui.notify(`SurMem: ${picked} = ${n} (saved to ${configPath})`, "info");
        await showSurmemSettings(ctx);
        return;
      }
      input = await ctx.ui.input(`${picked} (number required)`, trimmed);
    }
  }

  pi.on("session_shutdown", async () => {
    try {
      if (mem) {
        await mem.reflect(); // consolidate episodic clusters into semantic facts
        await mem.save();
      }
      if (judge && "dispose" in judge) {
        await (judge as GgufMemorabilityJudge).dispose();
      }
    } catch {
      // Best effort on the way out.
    }
  });
}
