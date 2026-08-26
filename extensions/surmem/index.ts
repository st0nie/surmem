/**
 * SurMem pi extension — surprise-gated long-term memory for the pi coding agent.
 *
 * What it does:
 *   - Auto-observes user messages through the surprise gate (routine chatter is
 *     discarded; novel facts are stored; contradictions supersede old facts)
 *   - Injects a KV-cache-stable snapshot of the strongest memories into the
 *     system prompt (refreshed only at checkpoints, pi-memory style)
 *   - Registers `memory_remember` / `memory_recall` tools for explicit,
 *     prompt-dependent memory access
 *   - Registers the `/memory` command for stats
 *   - Consolidates (episodic -> semantic) and persists on session shutdown
 *
 * Load it with:
 *   pi -e ./extensions/surmem/index.ts
 * or add the directory to the "extensions" list in settings.json.
 *
 * Configuration (env):
 *   SURMEM_GGUF_MODEL_PATH     - fully local embeddings via node-llama-cpp
 *                                (e.g. ~/.cache/qmd/models/hf_ggml-org_embeddinggemma-300M-Q8_0.gguf)
 *   SURMEM_GGUF_DIM            - model output dim (default 768; use 1024 for Qwen3-Embedding-0.6B)
 *   SURMEM_EMBEDDING_API_KEY   - enables the OpenAI-compatible embedder
 *   SURMEM_EMBEDDING_BASE_URL  - default https://api.openai.com/v1
 *   SURMEM_EMBEDDING_MODEL     - default text-embedding-3-small
 *   SURMEM_EMBEDDING_DIM       - default 1536
 *   SURMEM_STORE_PATH          - default <cwd>/.pi/surmem/memory.json
 *
 * Without embedding env vars it falls back to the built-in HashEmbedder
 * (offline, zero-dependency, lower quality).
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { CONFIG_DIR_NAME } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { join } from "node:path";

import {
  GgufEmbedder,
  HashEmbedder,
  OpenAIEmbedder,
  SurpriseMemory,
  WriteVerdict,
  type Embedder,
} from "../../src/index";

const SNAPSHOT_SIZE = 8;
const OBSERVE_MAX_CHARS = 2000;

interface EmbedderPair {
  embedder: Embedder;
  queryEmbedder?: Embedder;
}

/**
 * Embedder selection precedence:
 *   1. SURMEM_GGUF_MODEL_PATH -> fully local GGUF embeddings (qmd-style,
 *      e.g. ~/.cache/qmd/models/hf_ggml-org_embeddinggemma-300M-Q8_0.gguf)
 *   2. SURMEM_EMBEDDING_API_KEY -> OpenAI-compatible endpoint
 *   3. HashEmbedder fallback (offline, zero-dependency, lower quality)
 */
function embeddersFromEnv(): EmbedderPair {
  const ggufPath = process.env.SURMEM_GGUF_MODEL_PATH;
  if (ggufPath) {
    const dim = Number(process.env.SURMEM_GGUF_DIM ?? 768);
    const pair = GgufEmbedder.createPair({ modelPath: ggufPath, dim });
    return { embedder: pair.document, queryEmbedder: pair.query };
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
    return { embedder };
  }
  return { embedder: new HashEmbedder() };
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

export default function (pi: ExtensionAPI) {
  let mem: SurpriseMemory | null = null;
  let snapshot = "";

  /** Top-N strongest memories, prompt-independent for KV-cache stability. */
  function buildSnapshot(): string {
    if (!mem) return "";
    const top = mem.store
      .active()
      .map((r) => ({ r, s: mem!.store.effectiveStrength(r) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, SNAPSHOT_SIZE);
    if (top.length === 0) return "";
    const lines = top.map(({ r }) => `- [${r.kind}] ${r.text}`);
    return [
      "## Long-term memories (SurMem)",
      "",
      ...lines,
      "",
      "Use the `memory_recall` tool to search memory for a specific topic, and " +
        "`memory_remember` to explicitly store an important fact, preference, or decision.",
    ].join("\n");
  }

  async function observeAndMaybeRefresh(text: string, origin: string): Promise<void> {
    if (!mem) return;
    const trimmed = text.slice(0, OBSERVE_MAX_CHARS);
    try {
      const r = await mem.observe(trimmed, { origin });
      if (r.verdict === WriteVerdict.ADD || r.verdict === WriteVerdict.UPDATE) {
        snapshot = buildSnapshot();
      }
    } catch {
      // Memory writes must never break the agent loop.
    }
  }

  pi.on("session_start", async (_event, ctx) => {
    const persistPath =
      process.env.SURMEM_STORE_PATH ??
      join(ctx.cwd, CONFIG_DIR_NAME, "surmem", "memory.json");
    const embedders = embeddersFromEnv();
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
    snapshot = buildSnapshot();
    const n = mem.store.active().length;
    if (ctx.hasUI) {
      ctx.ui.setStatus("surmem", `surmem: ${n} memories`);
    }
  });

  // Inject the stable snapshot into the system prompt each turn. The bytes are
  // identical between checkpoints, so prefix KV caches stay valid.
  pi.on("before_agent_start", async (event) => {
    if (!snapshot) return;
    return { systemPrompt: event.systemPrompt + "\n\n" + snapshot };
  });

  // Auto-observe user prompts; the surprise gate filters routine chatter.
  pi.on("message_end", async (event) => {
    if (!mem || event.message.role !== "user") return;
    const text = extractText(
      (event.message as { content?: unknown }).content,
    );
    if (text) await observeAndMaybeRefresh(text, "user-message");
  });

  pi.registerTool({
    name: "memory_remember",
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
    name: "memory_recall",
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

  pi.registerCommand("memory", {
    description: "Show SurMem memory stats and strongest memories",
    handler: async (_args, ctx) => {
      if (!mem) {
        ctx.ui.notify("SurMem not initialized", "warning");
        return;
      }
      const all = mem.store.active();
      const episodic = all.filter((m) => m.kind === "episodic").length;
      const semantic = all.filter((m) => m.kind === "semantic").length;
      const top = all
        .map((r) => ({ r, s: mem!.store.effectiveStrength(r) }))
        .sort((a, b) => b.s - a.s)
        .slice(0, 5)
        .map(({ r, s }) => `  [${r.kind} ${s.toFixed(2)}] ${r.text}`)
        .join("\n");
      ctx.ui.notify(
        `SurMem: ${all.length} memories (${episodic} episodic, ${semantic} semantic)\n${top}`,
        "info",
      );
    },
  });

  pi.on("session_shutdown", async () => {
    if (!mem) return;
    try {
      await mem.reflect(); // consolidate episodic clusters into semantic facts
      await mem.save();
    } catch {
      // Best effort on the way out.
    }
  });
}
