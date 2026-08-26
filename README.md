# SurMem

**Surprise-gated, agent-agnostic long-term memory framework** — TypeScript, zero
runtime dependencies, Bun-first.

Inspired by Google's [Titans](https://arxiv.org/abs/2501.00663) ("learning to
memorize at test time"), but implemented as an **external** memory layer so it
works with any agent and any LLM:

> Store what is surprising. Reinforce what repeats. Forget what is routine.

See [DESIGN.md](DESIGN.md) for the architecture and
[agent-memory-survey-2026.md](agent-memory-survey-2026.md) for the research
landscape this design is grounded in.

## Features

- **Surprise gate** — every observation gets a verdict:
  `ADD` (novel) / `REINFORCE` (duplicate → spacing effect) /
  `UPDATE` (contradiction → supersedes old fact) / `NOOP` (trivial → discarded)
- **Titans-style momentum** — a stream of mildly-novel related events can cross
  the write threshold
- **Ebbinghaus decay** — memory strength decays exponentially; retrieval and
  reinforcement rescue memories from forgetting (spaced repetition)
- **Consolidation** — episodic clusters are distilled into long-lived semantic
  memories that decay ~10× slower
- **Hybrid retrieval** — relevance + recency + strength scoring
- **Pluggable backends** — built-in zero-dependency `HashEmbedder`;
  `OpenAIEmbedder` for any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama,
  llama.cpp — e.g. local embeddinggemma / Qwen3-Embedding GGUFs);
  optional `OpenAIJudge` / `OpenAISummarizer` for conflict arbitration and
  consolidation summaries
- **Persistence** — JSON file storage, Node and Bun compatible
- **pi extension** — ready-to-use extension for the
  [pi coding agent](https://github.com/earendil-works/pi) in
  [`extensions/surmem/`](extensions/surmem/index.ts)

## Quick start

```bash
bun run demo.ts     # full lifecycle demo: gate verdicts, recall, consolidation, forgetting
bun test            # unit tests
```

```typescript
import { SurpriseMemory } from "./src/index";

const mem = new SurpriseMemory({
  store: { persistPath: "./memory.json" },
});
await mem.load();

// Write path — gated by surprise
await mem.observe("The user moved from Beijing to Shanghai.");   // ADD
await mem.observe("The user moved from Beijing to Shanghai.");   // REINFORCE
await mem.observe("The user moved back to Beijing last week.");  // UPDATE (supersedes)
await mem.observe("ok");                                          // NOOP

// Read path — inject into your agent's prompt
console.log(await mem.recallAsContext("Where does the user live?"));

// Periodic maintenance
await mem.reflect();    // consolidate episodic clusters -> semantic facts
mem.forgetPass();       // prune memories whose strength decayed away
await mem.save();
```

## Production configuration

```typescript
import {
  SurpriseMemory,
  OpenAIEmbedder,
  OpenAIJudge,
  OpenAISummarizer,
} from "./src/index";

const mem = new SurpriseMemory({
  embedder: new OpenAIEmbedder({
    apiKey: process.env.OPENAI_API_KEY!,
    model: "text-embedding-3-small",
    // baseUrl: "http://localhost:8080/v1", // llama.cpp / Ollama / vLLM
  }),
  gate: {
    tauAdd: 0.45,        // novelty threshold for ADD
    dupSim: 0.85,        // similarity threshold for REINFORCE
    conflictSim: 0.55,   // similarity threshold for the UPDATE zone
    judge: new OpenAIJudge({
      apiKey: process.env.OPENAI_API_KEY!,
      model: "gpt-4o-mini",
    }),
  },
  consolidation: {
    clusterSim: 0.3,
    summarizer: new OpenAISummarizer({
      apiKey: process.env.OPENAI_API_KEY!,
      model: "gpt-4o-mini",
    }),
  },
  store: { persistPath: "./memory.json" },
});
```

For local-first deployments, serve
[embeddinggemma-300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF)
(English) or
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)
(multilingual/CJK) through llama.cpp's or Ollama's OpenAI-compatible endpoint
and point `baseUrl` at it.

## pi coding-agent extension

```bash
pi -e ./extensions/surmem/index.ts
```

The extension:

- auto-observes your prompts through the surprise gate,
- injects a KV-cache-stable snapshot of your strongest memories into the system
  prompt (checkpoint-refreshed, following pi-memory's approach),
- gives the agent `memory_remember` and `memory_recall` tools,
- adds a `/memory` stats command,
- consolidates and persists memory on session shutdown.

Environment variables: see the header comment in
[`extensions/surmem/index.ts`](extensions/surmem/index.ts).

## Tests

```bash
bun test            # unit tests (no network, deterministic)
bun x tsc --noEmit  # typecheck
```

## Roadmap

- SQLite / vector-DB storage backends behind the store interface
- Procedural memory (skill consolidation from successful trajectories)
- RL-trained write policy to replace the heuristic surprise gate
- Benchmark evaluation on LoCoMo / LongMemEval

## License

MIT
