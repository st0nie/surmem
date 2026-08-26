# SurMem

**Surprise-gated, agent-agnostic long-term memory framework** — TypeScript, zero
required runtime dependencies, Bun-first.

Inspired by Google's [Titans](https://arxiv.org/abs/2501.00663) ("learning to
memorize at test time"), but implemented as an **external** memory layer so it
works with any agent and any LLM:

> Store what is surprising. Reinforce what repeats. Forget what is routine.

---

## Design

### Why surprise?

Humans don't remember every brick on their daily commute, but they do remember
the deer that suddenly appeared on the road. The value of a memory is
proportional to how unexpected its information is. Titans uses next-token
prediction error as its surprise signal inside the model; since we cannot
instrument an arbitrary agent's model internals, we approximate surprise with
**externally computable signals**:

| Titans (internal)                  | SurMem (external proxy)                                       |
| ---------------------------------- | ------------------------------------------------------------- |
| Next-token prediction error        | **Semantic novelty**: 1 − max cosine(new embedding, memories) |
| Momentum term (sustained surprise) | Accumulated novelty over a sliding time window                |
| Decay gate (forgetting)            | Ebbinghaus-style strength decay + retrieval reinforcement     |

An optional LLM judge arbitrates the ambiguous middle zone (is this a new fact,
or does it contradict an old one?).

### Architecture

```
                        ┌─────────────────────────────────────────────┐
   Agent event stream ─▶│  SurpriseGate (write-side policy)            │
   (observe)            │                                              │
                        │   novelty = 1 − max cos(emb, memories)       │
                        │                                              │
                        │   novelty > tauAdd     → ADD (new memory)    │
                        │   sim     > dupSim     → REINFORCE (spacing) │
                        │   conflict zone        → UPDATE (supersede)  │
                        │   trivial / routine    → NOOP (discard)      │
                        └──────┬──────────────────────┬───────────────┘
                               ▼                      ▼
                    ┌──────────────────┐   ┌──────────────────────┐
                    │ Episodic store   │   │ Semantic store       │
                    │ raw, timestamped │──▶│ consolidated facts   │
                    │ decays fast      │consolidate│ decays slow   │
                    └──────────────────┘   └──────────────────────┘
                               │                      ▲
                               ▼                      │
                    ┌─────────────────────────────────┴───────────┐
                    │ Consolidator                                │
                    │ periodically clusters episodic memories and │
                    │ distills them into semantic facts           │
                    └─────────────────────────────────────────────┘

   Agent query ─────▶  Retriever: score = α·relevance + β·recency + γ·strength
                       strength = base · (1+ln(1+accessCount)) · exp(−λ·hours)
                       strength < threshold → forgotten (rescuable via REINFORCE)
```

### Cognitive layers

- **Episodic memory**: raw events with timestamps, recorded verbatim, decays fast
- **Semantic memory**: distilled by the consolidator from episodic clusters;
  stable, high-strength, decays ~10× slower
- **Procedural memory**: reserved interface — skills/workflows can be stored as
  special semantic memories

### Agent-agnostic design

1. **Two-call integration**: `await mem.observe(text)` (write path) and
   `await mem.recall(query)` (read path) plug into any agent loop.
2. **Pluggable embedder**: built-in zero-dependency `HashEmbedder` for demos;
   `GgufEmbedder` for fully local embeddings via node-llama-cpp (the same
   embeddinggemma-300M / Qwen3-Embedding models qmd uses); `OpenAIEmbedder`
   for any OpenAI-compatible `/embeddings` endpoint (OpenAI, vLLM, Ollama,
   llama.cpp).
3. **Pluggable LLM judge / summarizer**: optional; without them the gate falls
   back to recency-wins conflict resolution and representative promotion.
4. **Runtime-configurable**: all gate/store thresholds can be tuned live via
   `mem.configure()` (the pi extension exposes this as `/surmem`).
5. **Replaceable persistence**: in-memory by default; JSON and SQLite
   persisters built in; vector DBs can be added behind the same interface.

### Honest limitations

- External novelty ≠ Titans' true prediction-error surprise: a well-known fact
  absent from the store still counts as "surprising".
- Thresholds are hand-tuned per embedding model (cosine distributions vary).
  The long-term fix is an RL-trained write policy (see Roadmap).
- `HashEmbedder` does not understand paraphrases; use a real embedding model
  for anything serious.

### Project layout

```
src/
  types.ts            # MemoryRecord / WriteVerdict / Kind
  embeddings.ts       # Embedder interface, HashEmbedder, OpenAIEmbedder
  local-embedder.ts   # GgufEmbedder (node-llama-cpp, qmd-compatible models)
  gate.ts             # SurpriseGate: novelty + momentum + runtime-configurable verdicts
  store.ts            # two-layer store, Ebbinghaus decay, runtime-configurable
  persistence.ts      # JsonPersister / SqlitePersister
  retrieval.ts        # hybrid scoring (relevance + recency + strength)
  consolidation.ts    # episodic cluster -> semantic fact
  judge.ts            # OpenAIJudge (conflict arbitration), OpenAISummarizer
  index.ts            # SurpriseMemory facade + public exports
demo.ts               # full lifecycle demo (bun run demo.ts)
tests/                # bun test suite (14 tests)
scripts/gguf-smoke.ts # local-model smoke test
extensions/surmem/    # pi coding-agent extension
```

---

## Quick start

```bash
bun install
bun run demo.ts     # full lifecycle: gate verdicts, recall, consolidation, forgetting
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

// Runtime tuning (also exposed as /surmem in the pi extension)
mem.configure({ gate: { tauAdd: 0.5 }, store: { forgetThreshold: 0.2 } });
```

## Production configuration

Fully local (zero API cost, qmd's embedding model):

```typescript
import { GgufEmbedder, SurpriseMemory, SqlitePersister } from "./src/index";

const pair = GgufEmbedder.createPair({
  modelPath: `${process.env.HOME}/.cache/qmd/models/hf_ggml-org_embeddinggemma-300M-Q8_0.gguf`,
  dim: 768,
});
const mem = new SurpriseMemory({
  embedder: pair.document,
  queryEmbedder: pair.query,
  store: { persister: new SqlitePersister("./memory.sqlite") },
});
```

OpenAI-compatible endpoint (OpenAI / vLLM / Ollama / llama.cpp) with LLM
arbitration and consolidation summaries:

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
    // baseUrl: "http://localhost:8080/v1",
  }),
  gate: {
    judge: new OpenAIJudge({ apiKey: process.env.OPENAI_API_KEY!, model: "gpt-4o-mini" }),
  },
  consolidation: {
    summarizer: new OpenAISummarizer({ apiKey: process.env.OPENAI_API_KEY!, model: "gpt-4o-mini" }),
  },
  store: { persistPath: "./memory.json" },
});
```

## pi coding-agent extension

Load once for testing:

```bash
pi -e ./extensions/surmem/index.ts
```

Or register it globally so it auto-loads in every pi session — add the absolute
path to the `extensions` list in `~/.pi/agent/settings.json`:

```json
{
  "extensions": ["/absolute/path/to/memoresearch/extensions/surmem/index.ts"]
}
```

(Use the absolute path rather than a symlink: pi's loader does not resolve
symlinks, and the extension imports the core library relatively.)

The extension:

- **fully automatic observation**: every user message is screened by a cheap
  prefilter, then judged by a **local GGUF memorability judge** (qmd's cached
  qmd-query-expansion-1.7B — zero API cost). Questions and commands are
  dropped; durable facts proceed,
- **steering reminders**: memorable candidates are not written silently. A
  `<system-reminder name="surmem">` steering message nudges the main agent,
  which decides in context and stores via the `surmem_remember` tool (the
  surprise gate inside still handles ADD/UPDATE/REINFORCE dedup),
- injects a KV-cache-stable snapshot of your strongest memories into the system
  prompt (checkpoint-refreshed, following pi-memory's approach),
- gives the agent `surmem_remember` and `surmem_recall` tools,
- adds a single `/surmem` command opening a settings-style panel (status +
  live-tunable parameters),
- consolidates (episodic -> semantic, idempotent + deduplicated) and persists
  memory on session shutdown.

### Extension commands

```
/surmem     # settings panel: status (memory counts, embedder, judge, config
            # path) on top; parameters below. up/down to select, enter to edit,
            # enter to save, esc to close.
            # Saved to ~/.pi/agent/surmem.json (global) and applied live.
```

Tunable parameters: `tauAdd`, `dupSim`, `conflictSim`, `minTokens`,
`decayRatePerHour`, `semanticDecayRatePerHour`, `forgetThreshold`,
`snapshotSize`.

### Extension environment variables

| Variable | Default | Description |
|---|---|---|
| `SURMEM_GGUF_MODEL_PATH` | qmd-cached embeddinggemma if present | Local GGUF embedding model (highest precedence) |
| `SURMEM_GGUF_DIM` | `768` | Embedding dim (`1024` for Qwen3-Embedding-0.6B) |
| `SURMEM_JUDGE_GGUF` | qmd-cached qmd-query-expansion-1.7B if present | Local GGUF memorability judge |
| `SURMEM_GGUF_GPU` | unset (CPU) | llama.cpp GPU backend: `auto`, `cuda`, `metal`, `vulkan`. CPU is the default so the embedder and judge never fight over VRAM |
| `SURMEM_EMBEDDING_API_KEY` | unset | OpenAI-compatible embedder (used when no GGUF model) |
| `SURMEM_EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | Endpoint override |
| `SURMEM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `SURMEM_EMBEDDING_DIM` | `1536` | Endpoint model dim |
| `SURMEM_STORE_PATH` | `<cwd>/.pi/surmem/memory.json` | Memory store location (per-project) |
| `SURMEM_CONFIG_PATH` | `~/.pi/agent/surmem.json` | Config file location (global) |

## Tests

```bash
bun test                              # unit tests (no network, deterministic)
bun x tsc --noEmit                    # core typecheck
bun x tsc -p extensions/surmem        # extension typecheck (against real pi types)
bun run scripts/gguf-smoke.ts         # local-model smoke test (needs the GGUF file)
```

## Roadmap

- Vector-index acceleration (sqlite-vec) for >10⁴ memories
- Procedural memory (skill consolidation from successful trajectories)
- RL-trained write policy to replace the heuristic surprise gate
  (decision logs already carry the audit fields needed for training data)
- Benchmark evaluation on LoCoMo / LongMemEval

## License

MIT
