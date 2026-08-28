# Pi SurMem

Production-grade long-term memory for [Pi](https://pi.dev): **surprise-gated learning, native hybrid retrieval, scoped SQLite storage, session search, safety scanning, and Pi-native procedural skills**.

> Remember what is novel, reinforce what repeats, supersede only proven contradictions, and forget weak episodic traces.

## Why SurMem

SurMem was explicitly benchmarked against [`pi-memory`](https://pi.dev/packages/pi-memory) 0.4.2 and [`pi-hermes-memory`](https://pi.dev/packages/pi-hermes-memory) 0.9.6.

| Capability | pi-memory | pi-hermes-memory | SurMem |
|---|---:|---:|---:|
| Zero-config durable memory | Markdown | Markdown + SQLite | **Built-in SQLite** |
| Semantic memory search | qmd required | Keyword FTS5 | **Native vector + lexical hybrid** |
| External/native dependency required | qmd for semantic search | `better-sqlite3` ABI | **No** (Bun/Node built-in SQLite) |
| Global + project scope | Global | Global + project | **Global + canonical project isolation** |
| Surprise gate / reinforcement / decay | No | Aging for consolidation | **Yes** |
| Episodic → semantic consolidation | No | LLM consolidation | **Idempotent native consolidation; optional LLM summary** |
| Cross-session conversation search | No | SQLite FTS5 | **SQLite FTS5, CJK fallback, incremental indexing** |
| Secret and injection scanning | No | Yes | **Yes, before every durable write** |
| Recoverable deletion | Yes | No dedicated recovery ID | **Yes** |
| Embedding model migration | qmd-managed | N/A | **Fingerprint detection + automatic reindex** |
| Multi-process write safety | File-oriented | Atomic/SQLite locks | **Revision CAS, merge, tombstones, WAL** |
| Pi-native procedural skills | No | Yes | **Yes, structured and safety-scanned** |
| Full memory prompt tax by default | Bounded snapshot | Policy-only | **Policy + small cache-stable snapshot + on-demand recall** |

SurMem's main differentiator is not “more automatic LLM calls.” It is a single coherent cognitive store with semantic retrieval and explicit data-integrity guarantees. Automatic candidates are transient and require the main agent to confirm them, avoiding silent memory pollution.

## Install

From this repository:

```bash
pi install git:github.com/st0nie/surmem
```

For local development:

```bash
pi install /absolute/path/to/surmem
# or one-shot:
pi -e /absolute/path/to/surmem/extensions/surmem/index.ts
```

The package is ready for npm publication as `pi-surmem`; after publication it can be installed with `pi install npm:pi-surmem`.

## Zero-config behavior

No qmd, API key, paid inference, or native SQLite addon is required. On first use SurMem automatically downloads two GGUF files into qmd's shared model cache:

- `EmbeddingGemma-300M-Q8_0` (about 334 MB, 768 dimensions) for semantic vectors.
- `Qwen3-4B-Q4_K_M` (about 2.5 GB) for durable-memory judgment and contradiction arbitration.

Each model runs in a private loopback daemon. All Pi processes share the same embedding PID and the same judgment PID; opening more sessions does not load duplicate model instances. Daemons use bearer-token authentication, private state files, progress reporting, crash-safe startup locks, proxy-aware downloads, and a 30-minute idle timeout.

Built-in SQLite provides WAL-backed storage and FTS5 session search. Durable memories are split into global and current-project stores. Candidate memories remain transient until the main agent confirms them with `surmem_remember`. A small strongest-memory snapshot stays stable between deliberate refreshes.

`HashEmbedder` remains available only as an explicit emergency/test fallback with `SURMEM_EMBEDDER=hash`.

## Tools

| Tool | Purpose |
|---|---|
| `surmem_remember` | Store a durable fact with ADD/UPDATE/REINFORCE/NOOP gating; `supersedes` explicitly replaces a refined or corrected memory |
| `surmem_recall` | Hybrid semantic + lexical recall across global/project scope |
| `surmem_list` | Inspect recent memories and stable IDs |
| `surmem_forget` | Delete by ID and create a recovery record; a skill-backed memory also removes its on-disk skill files |
| `surmem_restore` | Restore a deleted memory by recovery ID; recreates skill files when the recovery record carries them |
| `surmem_status` | Store, model, index, configuration, and error health report |
| `surmem_session_search` | Search past Pi JSONL conversations through SQLite FTS5 |
| `surmem_export` | Create a private JSON export |
| `surmem_skill` | Create/view/delete structured Pi-native procedural skills; delete also tombstones the backing memory record (recoverable via `surmem_restore`) |
| `surmem_clear` | Explicitly clear one scope with a confirmation phrase |

Automatic deduplication cannot recognize every refinement: a corrected or generalized fact is semantically close to the memory it replaces, so the gate may REINFORCE or NOOP it. When a NOOP blocks a write, the result names the nearest blocking memory ID; pass it back as `supersedes` to deterministically replace the old record (it is retained as superseded for audit).

`/surmem` opens an interactive menu in TUI mode. The title shows active memory counts; from the menu you can:

- **Manage project memories** / **Manage global memories** — full CRUD per scope: list recent memories, search, add (episodic or semantic), view/edit text, and delete. Edits keep the record ID and writes are safety-scanned and re-embedded. Both edits and deletes write a recovery file under `recovery/` first, so the previous version can be brought back with `surmem_restore`.
- **Status details** — show embedder, judge, arbiter, storage path, and warnings.
- Toggle common settings (`snapshotSize`, `autoCandidates`, `autoMaintenance`, `sessionSearch`) and export both scopes to JSON.

`/surmem status` prints a one-line summary and works in all modes.

## Data layout

```text
~/.pi/agent/surmem/
├── config.json
├── global.sqlite
├── sessions.sqlite
├── projects/
│   └── <sha256-prefix-of-canonical-project-path>.sqlite
├── recovery/
├── exports/
├── migrations/
├── embedding-daemon/
├── judgment-daemon/
└── skills/
    ├── global/<skill>/SKILL.md
    └── projects/<project-key>/<skill>/SKILL.md
```

Files are created with private permissions (`0600`, directories `0700`). Project identity uses the canonical real path (project keys are the first 20 hex characters of its SHA-256), not only the directory basename, so repositories with the same name do not collide.

## Safety model

Memory is **untrusted historical data**, never instruction-level authority.

- API keys, tokens, private keys, password assignments, invisible Unicode, prompt-injection phrases, and exfiltration payloads are rejected.
- Snapshot and recall output are fenced as `trust="untrusted-data"` and XML-escaped.
- Current user requests, repository content, and tool output explicitly override recalled memory.
- Candidate reminders are injected through Pi's transient `context` hook and do not pollute the session transcript.
- Tool output, result count, candidate count, config size, and memory text have hard limits.

Do not use memory as a hard security policy. Enforce dangerous-operation prohibitions with a Pi `tool_call` guard.

## Configuration

Edit `~/.pi/agent/surmem/config.json` or use `/surmem` for common settings:

```json
{
  "tauAdd": 0.45,
  "dupSim": 0.85,
  "conflictSim": 0.55,
  "minTokens": 3,
  "decayRatePerHour": 0.02,
  "semanticDecayRatePerHour": 0.002,
  "forgetThreshold": 0.1,
  "snapshotSize": 8,
  "autoCandidates": true,
  "autoMaintenance": true,
  "sessionSearch": true
}
```

Configuration is strictly range-validated, capped at 64 KiB, atomically replaced, and never overwritten when malformed.

### Embedding backends

**Default shared embedding daemon**:

```bash
# Defaults shown; no configuration is required.
export SURMEM_GGUF_MODEL_URI='hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf'
export SURMEM_GGUF_DIM=768
# optional: auto, cuda, metal, vulkan; CPU is default
export SURMEM_GGUF_GPU=auto
```

Use `SURMEM_GGUF_MODEL_PATH` to supply an already-downloaded model. Use `SURMEM_EMBEDDER=hash` only to explicitly disable neural embeddings.

**OpenAI-compatible endpoint**:

```bash
export SURMEM_EMBEDDING_API_KEY=...
export SURMEM_EMBEDDING_BASE_URL=https://api.openai.com/v1
export SURMEM_EMBEDDING_MODEL=text-embedding-3-small
export SURMEM_EMBEDDING_DIM=1536
```

**Default shared judgment + arbitration daemon**:

```bash
# One Qwen model serves both roles; no remote tokens are consumed.
export SURMEM_JUDGE_GGUF_URI='hf:ggml-org/Qwen3-4B-GGUF:Q4_K_M'
# Optional local file override:
export SURMEM_JUDGE_GGUF=/models/Qwen3-4B-Q4_K_M.gguf
export SURMEM_JUDGE_GGUF_GPU=auto
```

For constrained machines, explicitly use heuristic mode:

```bash
export SURMEM_JUDGE_MODE=heuristic
```

An OpenAI-compatible judge remains an opt-in override:

```bash
export SURMEM_JUDGE_API_KEY=...
export SURMEM_JUDGE_MODEL=gpt-4o-mini
export SURMEM_JUDGE_BASE_URL=https://api.openai.com/v1
```

**Optional separate contradiction arbiter override**:

```bash
export SURMEM_ARBITER_API_KEY=...
export SURMEM_ARBITER_MODEL=gpt-4o-mini
export SURMEM_ARBITER_BASE_URL=https://api.openai.com/v1
```

Other paths:

| Variable | Default |
|---|---|
| `SURMEM_DIR` | `~/.pi/agent/surmem` |
| `SURMEM_CONFIG_PATH` | `<SURMEM_DIR>/config.json` |
| `SURMEM_STORE_PATH` | Only used as a legacy JSON migration source |
| `SURMEM_HTTP_TIMEOUT_MS` | `30000` |
| `SURMEM_GGUF_DAEMON_IDLE_MS` | `1800000` (30 minutes) |
| `SURMEM_JUDGE_DAEMON_IDLE_MS` | `1800000` (30 minutes) |
| `SURMEM_DAEMON_MODEL_DIR` | `~/.cache/qmd/models` |
| `SURMEM_JUDGE_MODE` | local shared GGUF; set `heuristic` only to disable it |

Live model state and download progress:

```bash
cat ~/.pi/agent/surmem/embedding-daemon/state.json
cat ~/.pi/agent/surmem/judgment-daemon/state.json
```

## Migration from SurMem 0.1

On first startup, the extension automatically imports:

- project JSON memory from `<cwd>/.pi/surmem/memory.json` (or `SURMEM_STORE_PATH`),
- global config from `~/.pi/agent/surmem.json`.

The old files are preserved. A migration report is written under `~/.pi/agent/surmem/migrations/`. Unsafe legacy records are skipped and listed in that report rather than injected.

Core `JsonPersister` also reads the old top-level array format. `SqlitePersister` migrates the original column-based `memories` table and re-embeds records when the embedding fingerprint differs.

## Core library

```ts
import { SqlitePersister, SurpriseMemory } from "pi-surmem";

const memory = new SurpriseMemory({
  store: { persister: new SqlitePersister("./memory.sqlite") },
});

await memory.load();
await memory.observe("The project uses pnpm workspaces.", {
  scope: "project",
  project: "example",
});

const hits = await memory.recall("package manager", 5);
console.log(hits);
await memory.close();
```

The core supports custom embedders, query/document asymmetric embeddings, LLM conflict judges, consolidation summarizers, custom persisters, explicit reindex, export, restore, and health reporting.

## Reliability and operations

- Writes auto-save immediately after mutation; shutdown is not the only durability boundary.
- JSON uses lock files, optimistic revisions, merge-on-conflict, atomic rename, and deletion tombstones.
- SQLite uses WAL, `BEGIN IMMEDIATE`, revision checks, upserts, and tombstones.
- Corrupt/unreadable stores produce explicit path-bearing errors and are preserved; they are never silently replaced with an empty store.
- SQLite handles are checkpointed and closed on session replacement, reload, and shutdown.
- Embedding/judge async results carry session-generation guards so stale work cannot leak into a replacement session.

For backup, stop Pi (or allow graceful shutdown) and copy `~/.pi/agent/surmem/`. JSON exports created by `surmem_export` are portable and do not depend on SQLite.

## Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for subsystem boundaries, persistence invariants, retrieval scoring, security rules, and Pi lifecycle design. Contributors and coding agents should also follow [`AGENTS.md`](./AGENTS.md).

## Development and verification

```bash
bun install
bun run check       # core + extension typecheck, Biome, all tests
bun run pack:check  # verify published file set
bun run demo
```

Current deterministic suite: **45 tests, 151 assertions across 6 files**, including concurrent writers, deletion non-resurrection, corruption behavior, legacy migrations, embedding reindex, safety fencing, extension lifecycle, explicit supersede, FTS5, CJK fallback, and package integration.

Optional real-model smoke test:

```bash
SURMEM_GGUF_MODEL_PATH=/models/embeddinggemma.gguf bun run smoke:gguf
```

## Honest limits

- Hash embeddings are lexical approximations. Use a real embedding model for strong paraphrase recall.
- Similarity alone cannot prove contradiction; without an arbiter SurMem preserves related facts instead of destructively updating them.
- Brute-force in-memory vector scoring is designed for tens of thousands of curated memories, not millions.
- Session search indexes text messages, not tool-result payloads, to avoid persisting large or sensitive command output.
- Automatic candidates intentionally require agent confirmation; this trades maximum automation for lower memory pollution.

## License

MIT
