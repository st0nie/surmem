# SurMem Architecture

## Objective

SurMem is a production memory substrate for Pi and other agents. It combines cognitive memory policies with database-grade durability:

- remember novel durable facts,
- reinforce repetition,
- supersede only high-confidence contradictions,
- consolidate repeated episodes into semantic knowledge,
- decay weak episodic traces,
- retrieve under a strict token/result budget,
- never let persisted text become trusted instructions.

## Architecture

```text
user / agent event
      │
      ▼
content scanner ── blocked ──▶ explicit safe error
      │
      ▼
embed + SurpriseGate
      │
      ├─ ADD ───────▶ new episodic/semantic/procedural record
      ├─ REINFORCE ─▶ access/strength update
      ├─ UPDATE ────▶ new record + superseded link (judge or explicit supersedes)
      └─ NOOP
      │
      ▼
scoped MemoryStore ──▶ immediate CAS/transactional persistence
      │
      ├─ hybrid recall: vector + lexical + recency + strength
      ├─ consolidation: connected episodic components → semantic
      └─ forgetting: configurable Ebbinghaus decay

Pi sessions ──▶ incremental SQLite FTS5 session index
Pi skills   ──▶ structured SKILL.md procedural memory

OpenCode native plugin ──▶ direct scoped backend + system context hook

Codex CLI / Claude Code native plugins
      ├─▶ SessionStart hook ──▶ bounded read-only memory snapshot
      │
      ▼
bundled stdio MCP adapter ──▶ scoped SurpriseMemory facade

all clients ─┬─▶ one shared EmbeddingGemma daemon per storage root
             └─▶ one shared Qwen3 judgment/arbitration daemon per storage root
```

## Scope model

The extension owns two independent stores:

1. **Global**: user preferences and facts valid across projects.
2. **Project**: repository decisions, conventions, and procedures.

Project identity is the first 20 hex characters of the SHA-256 of the canonical real path. Basename-only identity is not safe because unrelated repositories can share a name.

## MCP integration

`src/mcp/cli.ts` exposes SurMem through local stdio MCP, the integration contract shared by OpenCode, Codex CLI, and Claude Code. It reuses the core memory facade and persistence invariants rather than importing Pi APIs.

The server binds project scope at launch with a required absolute `--project` directory. Its storage root defaults to Pi's existing location; `--storage-dir` or `SURMEM_DIR` allows isolation. Clients pointed at the same canonical project and storage root share records rather than maintaining host-specific copies.

Protocol initialization and tool discovery are independent of lazy store/model initialization. Standard output is reserved for MCP messages. Memory mutations persist immediately, and server shutdown closes client resources without terminating shared daemons.

Requests execute serially and reload stores before each operation so long-lived MCP clients see other clients' writes. Recall is read-only with respect to access strength. Forget recoveries use private, scope/project-fenced snapshots under `mcp-recovery/`, separate from Pi recovery files. Operations that explicitly delete or supersede Pi skill-backed records are rejected rather than leaving skill files inconsistent.

MCP alone does not provide host lifecycle events. Native adapters add the supported context hooks described below; transcript indexing, transient candidate extraction, periodic maintenance, and learned skill discovery remain Pi features. No other host transcript is silently indexed.

## Native host integrations

`extensions/opencode/index.ts` exports an OpenCode plugin with direct tools backed by the scoped backend, not an MCP connection. Tool calls and context/lifecycle operations are serialized. The experimental system-transform hook adds a bounded, XML-escaped untrusted snapshot, cached per session until a user message or memory mutation. Idle events close store resources; final disposal prevents reopening.

The repository root is also the Claude and Codex plugin root, with separate `.claude-plugin/plugin.json` and `.codex-plugin/plugin.json` manifests. Keeping core files inside that root makes copied plugin caches self-contained apart from runtime dependencies. Root `skills/memory/SKILL.md` supplies an explicit memory workflow and `hooks/hooks.json` declares a shared SessionStart command. The hook reads the canonical project from host JSON, lists existing scoped records without model calls, emits bounded escaped context, and reports optional failures without blocking the host.

Claude loads root `.mcp.json` and resolves its plugin-root variable. Codex uses its manifest's inline MCP declaration instead: its MCP loader does not interpolate plugin-root variables, so the launcher runs at the plugin root and requires an explicit absolute `SURMEM_PROJECT`. Codex hook discovery does support the shared Claude-style root variable and SessionStart output, subject to host hook trust policy.

Native MCP launchers reuse `src/mcp/cli.ts` for protocol, tools, persistence, and shutdown. Cached installations require explicit runtime dependency setup; hooks and launchers never automatically install packages. `activeMemory` controls proactive guidance, not automatic durable writes. None of these adapters uses session-end as its durability boundary.

## Retrieval

For each active memory:

```text
score = 0.58 × vector_similarity
      + 0.22 × lexical_overlap
      + 0.08 × recency
      + 0.12 × normalized_strength
```

Weights are normalized. Superseded records are excluded. Scope, project, and kind filters are applied before scoring. Retrieval reinforcement is explicit and persists immediately.

Strength follows:

```text
strength = base_strength
         × (1 + ln(1 + access_count))
         × exp(-decay_rate × hours_since_access)
```

Semantic/procedural memories use the slower semantic decay rate.

## Write policy

```text
similarity >= dupSim (shortTextDupSim when the text has fewer than shortTextTokens tokens)
  → REINFORCE

conflictSim <= similarity < dupSim and a judge is configured
  → accept only a strict verdict with sufficient confidence;
    an uncertain judge falls back to the surprise check below

momentum-adjusted surprise > tauAdd
  → ADD

otherwise
  → NOOP (the result reports the nearest blocking memory)
```

Defaults: `tauAdd` 0.40, `dupSim` 0.85, `conflictSim` 0.60, `shortTextTokens` 16, `shortTextDupSim` 0.92. Two invariants are validated at construction and reconfiguration:

- `tauAdd <= 1 - conflictSim`. Isolated writes can only ADD when `1 - similarity > tauAdd`; a larger `tauAdd` would leave similarities in `[1 - tauAdd, conflictSim)` in an unjudged dead zone where every isolated write is NOOP.
- `shortTextDupSim >= dupSim`. Short texts embed noisily and share most of their hashed features, so cosine overstates their similarity; they must clear a higher bar before REINFORCE can merge a distinct fact or silently strengthen a contradicting one.

Momentum is accumulated only from accepted writes (ADD/REINFORCE/UPDATE). Rejected NOOP attempts never enter the novelty window, so rapidly retrying a rejected fact cannot build enough momentum to force it through the gate.

Similarity does not establish contradiction. The Pi extension defaults to one shared local Qwen3-4B GGUF daemon for both judgment and arbitration, so this path consumes no remote-model tokens. If judgment is explicitly disabled or unavailable, related facts are not destructively superseded.

A caller that knows a new fact refines, generalizes, or corrects an existing record can pass `supersedes=<id>` to `observe()` (or the `surmem_remember` tool). This performs a deterministic UPDATE after validating that the target exists, is still active, and belongs to the same scope. Explicit caller instructions are trusted; similarity alone never is.

## Persistence invariants

### Common

- schema version is explicit,
- every record is validated at the read boundary,
- embedding backend has a stable fingerprint,
- fingerprint changes trigger reindex or an explicit mismatch error,
- corruption is surfaced with the path and source file is preserved,
- shutdown closes resources, but normal writes do not depend on shutdown.

### JSON

- private file mode,
- exclusive lock file with stale-lock recovery,
- optimistic revision check,
- read-latest/merge/retry on conflict,
- same-directory temporary file + atomic rename,
- tombstones prevent stale writers from resurrecting deleted records,
- explicit restore carries an undelete marker.

### SQLite

- Bun `bun:sqlite` or Node `node:sqlite`; no native addon ABI,
- WAL, busy timeout, foreign keys,
- `BEGIN IMMEDIATE` revision check,
- record upserts and tombstones,
- legacy column schema migration,
- WAL checkpoint on close.

## Security invariants

1. Durable writes pass secret/injection/invisible-Unicode scanning.
2. Memory text is normalized and bounded.
3. Metadata must be bounded JSON.
4. Prompt output strips control characters and dangerous tags.
5. Snapshot and recall data are XML-escaped and fenced as untrusted data.
6. Candidate reminders use Pi's transient `context` transformation, never a persisted fake user message.
7. Tool result counts and output bytes are capped.
8. Memory is advisory; hard safety rules require tool enforcement.

## Pi lifecycle

| Event | Action |
|---|---|
| `session_start` | close stale resources, load config/stores, migrate legacy data, start bounded session backfill |
| `before_agent_start` | append policy and stable strongest-memory snapshot |
| `message_end(user)` | asynchronously extract a generation-guarded candidate |
| `context` | inject pending candidates once, transiently |
| tool execution | mutate, auto-save, refresh snapshot |
| `agent_settled` | periodic consolidation, forgetting, and flush |
| `resources_discover` | expose global/current-project SurMem skills |
| `session_shutdown` | await backfill, index final session, save, checkpoint, close models/databases |

Session generation guards prevent late judge results from an old `/new`, `/resume`, `/fork`, or reload lifecycle from reaching a replacement session.

The `activeMemory` setting (default `true`) adds concise proactive remember/forget guidance in `before_agent_start`. The main agent still decides and invokes existing memory tools; the setting adds no background writes or persistence bypass. Legacy `experimentalActiveMemory` values are read when `activeMemory` is absent, preserving explicit opt-outs, and subsequent config saves use only the stable name.

## Shared model daemons

The default extension uses two cross-process loopback services:

1. `embedding-daemon`: one loaded EmbeddingGemma model for every Pi process.
2. `judgment-daemon`: one loaded Qwen3 model serving both candidate judgment and contradiction arbitration.

Both bind to `127.0.0.1` on an ephemeral port and require a private bearer token. Startup uses an exclusive lock, PID/health validation, atomic endpoint/state files, proxy-aware model download, and model-fingerprint fencing. Client HTTP requests disable keep-alive so they do not prevent one-shot Pi processes from exiting. Session shutdown disconnects clients but intentionally does not stop a daemon used by other Pi processes. Each daemon exits after its idle TTL.

On macOS, local GGUF daemons default to Metal so `node-llama-cpp` can use its supported prebuilt binary instead
of requiring a local C++ build. Other platforms retain the CPU default unless GPU selection is configured.
Default model downloads use Hugging Face except when the local system timezone identifies mainland China,
Hong Kong, or Macau, where equivalent ModelScope GGUF URLs are used. This local-only signal avoids IP
geolocation; `SURMEM_MODEL_SOURCE` provides an explicit `auto`, `huggingface`, or `modelscope` override.

## Competitive design choices

Compared with plain Markdown + qmd, SurMem removes the external search/index dependency and gives records stable IDs, vectors, revisions, provenance, strength, and tombstones.

Compared with Markdown + SQLite mirroring, SurMem keeps one authoritative structured memory store, avoiding source/mirror drift. Pi-native skill files and conversation FTS remain separate because they are different data types with different consumers.

SurMem intentionally avoids silent background memory writes. This is less automatic than some systems, but it reduces durable hallucination and prompt-pollution risk. Shared local GGUF judgment improves candidate extraction and contradiction handling without remote token cost or per-Pi duplicate model instances; heuristic fallback remains available when the daemon is unavailable.

## Scale

Memory recall currently scans active vectors in memory. This is appropriate for curated stores up to tens of thousands of records. Session history uses SQLite FTS5 and is incrementally indexed by file size/mtime. A future large-scale backend can implement `Persister` and a retrieval adapter without changing the write policy.
