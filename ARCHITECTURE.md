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

all Pi processes ─┬─▶ one shared EmbeddingGemma daemon
                  └─▶ one shared Qwen3 judgment/arbitration daemon
```

## Scope model

The extension owns two independent stores:

1. **Global**: user preferences and facts valid across projects.
2. **Project**: repository decisions, conventions, and procedures.

Project identity is the first 20 hex characters of the SHA-256 of the canonical real path. Basename-only identity is not safe because unrelated repositories can share a name.

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
similarity >= dupSim
  → REINFORCE

conflictSim <= similarity < dupSim and a judge is configured
  → accept only a strict verdict with sufficient confidence;
    an uncertain judge falls back to the surprise check below

momentum-adjusted surprise > tauAdd
  → ADD

otherwise
  → NOOP (the result reports the nearest blocking memory)
```

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

## Shared model daemons

The default extension uses two cross-process loopback services:

1. `embedding-daemon`: one loaded EmbeddingGemma model for every Pi process.
2. `judgment-daemon`: one loaded Qwen3 model serving both candidate judgment and contradiction arbitration.

Both bind to `127.0.0.1` on an ephemeral port and require a private bearer token. Startup uses an exclusive lock, PID/health validation, atomic endpoint/state files, proxy-aware model download, and model-fingerprint fencing. Client HTTP requests disable keep-alive so they do not prevent one-shot Pi processes from exiting. Session shutdown disconnects clients but intentionally does not stop a daemon used by other Pi processes. Each daemon exits after its idle TTL.

## Competitive design choices

Compared with plain Markdown + qmd, SurMem removes the external search/index dependency and gives records stable IDs, vectors, revisions, provenance, strength, and tombstones.

Compared with Markdown + SQLite mirroring, SurMem keeps one authoritative structured memory store, avoiding source/mirror drift. Pi-native skill files and conversation FTS remain separate because they are different data types with different consumers.

SurMem intentionally avoids silent background memory writes. This is less automatic than some systems, but it reduces durable hallucination and prompt-pollution risk. Shared local GGUF judgment improves candidate extraction and contradiction handling without remote token cost or per-Pi duplicate model instances; heuristic fallback remains available when the daemon is unavailable.

## Scale

Memory recall currently scans active vectors in memory. This is appropriate for curated stores up to tens of thousands of records. Session history uses SQLite FTS5 and is incrementally indexed by file size/mtime. A future large-scale backend can implement `Persister` and a retrieval adapter without changing the write policy.
