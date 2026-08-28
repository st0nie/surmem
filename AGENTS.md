# AGENTS.md

Guidance for coding agents and contributors working on Pi SurMem.

## Project purpose

SurMem is a production-grade long-term memory package for Pi. It includes:

- an agent-agnostic TypeScript core under `src/`,
- the Pi extension at `extensions/surmem/index.ts`,
- deterministic tests under `tests/`,
- shared cross-process GGUF embedding and judgment daemons through lazy `node-llama-cpp` imports.

Read `ARCHITECTURE.md` before changing storage, retrieval, lifecycle, or security behavior.

## Required commands

Run these before declaring work complete:

```bash
bun install
bun run check       # core + extension typecheck, Biome, all tests
bun run pack:check  # published file-set verification
```

Useful targeted commands:

```bash
bun test tests/production.test.ts
bun test tests/extension.test.ts
bun test tests/session-index.test.ts
bun x tsc --noEmit
bun x tsc -p extensions/surmem/tsconfig.json
bun run demo
```

The GGUF smoke test is optional because it requires a local model:

```bash
SURMEM_GGUF_MODEL_PATH=/path/model.gguf bun run smoke:gguf
```

## Architecture boundaries

- `src/index.ts`: public facade and exports.
- `src/store.ts`: in-process record/tombstone state and persistence coordination.
- `src/persistence.ts`: versioned JSON/SQLite persistence and legacy migrations.
- `src/gate.ts`: surprise write policy; similarity alone must not imply contradiction.
- `src/retrieval.ts`: native vector/lexical/recency/strength ranking.
- `src/consolidation.ts`: idempotent episodic-to-semantic consolidation.
- `src/embeddings.ts`, `src/local-embedder.ts`: hash/OpenAI/GGUF embedders; GGUF imports stay lazy.
- `src/daemon-embedder.ts`, `src/daemon-judge.ts`: clients for the shared cross-process embedding and judgment daemons (entries: `src/embedding-daemon-entry.mjs`, `src/judgment-daemon-entry.mjs`).
- `src/judge.ts`: memorability/conflict judges and summarizers.
- `src/errors.ts`, `src/types.ts`: typed errors and shared record types.
- `src/safety.ts`: durable-write scanner and prompt sanitization.
- `src/session-index.ts`: conversation FTS index; it must not index tool-result payloads.
- `src/extension-config.ts`: strict, bounded, atomic extension config.
- `extensions/surmem/index.ts`: Pi lifecycle, tools, scoped stores, transient candidates, skills.

Keep core logic independent of Pi. Pi APIs belong only in `extensions/`.

## Non-negotiable invariants

### Persistence

- Never treat malformed or unreadable durable data as an empty store.
- Preserve corrupt source files and return path-bearing errors.
- Keep schema versions and backward migrations explicit.
- JSON writes require lock + revision check + atomic rename.
- SQLite writes require WAL + transaction + revision check.
- Deletions require tombstones so stale writers cannot resurrect records.
- Explicit restore must remain distinguishable from a stale update.
- New files containing memory/config/export data must use private permissions.
- Do not make session shutdown the only durability boundary.

### Embeddings

- Every embedder must expose a stable `fingerprint` and fixed `dim`.
- Validate vector count, dimensions, finiteness, and non-zero norm.
- Never mix vectors from different fingerprints; reindex or fail explicitly.
- Keep GGUF imports lazy; model loading belongs only in detached singleton daemons.
- Multiple Pi processes must reuse one embedding daemon and one shared judgment/arbitration daemon.
- Client shutdown must not stop a daemon still usable by other Pi processes; daemon idle TTL owns model disposal.

### Safety

- Scan every durable memory and skill write before persistence.
- Do not persist credentials, private keys, tokens, injection payloads, or invisible Unicode.
- Treat recalled/snapshotted memory as untrusted historical data, not instructions.
- XML-escape prompt-injected data and strip dangerous tags/control characters.
- Candidate reminders must use Pi's transient `context` hook; do not fake persisted user messages.
- Bound tool output, top-k, text size, metadata size, config size, and candidate count.

### Memory semantics

- Without a trusted arbiter, related facts must not destructively `UPDATE` one another.
- `REINFORCE` updates retrieval/access strength but does not create duplicates.
- Superseded records are retained for audit but excluded from active recall.
- Consolidation must remain idempotent and preserve source IDs.
- Global and project memories must not leak across scopes.

### Pi lifecycle

- Guard late async candidate/judge results with a session generation.
- Await background session indexing before closing the database.
- Use `agent_settled`, not `agent_end`, for periodic maintenance.
- Extension failures may degrade memory features but must not break the main agent loop.
- Configuration errors should be visible and actionable, not silently reset.

## Testing expectations

Add or update deterministic tests for every behavior change. Production-sensitive changes should cover the relevant failure mode:

- concurrent writers,
- stale deletion/resurrection,
- legacy migration,
- corrupt input,
- private file permissions,
- embedding fingerprint changes,
- secret/injection rejection,
- scope isolation,
- transient context behavior,
- session index incrementality and query fallback,
- resource close/reopen behavior.

No test may require network access, API keys, qmd, or a GGUF file. Put real-model checks in the optional smoke script.

## Code style

- TypeScript strict mode is mandatory.
- Use Node built-ins with `node:` specifiers.
- Prefer typed errors from `src/errors.ts` at public/data boundaries.
- Avoid broad `catch { return [] }` or `catch {}` around authoritative data operations.
- Best-effort catches are acceptable only for cleanup or explicitly optional features; record or surface diagnostics when useful.
- Keep output and error messages actionable and include relevant paths/IDs without leaking secrets.
- Run Biome rather than manually fighting formatting.

## Packaging

The repository root is the Pi package root. Do not recreate a nested extension `package.json` or lockfile.

Runtime Pi packages belong in `peerDependencies`; optional GGUF support belongs in `optionalDependencies`. Keep the `files` allowlist small and verify it with `bun run pack:check`. The tarball must not contain any `node_modules` directory, test fixtures, local databases, or generated task state.

## Documentation

Update `README.md` when user-facing tools, paths, environment variables, config, migration, security behavior, or limitations change. Update `ARCHITECTURE.md` when invariants or subsystem boundaries change. Keep competitive claims precise and versioned; do not claim stronger behavior than tests and implementation prove.
