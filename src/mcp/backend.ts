/** Scoped MCP operations. Durability, scanning, and write decisions remain in the core. */
import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { ValidationError } from "../errors";
import { loadExtensionConfig } from "../extension-config";
import { SurpriseMemory } from "../index";
import { JsonPersister, SqlitePersister, STORE_SCHEMA_VERSION } from "../persistence";
import type { MemoryRecord, MemoryScope } from "../types";
import { createArbiter, createEmbedders, type McpConfig } from "./config";

export class McpBackend {
  private readonly stores = new Map<MemoryScope, SurpriseMemory>();
  private readonly arbiters: Array<ReturnType<typeof createArbiter>> = [];
  private tail: Promise<unknown> = Promise.resolve();
  private closing = false;
  private closePromise?: Promise<void>;
  readonly shutdown = new AbortController();

  constructor(readonly config: McpConfig) {}

  /** Serial execution also makes reload-before-operation safe for shared host stores. */
  run<T>(operation: () => Promise<T>, signal: AbortSignal): Promise<T> {
    if (this.closing) return Promise.reject(new ValidationError("MCP server is shutting down."));
    const result = this.tail.then(() => {
      signal.throwIfAborted();
      this.shutdown.signal.throwIfAborted();
      return operation();
    });
    // The request handler reports failures; a rejected request must not poison the queue.
    this.tail = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  async memory(scope: MemoryScope): Promise<SurpriseMemory> {
    let memory = this.stores.get(scope);
    if (!memory) {
      const config = await loadExtensionConfig(this.config.configPath);
      const embeddings = createEmbedders(this.config.storageDir);
      const arbiter = createArbiter(this.config.storageDir);
      const path =
        scope === "global"
          ? join(this.config.storageDir, "global.sqlite")
          : join(this.config.storageDir, "projects", `${this.config.projectKey}.sqlite`);
      const persister = new SqlitePersister(path);
      try {
        memory = new SurpriseMemory({
          embedder: embeddings.document,
          queryEmbedder: embeddings.query,
          gate: {
            tauAdd: config.tauAdd,
            dupSim: config.dupSim,
            conflictSim: config.conflictSim,
            minTokens: config.minTokens,
            judge: arbiter,
          },
          store: {
            persister,
            decayRatePerHour: config.decayRatePerHour,
            semanticDecayRatePerHour: config.semanticDecayRatePerHour,
            forgetThreshold: config.forgetThreshold,
          },
          retrieval: { maxResults: 10 },
          autoSave: true,
          reindexOnEmbeddingChange: "lazy",
        });
        await memory.load();
      } catch (error) {
        // Close the persister directly: never save a partially loaded/corrupt store.
        await persister.close();
        await embeddings.document.dispose?.();
        if (embeddings.query !== embeddings.document) await embeddings.query.dispose?.();
        arbiter?.dispose?.();
        throw error;
      }
      this.stores.set(scope, memory);
      this.arbiters.push(arbiter);
    } else {
      // Writes auto-save before returning. Do not discard an unresolved failed write.
      if (memory.stats.dirty) await memory.save();
      await memory.load();
    }
    return memory;
  }

  assertNotSkill(record: MemoryRecord): void {
    if (record.metadata.origin === "surmem-skill") {
      throw new ValidationError(
        `Memory ${record.id} backs a Pi skill. Manage it with the Pi adapter; MCP cannot modify skill files.`,
      );
    }
  }

  private recoveryPath(scope: MemoryScope, recoveryId: string): string {
    return join(
      this.config.storageDir,
      "mcp-recovery",
      scope === "global" ? "global" : this.config.projectKey,
      `${recoveryId}.json`,
    );
  }

  private async snapshot(path: string, memory: SurpriseMemory, records: MemoryRecord[]): Promise<void> {
    await new JsonPersister(path).save(
      {
        schemaVersion: STORE_SCHEMA_VERSION,
        revision: 0,
        embeddingFingerprint: memory.stats.reindexRequired
          ? memory.store.persistedEmbeddingFingerprint
          : memory.embedder.fingerprint,
        updatedAt: Date.now() / 1000,
        records,
        tombstones: [],
      },
      0,
    );
  }

  async forget(scope: MemoryScope, id: string) {
    const memory = await this.memory(scope);
    const record = memory.store.get(id);
    if (!record) throw new ValidationError(`Memory ${id} was not found in ${scope} scope.`);
    this.assertNotSkill(record);
    const recoveryId = randomUUID();
    await this.snapshot(this.recoveryPath(scope, recoveryId), memory, [record]);
    await memory.forget(id);
    return { id, scope, recoveryId };
  }

  async restore(scope: MemoryScope, recoveryId: string, signal: AbortSignal) {
    const path = this.recoveryPath(scope, recoveryId);
    const snapshot = await new JsonPersister(path).load();
    if (snapshot?.records.length !== 1) throw new ValidationError(`Recovery not found or invalid: ${path}`);
    const record = snapshot.records[0];
    if (record.metadata.scope !== scope) throw new ValidationError(`Recovery scope mismatch: ${path}`);
    this.assertNotSkill(record);
    const memory = await this.memory(scope);
    if (memory.store.get(record.id))
      throw new ValidationError(`Memory ${record.id} already exists; recovery cannot overwrite it.`);
    const restored = await memory.restore(record, signal);
    return { id: restored.id, scope, restored: true };
  }

  async export(scope: MemoryScope, memory?: SurpriseMemory) {
    const current = memory ?? (await this.memory(scope));
    const records = current.export().records;
    const path = join(this.config.storageDir, "exports", `mcp-${scope}-${randomUUID()}.json`);
    await this.snapshot(path, current, records);
    return {
      path,
      scope,
      format: "surmem-snapshot",
      schemaVersion: STORE_SCHEMA_VERSION,
      count: records.length,
    };
  }

  async clear(scope: MemoryScope) {
    const memory = await this.memory(scope);
    for (const record of memory.store.all()) this.assertNotSkill(record);
    const backup = await this.export(scope, memory);
    const count = await memory.clear();
    return { scope, count, backup };
  }

  close(): Promise<void> {
    if (this.closePromise) return this.closePromise;
    this.closing = true;
    this.shutdown.abort(new Error("MCP server is shutting down."));
    this.closePromise = (async () => {
      await this.tail;
      const results = await Promise.allSettled([...this.stores.values()].map((memory) => memory.close()));
      for (const arbiter of this.arbiters) arbiter?.dispose?.();
      const failures = results.filter(
        (result): result is PromiseRejectedResult => result.status === "rejected",
      );
      if (failures.length)
        throw new AggregateError(
          failures.map((result) => result.reason),
          "Failed to close SurMem stores.",
        );
    })();
    return this.closePromise;
  }
}
