/** In-memory cognitive store with validated decay and conflict-safe persistence. */

import { PersistenceConflictError, ValidationError } from "./errors";
import {
  JsonPersister,
  type PersistenceHealth,
  type PersistenceSnapshot,
  type Persister,
  STORE_SCHEMA_VERSION,
} from "./persistence";
import type { MemoryRecord, MemoryTombstone } from "./types";
import { Kind, nowSeconds } from "./types";

export interface StoreOptions {
  decayRatePerHour?: number;
  semanticDecayRatePerHour?: number;
  forgetThreshold?: number;
  forgetSemantic?: boolean;
  persistPath?: string;
  persister?: Persister;
  embeddingFingerprint?: string;
}

function nonNegative(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) throw new ValidationError(`${name} must be non-negative.`);
  return value;
}

function mergeSnapshots(local: PersistenceSnapshot, remote: PersistenceSnapshot): PersistenceSnapshot {
  const records = new Map<string, MemoryRecord>();
  for (const record of [...remote.records, ...local.records]) {
    const existing = records.get(record.id);
    if (!existing || record.updatedAt >= existing.updatedAt) records.set(record.id, record);
  }
  const tombstones = new Map<string, MemoryTombstone>();
  for (const deleted of [...remote.tombstones, ...local.tombstones]) {
    const existing = tombstones.get(deleted.id);
    if (!existing || deleted.deletedAt > existing.deletedAt) tombstones.set(deleted.id, deleted);
  }
  for (const [id, deleted] of [...tombstones]) {
    const record = records.get(id);
    const restoredAt = Number(record?.metadata.restoredAt ?? 0);
    if (record && Number.isFinite(restoredAt) && restoredAt > deleted.deletedAt) tombstones.delete(id);
    else records.delete(id);
  }
  return {
    schemaVersion: STORE_SCHEMA_VERSION,
    revision: remote.revision,
    embeddingFingerprint: local.embeddingFingerprint,
    updatedAt: Math.max(local.updatedAt, remote.updatedAt),
    records: [...records.values()],
    tombstones: [...tombstones.values()],
  };
}

export class MemoryStore {
  private records = new Map<string, MemoryRecord>();
  private tombstones = new Map<string, MemoryTombstone>();
  private decayRate: number;
  private semanticDecayRate: number;
  private forgetThreshold: number;
  private forgetSemantic: boolean;
  private readonly persister?: Persister;
  private revision = 0;
  private fingerprint: string;
  private loadedFingerprint = "";
  private mutationTail: Promise<void> = Promise.resolve();
  private dirty = false;

  constructor(opts: StoreOptions = {}) {
    this.decayRate = nonNegative(opts.decayRatePerHour ?? 0.02, "decayRatePerHour");
    this.semanticDecayRate = nonNegative(opts.semanticDecayRatePerHour ?? 0.002, "semanticDecayRatePerHour");
    this.forgetThreshold = nonNegative(opts.forgetThreshold ?? 0.1, "forgetThreshold");
    this.forgetSemantic = opts.forgetSemantic ?? false;
    this.persister = opts.persister ?? (opts.persistPath ? new JsonPersister(opts.persistPath) : undefined);
    this.fingerprint = opts.embeddingFingerprint ?? "unknown";
  }

  get config() {
    return {
      decayRatePerHour: this.decayRate,
      semanticDecayRatePerHour: this.semanticDecayRate,
      forgetThreshold: this.forgetThreshold,
      forgetSemantic: this.forgetSemantic,
    };
  }

  get embeddingFingerprint(): string {
    return this.fingerprint;
  }
  get persistedEmbeddingFingerprint(): string {
    return this.loadedFingerprint;
  }
  get isDirty(): boolean {
    return this.dirty;
  }
  get currentRevision(): number {
    return this.revision;
  }

  configure(
    opts: Partial<
      Pick<
        StoreOptions,
        "decayRatePerHour" | "semanticDecayRatePerHour" | "forgetThreshold" | "forgetSemantic"
      >
    >,
  ): void {
    if (opts.decayRatePerHour !== undefined)
      this.decayRate = nonNegative(opts.decayRatePerHour, "decayRatePerHour");
    if (opts.semanticDecayRatePerHour !== undefined)
      this.semanticDecayRate = nonNegative(opts.semanticDecayRatePerHour, "semanticDecayRatePerHour");
    if (opts.forgetThreshold !== undefined)
      this.forgetThreshold = nonNegative(opts.forgetThreshold, "forgetThreshold");
    if (opts.forgetSemantic !== undefined) this.forgetSemantic = opts.forgetSemantic;
  }

  setEmbeddingFingerprint(value: string): void {
    if (!value) throw new ValidationError("Embedding fingerprint must not be empty.");
    this.fingerprint = value;
    this.dirty = true;
  }

  async exclusive<T>(operation: () => Promise<T> | T): Promise<T> {
    const previous = this.mutationTail;
    let release!: () => void;
    this.mutationTail = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }

  add(record: MemoryRecord): void {
    const deleted = this.tombstones.get(record.id);
    if (deleted && deleted.deletedAt >= record.updatedAt) return;
    this.records.set(record.id, record);
    this.tombstones.delete(record.id);
    this.dirty = true;
  }

  /** Explicit user-authorized undelete; unlike add(), this clears a tombstone. */
  restore(record: MemoryRecord): void {
    this.tombstones.delete(record.id);
    this.records.set(record.id, record);
    this.dirty = true;
  }

  get(id: string): MemoryRecord | undefined {
    return this.records.get(id);
  }
  all(): MemoryRecord[] {
    return [...this.records.values()];
  }
  active(): MemoryRecord[] {
    return this.all().filter((memory) => memory.supersededBy === null);
  }
  byKind(kind: Kind): MemoryRecord[] {
    return this.active().filter((memory) => memory.kind === kind);
  }

  remove(id: string, deletedAt = nowSeconds()): MemoryRecord | null {
    const record = this.records.get(id) ?? null;
    if (!record) return null;
    this.records.delete(id);
    this.tombstones.set(id, { id, deletedAt });
    this.dirty = true;
    return record;
  }

  clear(): number {
    const ids = [...this.records.keys()];
    const now = nowSeconds();
    for (const id of ids) this.remove(id, now);
    return ids.length;
  }

  markDirty(record?: MemoryRecord): void {
    if (record) record.updatedAt = nowSeconds();
    this.dirty = true;
  }

  effectiveStrength(record: MemoryRecord, nowS = nowSeconds()): number {
    const hours = Math.max(0, (nowS - record.lastAccessed) / 3600);
    const spacing = 1 + Math.log1p(record.accessCount);
    const rate = record.kind === Kind.EPISODIC ? this.decayRate : this.semanticDecayRate;
    return record.baseStrength * spacing * Math.exp(-rate * hours);
  }

  forgetPass(nowS = nowSeconds()): MemoryRecord[] {
    const forgotten: MemoryRecord[] = [];
    for (const record of this.active()) {
      const eligible =
        record.kind === Kind.EPISODIC || (this.forgetSemantic && record.kind === Kind.SEMANTIC);
      if (eligible && this.effectiveStrength(record, nowS) < this.forgetThreshold) {
        const removed = this.remove(record.id, nowS);
        if (removed) forgotten.push(removed);
      }
    }
    return forgotten;
  }

  private snapshot(): PersistenceSnapshot {
    const cutoff = nowSeconds() - 30 * 24 * 3600;
    const tombstones = [...this.tombstones.values()].filter((item) => item.deletedAt >= cutoff);
    return {
      schemaVersion: STORE_SCHEMA_VERSION,
      revision: this.revision,
      embeddingFingerprint: this.fingerprint,
      updatedAt: nowSeconds(),
      records: this.all(),
      tombstones,
    };
  }

  private applySnapshot(snapshot: PersistenceSnapshot): void {
    this.records = new Map(snapshot.records.map((record) => [record.id, record]));
    this.tombstones = new Map(snapshot.tombstones.map((item) => [item.id, item]));
    for (const deleted of [...this.tombstones.values()]) {
      const record = this.records.get(deleted.id);
      const restoredAt = Number(record?.metadata.restoredAt ?? 0);
      if (record && Number.isFinite(restoredAt) && restoredAt > deleted.deletedAt)
        this.tombstones.delete(deleted.id);
      else this.records.delete(deleted.id);
    }
    this.revision = snapshot.revision;
    this.loadedFingerprint = snapshot.embeddingFingerprint;
  }

  async load(): Promise<void> {
    const persister = this.persister;
    if (!persister) return;
    await this.exclusive(async () => {
      const snapshot = await persister.load();
      if (snapshot) this.applySnapshot(snapshot);
      this.dirty = false;
    });
  }

  async save(force = false): Promise<void> {
    const persister = this.persister;
    if (!persister || (!this.dirty && !force)) return;
    await this.exclusive(async () => {
      let local = this.snapshot();
      for (let attempt = 0; attempt < 4; attempt++) {
        try {
          this.revision = await persister.save(local, this.revision);
          this.dirty = false;
          return;
        } catch (error) {
          if (!(error instanceof PersistenceConflictError) || attempt === 3) throw error;
          const remote = await persister.load();
          if (!remote) throw error;
          local = mergeSnapshots(local, remote);
          this.applySnapshot(local);
          this.revision = remote.revision;
          this.dirty = true;
        }
      }
    });
  }

  async health(): Promise<PersistenceHealth> {
    const base = await this.persister?.health?.();
    return {
      ...(base ?? { backend: this.persister ? "custom" : "memory" }),
      revision: this.revision,
      schemaVersion: STORE_SCHEMA_VERSION,
    };
  }

  async close(): Promise<void> {
    await this.save();
    await this.persister?.close?.();
  }
}
