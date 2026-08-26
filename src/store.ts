/**
 * Memory stores: episodic + semantic layers with Ebbinghaus-style decay,
 * JSON persistence, and strength-based pruning (forgetting).
 *
 * Effective strength:
 *   S(t) = baseStrength * (1 + log(1 + accessCount)) * exp(-lambda * hoursSince(lastAccessed))
 *
 * The spacing-effect term rewards memories that are repeatedly reinforced,
 * while the exponential decay forgets what is never touched again.
 */

import type { MemoryRecord } from "./types";
import { Kind } from "./types";
import { JsonPersister, type Persister } from "./persistence";

export interface StoreOptions {
  decayRatePerHour?: number; // lambda for episodic memory (default 0.02: ~34h half-life)
  semanticDecayRatePerHour?: number; // lambda for semantic memory (default 0.002: ~2 weeks half-life)
  forgetThreshold?: number; // effective strength below this -> forget (default 0.1)
  persistPath?: string; // JSON file path; convenience shortcut for JsonPersister
  persister?: Persister; // custom backend (e.g. SqlitePersister); takes precedence over persistPath
}

export class MemoryStore {
  private records = new Map<string, MemoryRecord>();
  private decayRate: number;
  private semanticDecayRate: number;
  private forgetThreshold: number;
  private readonly persister?: Persister;

  constructor(opts: StoreOptions = {}) {
    this.decayRate = opts.decayRatePerHour ?? 0.02;
    this.semanticDecayRate = opts.semanticDecayRatePerHour ?? 0.002;
    this.forgetThreshold = opts.forgetThreshold ?? 0.1;
    this.persister =
      opts.persister ?? (opts.persistPath ? new JsonPersister(opts.persistPath) : undefined);
  }

  /** Current store configuration. */
  get config() {
    return {
      decayRatePerHour: this.decayRate,
      semanticDecayRatePerHour: this.semanticDecayRate,
      forgetThreshold: this.forgetThreshold,
    };
  }

  /** Update decay/forget parameters at runtime. */
  configure(
    opts: Partial<Pick<StoreOptions, "decayRatePerHour" | "semanticDecayRatePerHour" | "forgetThreshold">>,
  ): void {
    if (opts.decayRatePerHour !== undefined) this.decayRate = opts.decayRatePerHour;
    if (opts.semanticDecayRatePerHour !== undefined) this.semanticDecayRate = opts.semanticDecayRatePerHour;
    if (opts.forgetThreshold !== undefined) this.forgetThreshold = opts.forgetThreshold;
  }

  add(rec: MemoryRecord): void {
    this.records.set(rec.id, rec);
  }

  get(id: string): MemoryRecord | undefined {
    return this.records.get(id);
  }

  all(): MemoryRecord[] {
    return [...this.records.values()];
  }

  active(): MemoryRecord[] {
    return this.all().filter((m) => m.supersededBy === null);
  }

  byKind(kind: Kind): MemoryRecord[] {
    return this.active().filter((m) => m.kind === kind);
  }

  /** Ebbinghaus-style effective strength with a spacing-effect bonus.
   *  Semantic memories decay ~10x slower: consolidation grants permanence. */
  effectiveStrength(rec: MemoryRecord, nowS = Date.now() / 1000): number {
    const hours = Math.max(0, (nowS - rec.lastAccessed) / 3600);
    const spacing = 1 + Math.log1p(rec.accessCount);
    const rate = rec.kind === Kind.EPISODIC ? this.decayRate : this.semanticDecayRate;
    return rec.baseStrength * spacing * Math.exp(-rate * hours);
  }

  /** Prune records whose effective strength fell below the forget threshold. */
  forgetPass(nowS = Date.now() / 1000): MemoryRecord[] {
    const forgotten: MemoryRecord[] = [];
    for (const rec of this.active()) {
      if (
        rec.kind === Kind.EPISODIC &&
        this.effectiveStrength(rec, nowS) < this.forgetThreshold
      ) {
        forgotten.push(rec);
        this.records.delete(rec.id);
      }
    }
    return forgotten;
  }

  async save(): Promise<void> {
    await this.persister?.save(this.all());
  }

  async load(): Promise<void> {
    if (!this.persister) return;
    const data = await this.persister.load();
    for (const rec of data) this.records.set(rec.id, rec);
  }
}
