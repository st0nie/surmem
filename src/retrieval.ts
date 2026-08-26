/** Native hybrid retrieval: vector + lexical + recency + strength. */

import { cosine, tokenize } from "./embeddings";
import { ValidationError } from "./errors";
import type { MemoryStore } from "./store";
import type { MemoryRecord, MemoryScope } from "./types";
import { touch } from "./types";

export interface RetrievalOptions {
  wVector?: number;
  wLexical?: number;
  wRecency?: number;
  wStrength?: number;
  recencyHalfLifeHours?: number;
  minScore?: number;
  maxResults?: number;
}

export interface RecallFilter {
  scope?: MemoryScope;
  project?: string;
  kind?: MemoryRecord["kind"];
  reinforce?: boolean;
}

export interface ScoredMemory {
  record: MemoryRecord;
  score: number;
  signals: { vector: number; lexical: number; recency: number; strength: number };
}

function nonNegative(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) throw new ValidationError(`${name} must be non-negative.`);
  return value;
}

function lexicalScore(query: string, text: string): number {
  const queryTerms = new Set(tokenize(query));
  if (queryTerms.size === 0) return 0;
  const textTerms = new Set(tokenize(text));
  let overlap = 0;
  for (const term of queryTerms) if (textTerms.has(term)) overlap++;
  return overlap / queryTerms.size;
}

export class Retriever {
  private readonly wVector: number;
  private readonly wLexical: number;
  private readonly wRecency: number;
  private readonly wStrength: number;
  private readonly recencyHalfLife: number;
  private readonly minScore: number;
  private readonly maxResults: number;

  constructor(
    private readonly store: MemoryStore,
    opts: RetrievalOptions = {},
  ) {
    this.wVector = nonNegative(opts.wVector ?? 0.58, "wVector");
    this.wLexical = nonNegative(opts.wLexical ?? 0.22, "wLexical");
    this.wRecency = nonNegative(opts.wRecency ?? 0.08, "wRecency");
    this.wStrength = nonNegative(opts.wStrength ?? 0.12, "wStrength");
    this.recencyHalfLife = nonNegative(opts.recencyHalfLifeHours ?? 168, "recencyHalfLifeHours");
    this.minScore = nonNegative(opts.minScore ?? 0.08, "minScore");
    this.maxResults = Math.max(1, Math.min(100, Math.floor(opts.maxResults ?? 20)));
  }

  retrieve(query: string, queryVector: number[], k = 5, filter: RecallFilter = {}): ScoredMemory[] {
    const limit = Math.max(1, Math.min(this.maxResults, Math.floor(k)));
    const now = Date.now() / 1000;
    const candidates = this.store.active().filter((record) => {
      if (filter.kind && record.kind !== filter.kind) return false;
      if (filter.scope && record.metadata.scope !== filter.scope) return false;
      if (filter.project && record.metadata.project !== filter.project) return false;
      return true;
    });
    const scored = candidates
      .map((record) => {
        const vector = Math.max(0, cosine(queryVector, record.vector));
        const lexical = lexicalScore(query, record.text);
        const hours = Math.max(0, (now - record.lastAccessed) / 3600);
        const recency = this.recencyHalfLife === 0 ? 0 : Math.exp((-Math.LN2 * hours) / this.recencyHalfLife);
        const rawStrength = this.store.effectiveStrength(record, now);
        const strength = rawStrength / (1 + rawStrength);
        const weight = this.wVector + this.wLexical + this.wRecency + this.wStrength || 1;
        const score =
          (this.wVector * vector +
            this.wLexical * lexical +
            this.wRecency * recency +
            this.wStrength * strength) /
          weight;
        return { record, score, signals: { vector, lexical, recency, strength } };
      })
      .filter((item) => item.score >= this.minScore);
    scored.sort(
      (a, b) =>
        b.score - a.score ||
        b.record.updatedAt - a.record.updatedAt ||
        a.record.id.localeCompare(b.record.id),
    );
    const top = scored.slice(0, limit);
    if (filter.reinforce !== false) {
      for (const { record } of top) {
        touch(record);
        this.store.markDirty(record);
      }
    }
    return top;
  }
}
