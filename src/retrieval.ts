/**
 * Hybrid-scoring retrieval: relevance + recency + effective strength.
 *
 *   score = wRel * cosine(query, memory)
 *         + wRec * recencyBoost(lastAccessed)
 *         + wStr * effectiveStrength(memory)
 *
 * Retrieved memories are "touched", which both refreshes their recency and
 * strengthens them (retrieval practice effect).
 */

import { cosine } from "./embeddings";
import type { MemoryStore } from "./store";
import type { MemoryRecord } from "./types";
import { touch } from "./types";

export interface RetrievalOptions {
  wRel?: number; // default 1.0
  wRec?: number; // default 0.2
  wStr?: number; // default 0.3
  recencyHalfLifeHours?: number; // default 24
}

export interface ScoredMemory {
  record: MemoryRecord;
  score: number;
}

export class Retriever {
  private readonly wRel: number;
  private readonly wRec: number;
  private readonly wStr: number;
  private readonly recencyHalfLife: number;

  constructor(
    private store: MemoryStore,
    opts: RetrievalOptions = {},
  ) {
    this.wRel = opts.wRel ?? 1.0;
    this.wRec = opts.wRec ?? 0.2;
    this.wStr = opts.wStr ?? 0.3;
    this.recencyHalfLife = opts.recencyHalfLifeHours ?? 24;
  }

  retrieve(queryVector: number[], k = 5): ScoredMemory[] {
    const nowS = Date.now() / 1000;
    const scored = this.store.active().map((rec) => {
      const relevance = Math.max(0, cosine(queryVector, rec.vector));
      const hoursAgo = Math.max(0, (nowS - rec.lastAccessed) / 3600);
      const recency = Math.exp((-Math.LN2 * hoursAgo) / this.recencyHalfLife);
      const strength = this.store.effectiveStrength(rec, nowS);
      const score =
        this.wRel * relevance + this.wRec * recency + this.wStr * strength;
      return { record: rec, score };
    });

    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, k);
    for (const { record } of top) touch(record); // retrieval practice
    return top;
  }
}
