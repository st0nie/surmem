/** Idempotent connected-component consolidation: episodic traces -> semantic facts. */

import { cosine, type Embedder, validateVector } from "./embeddings";
import { ValidationError } from "./errors";
import { scanMemoryContent } from "./safety";
import type { MemoryStore } from "./store";
import type { MemoryRecord } from "./types";
import { createRecord, Kind, touch } from "./types";

export interface Summarizer {
  summarize(texts: string[], signal?: AbortSignal): string | Promise<string>;
}
export interface ConsolidationOptions {
  clusterSim?: number;
  minClusterSize?: number;
  dedupSim?: number;
  summarizer?: Summarizer;
}
export interface ConsolidationResult {
  created: MemoryRecord[];
  reinforced: MemoryRecord[];
  clusters: number;
}

function similarity(value: number, name: string): number {
  if (!Number.isFinite(value) || value < -1 || value > 1)
    throw new ValidationError(`${name} must be between -1 and 1.`);
  return value;
}

export class Consolidator {
  private readonly clusterSim: number;
  private readonly minClusterSize: number;
  private readonly dedupSim: number;
  private readonly summarizer?: Summarizer;

  constructor(
    private readonly store: MemoryStore,
    private readonly embedder: Embedder,
    opts: ConsolidationOptions = {},
  ) {
    this.clusterSim = similarity(opts.clusterSim ?? 0.6, "clusterSim");
    this.minClusterSize = Math.max(2, Math.floor(opts.minClusterSize ?? 2));
    this.dedupSim = similarity(opts.dedupSim ?? 0.85, "dedupSim");
    this.summarizer = opts.summarizer;
  }

  private components(memories: MemoryRecord[]): MemoryRecord[][] {
    const seen = new Set<string>();
    const output: MemoryRecord[][] = [];
    for (const start of memories) {
      if (seen.has(start.id)) continue;
      const component: MemoryRecord[] = [];
      const queue = [start];
      seen.add(start.id);
      while (queue.length) {
        const current = queue.shift();
        if (!current) break;
        component.push(current);
        for (const candidate of memories) {
          if (!seen.has(candidate.id) && cosine(current.vector, candidate.vector) >= this.clusterSim) {
            seen.add(candidate.id);
            queue.push(candidate);
          }
        }
      }
      if (component.length >= this.minClusterSize) output.push(component);
    }
    return output;
  }

  async consolidate(signal?: AbortSignal): Promise<ConsolidationResult> {
    const episodic = this.store
      .byKind(Kind.EPISODIC)
      .filter((memory) => memory.metadata.consolidatedInto === undefined);
    const clusters = this.components(episodic);
    const created: MemoryRecord[] = [];
    const reinforced: MemoryRecord[] = [];

    for (const cluster of clusters) {
      signal?.throwIfAborted();
      const texts = cluster.map((memory) => memory.text);
      const representative = cluster.reduce((left, right) =>
        this.store.effectiveStrength(left) >= this.store.effectiveStrength(right) ? left : right,
      );
      const text = (
        this.summarizer ? await this.summarizer.summarize(texts, signal) : representative.text
      ).trim();
      if (!text || text.length > 20_000 || scanMemoryContent(text).length > 0) continue;
      const [rawVector] = await this.embedder.embed([text], signal);
      const vector = validateVector(rawVector, this.embedder.dim, "consolidated embedding");
      const semantic = this.store.byKind(Kind.SEMANTIC);
      let nearest: MemoryRecord | null = null;
      let nearestSimilarity = -1;
      for (const memory of semantic) {
        const value = cosine(vector, memory.vector);
        if (value > nearestSimilarity) {
          nearest = memory;
          nearestSimilarity = value;
        }
      }
      if (nearest && nearestSimilarity >= this.dedupSim) {
        touch(nearest, 0.3);
        this.store.markDirty(nearest);
        for (const source of cluster) {
          source.metadata.consolidatedInto = nearest.id;
          this.store.markDirty(source);
        }
        reinforced.push(nearest);
        continue;
      }
      const maxStrength = Math.max(...cluster.map((memory) => memory.baseStrength));
      const record = createRecord({
        text,
        vector,
        kind: Kind.SEMANTIC,
        baseStrength: Math.min(2, maxStrength + 0.5),
        surpriseAtWrite: Math.max(...cluster.map((memory) => memory.surpriseAtWrite)),
        sourceIds: cluster.map((memory) => memory.id),
        metadata: {
          consolidatedFrom: cluster.length,
          scope: representative.metadata.scope,
          project: representative.metadata.project,
          embeddingFingerprint: this.embedder.fingerprint,
        },
      });
      this.store.add(record);
      for (const source of cluster) {
        source.metadata.consolidatedInto = record.id;
        this.store.markDirty(source);
      }
      created.push(record);
    }
    return { created, reinforced, clusters: clusters.length };
  }
}
