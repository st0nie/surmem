/**
 * Consolidator: turns clusters of episodic memories into semantic memories.
 *
 * Process (memory consolidation, inspired by hippocampal replay):
 *   1. Greedily cluster active episodic memories by embedding similarity.
 *   2. Clusters with >= minClusterSize members are consolidated.
 *   3. An optional summarizer (LLM) distills each cluster into one fact;
 *      without one, the highest-strength member is promoted as representative.
 *   4. The new semantic memory links back to its episodic sources, which are
 *      kept but marked so their decay can proceed independently.
 */

import { cosine, type Embedder } from "./embeddings";
import type { MemoryStore } from "./store";
import type { MemoryRecord } from "./types";
import { createRecord, Kind } from "./types";

/** Optional LLM summarizer for distilling a cluster into one semantic fact. */
export interface Summarizer {
  summarize(texts: string[]): string | Promise<string>;
}

export interface ConsolidationOptions {
  clusterSim?: number; // similarity to join a cluster (default 0.6)
  minClusterSize?: number; // minimum cluster size to consolidate (default 2)
  summarizer?: Summarizer;
}

export interface ConsolidationResult {
  created: MemoryRecord[];
  clusters: number;
}

export class Consolidator {
  private readonly clusterSim: number;
  private readonly minClusterSize: number;
  private readonly summarizer?: Summarizer;

  constructor(
    private store: MemoryStore,
    private embedder: Embedder,
    opts: ConsolidationOptions = {},
  ) {
    this.clusterSim = opts.clusterSim ?? 0.6;
    this.minClusterSize = opts.minClusterSize ?? 2;
    this.summarizer = opts.summarizer;
  }

  async consolidate(): Promise<ConsolidationResult> {
    const episodic = this.store.byKind(Kind.EPISODIC);
    const visited = new Set<string>();
    const clusters: MemoryRecord[][] = [];

    // Greedy single-pass clustering.
    for (const rec of episodic) {
      if (visited.has(rec.id)) continue;
      const cluster = [rec];
      visited.add(rec.id);
      for (const other of episodic) {
        if (visited.has(other.id)) continue;
        if (cosine(rec.vector, other.vector) >= this.clusterSim) {
          cluster.push(other);
          visited.add(other.id);
        }
      }
      if (cluster.length >= this.minClusterSize) clusters.push(cluster);
    }

    const created: MemoryRecord[] = [];
    for (const cluster of clusters) {
      const texts = cluster.map((m) => m.text);
      const text = this.summarizer
        ? await this.summarizer.summarize(texts)
        : // No LLM: promote the strongest member as the representative fact.
          cluster.reduce((a, b) =>
            this.store.effectiveStrength(a) >= this.store.effectiveStrength(b) ? a : b,
          ).text;

      const [vector] = await this.embedder.embed([text]);
      const maxStrength = Math.max(...cluster.map((m) => m.baseStrength));
      const rec = createRecord({
        text,
        vector,
        kind: Kind.SEMANTIC,
        // Semantic memories start stronger and decay slower in practice
        // because consolidation keeps re-touching their sources.
        baseStrength: Math.min(2.0, maxStrength + 0.5),
        surpriseAtWrite: maxStrength,
        sourceIds: cluster.map((m) => m.id),
        metadata: { consolidatedFrom: cluster.length },
      });
      this.store.add(rec);
      created.push(rec);
    }

    return { created, clusters: clusters.length };
  }
}
