/**
 * SurpriseMemory: the main framework class, wiring together the surprise gate,
 * memory stores, retrieval, and consolidation.
 *
 * Agent integration is two calls:
 *   await mem.observe(text)   // after each agent event (write path, gated)
 *   mem.recall(query)         // before each turn (read path, for context)
 */

import { HashEmbedder, type Embedder } from "./embeddings";
import { SurpriseGate, type GateOptions } from "./gate";
import { MemoryStore, type StoreOptions } from "./store";
import { Retriever, type RetrievalOptions, type ScoredMemory } from "./retrieval";
import {
  Consolidator,
  type ConsolidationOptions,
  type ConsolidationResult,
} from "./consolidation";
import {
  createRecord,
  touch,
  type MemoryRecord,
  WriteVerdict,
} from "./types";

export interface SurMemOptions {
  embedder?: Embedder;
  gate?: GateOptions;
  store?: StoreOptions;
  retrieval?: RetrievalOptions;
  consolidation?: ConsolidationOptions;
}

export interface ObserveResult {
  verdict: WriteVerdict;
  surprise: number;
  record: MemoryRecord | null; // new/updated/reinforced record, or null on NOOP
  superseded: MemoryRecord | null;
}

export class SurpriseMemory {
  readonly embedder: Embedder;
  readonly store: MemoryStore;
  private readonly gate: SurpriseGate;
  private readonly retriever: Retriever;
  private readonly consolidator: Consolidator;

  constructor(opts: SurMemOptions = {}) {
    this.embedder = opts.embedder ?? new HashEmbedder();
    this.store = new MemoryStore(opts.store);
    this.gate = new SurpriseGate(opts.gate);
    this.retriever = new Retriever(this.store, opts.retrieval);
    this.consolidator = new Consolidator(this.store, this.embedder, opts.consolidation);
  }

  /** Load persisted memories (if store.persistPath is set). */
  async load(): Promise<void> {
    await this.store.load();
  }

  /**
   * Write path: feed one agent observation through the surprise gate.
   * Returns what the gate decided to do with it.
   */
  async observe(
    text: string,
    metadata: Record<string, unknown> = {},
  ): Promise<ObserveResult> {
    const [vector] = await this.embedder.embed([text]);
    const { verdict, surprise, nearest } = await this.gate.decide(
      text,
      vector,
      this.store.all(),
    );

    switch (verdict) {
      case WriteVerdict.ADD: {
        const rec = createRecord({
          text,
          vector,
          metadata,
          surpriseAtWrite: surprise,
          baseStrength: 0.5 + surprise, // high surprise -> stronger initial trace
        });
        this.store.add(rec);
        return { verdict, surprise, record: rec, superseded: null };
      }

      case WriteVerdict.UPDATE: {
        const rec = createRecord({
          text,
          vector,
          metadata,
          surpriseAtWrite: surprise,
          baseStrength: 0.5 + surprise,
          sourceIds: nearest ? [nearest.id] : [],
        });
        this.store.add(rec);
        if (nearest) nearest.supersededBy = rec.id; // old fact superseded
        return { verdict, surprise, record: rec, superseded: nearest };
      }

      case WriteVerdict.REINFORCE: {
        if (nearest) {
          touch(nearest);
          nearest.baseStrength += 0.2 * (1 - surprise); // spacing effect
        }
        return { verdict, surprise, record: nearest, superseded: null };
      }

      case WriteVerdict.NOOP:
        return { verdict, surprise, record: null, superseded: null };
    }
  }

  /** Read path: hybrid-scored top-k memories for context injection. */
  async recall(query: string, k = 5): Promise<ScoredMemory[]> {
    const [vector] = await this.embedder.embed([query]);
    return this.retriever.retrieve(vector, k);
  }

  /** Convenience: render recalled memories as a context block for prompts. */
  async recallAsContext(query: string, k = 5): Promise<string> {
    const hits = await this.recall(query, k);
    if (hits.length === 0) return "";
    const lines = hits.map(
      ({ record, score }) =>
        `- [${record.kind} | score=${score.toFixed(2)}] ${record.text}`,
    );
    return "## Relevant memories\n" + lines.join("\n");
  }

  /** Consolidate episodic clusters into semantic memories. */
  async reflect(): Promise<ConsolidationResult> {
    return this.consolidator.consolidate();
  }

  /** Forget memories whose effective strength decayed below threshold. */
  forgetPass(): MemoryRecord[] {
    return this.store.forgetPass();
  }

  /** Persist memories (if store.persistPath is set). */
  async save(): Promise<void> {
    await this.store.save();
  }
}

export { WriteVerdict, Kind } from "./types";
export type { MemoryRecord } from "./types";
export type { LLMJudge } from "./gate";
export type { Summarizer } from "./consolidation";
export type { ScoredMemory } from "./retrieval";
export { HashEmbedder, OpenAIEmbedder } from "./embeddings";
export type { Embedder } from "./embeddings";
export { OpenAIJudge, OpenAISummarizer } from "./judge";
export { JsonPersister, SqlitePersister } from "./persistence";
export type { Persister } from "./persistence";
