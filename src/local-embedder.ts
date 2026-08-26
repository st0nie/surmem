/**
 * GgufEmbedder: fully local embeddings via node-llama-cpp and a GGUF model.
 *
 * Designed for the same models qmd uses (auto-cached under ~/.cache/qmd/models):
 *   - embeddinggemma-300M (768-dim, English-optimized)
 *   - Qwen3-Embedding-0.6B (1024-dim, multilingual incl. CJK)
 *
 * node-llama-cpp is an optional dependency: it is only imported when this
 * class is instantiated, so the core library stays dependency-free.
 *
 * Model-specific prompt formats matter for quality. embeddinggemma expects:
 *   documents: "title: none | text: {text}"
 *   queries:   "task: search result | query: {text}"
 *
 * Use `createPair()` to get document/query variants sharing one loaded model.
 */

import type { Embedder } from "./embeddings";

export interface GgufEmbedderOptions {
  /** Absolute path to the GGUF model file. */
  modelPath: string;
  /** Model output dimension (768 for embeddinggemma-300M, 1024 for Qwen3-0.6B). */
  dim: number;
  /**
   * Prompt template for documents (memories being stored).
   * Must contain "{text}". Default matches embeddinggemma's document format.
   */
  documentTemplate?: string;
  /**
   * Prompt template for queries (recall). Must contain "{text}".
   * Default matches embeddinggemma's query format.
   */
  queryTemplate?: string;
}

type Role = "document" | "query";

interface LlamaEmbeddingContext {
  getEmbeddingFor(text: string): Promise<{ vector: readonly number[] | Float32Array }>;
  dispose?(): Promise<void> | void;
}

interface SharedState {
  ctxPromise: Promise<LlamaEmbeddingContext> | null;
}

export class GgufEmbedder implements Embedder {
  readonly dim: number;
  private readonly modelPath: string;
  private readonly documentTemplate: string;
  private readonly queryTemplate: string;
  private readonly role: Role;
  private readonly shared: SharedState;

  constructor(opts: GgufEmbedderOptions, role: Role = "document", shared?: SharedState) {
    this.modelPath = opts.modelPath;
    this.dim = opts.dim;
    this.documentTemplate = opts.documentTemplate ?? "title: none | text: {text}";
    this.queryTemplate = opts.queryTemplate ?? "task: search result | query: {text}";
    this.role = role;
    this.shared = shared ?? { ctxPromise: null };
  }

  /**
   * Create document/query embedder variants that share one loaded model.
   * Pass the document embedder for writes and the query embedder for recalls.
   */
  static createPair(opts: GgufEmbedderOptions): { document: Embedder; query: Embedder } {
    const shared: SharedState = { ctxPromise: null };
    return {
      document: new GgufEmbedder(opts, "document", shared),
      query: new GgufEmbedder(opts, "query", shared),
    };
  }

  private async context(): Promise<LlamaEmbeddingContext> {
    if (!this.shared.ctxPromise) {
      this.shared.ctxPromise = (async () => {
        const { getLlama } = await import("node-llama-cpp");
        const llama = await getLlama();
        const model = await llama.loadModel({ modelPath: this.modelPath });
        return (await model.createEmbeddingContext()) as unknown as LlamaEmbeddingContext;
      })();
    }
    return this.shared.ctxPromise;
  }

  async embed(texts: string[]): Promise<number[][]> {
    const ctx = await this.context();
    const template = this.role === "query" ? this.queryTemplate : this.documentTemplate;
    const out: number[][] = [];
    for (const text of texts) {
      const prompt = template.replace("{text}", text);
      const emb = await ctx.getEmbeddingFor(prompt);
      out.push(Array.from(emb.vector as readonly number[]));
    }
    return out;
  }

  async dispose(): Promise<void> {
    if (this.shared.ctxPromise) {
      const ctx = await this.shared.ctxPromise;
      await ctx.dispose?.();
      this.shared.ctxPromise = null;
    }
  }
}
