/** Fully local GGUF embeddings via lazy node-llama-cpp loading. */

import { createHash } from "node:crypto";
import type { Embedder } from "./embeddings";
import { validateVector } from "./embeddings";
import { ValidationError } from "./errors";

export interface GgufEmbedderOptions {
  modelPath: string;
  dim: number;
  gpu?: "auto" | "metal" | "cuda" | "vulkan" | false;
  documentTemplate?: string;
  queryTemplate?: string;
}

type Role = "document" | "query";
interface Disposable {
  dispose?(): Promise<void> | void;
}
interface LlamaEmbeddingContext extends Disposable {
  getEmbeddingFor(text: string): Promise<{ vector: readonly number[] | Float32Array }>;
}
interface SharedState {
  ctxPromise: Promise<LlamaEmbeddingContext> | null;
  model: Disposable | null;
  llama: Disposable | null;
}

function configuredGpu(explicit?: GgufEmbedderOptions["gpu"]): GgufEmbedderOptions["gpu"] {
  const value = explicit ?? process.env.SURMEM_GGUF_GPU ?? false;
  if (value === false || value === "auto" || value === "metal" || value === "cuda" || value === "vulkan")
    return value;
  throw new ValidationError(`Unsupported SURMEM_GGUF_GPU value: ${value}`);
}

export class GgufEmbedder implements Embedder {
  readonly dim: number;
  readonly fingerprint: string;
  private readonly modelPath: string;
  private readonly gpu: GgufEmbedderOptions["gpu"];
  private readonly documentTemplate: string;
  private readonly queryTemplate: string;
  private readonly role: Role;
  private readonly shared: SharedState;

  constructor(opts: GgufEmbedderOptions, role: Role = "document", shared?: SharedState) {
    if (!opts.modelPath) throw new ValidationError("GgufEmbedder modelPath is required.");
    if (!Number.isInteger(opts.dim) || opts.dim < 1 || opts.dim > 65_536)
      throw new ValidationError("GgufEmbedder dim is invalid.");
    this.modelPath = opts.modelPath;
    this.dim = opts.dim;
    this.gpu = configuredGpu(opts.gpu);
    this.documentTemplate = opts.documentTemplate ?? "title: none | text: {text}";
    this.queryTemplate = opts.queryTemplate ?? "task: search result | query: {text}";
    if (!this.documentTemplate.includes("{text}") || !this.queryTemplate.includes("{text}")) {
      throw new ValidationError("GGUF embedding templates must contain {text}.");
    }
    this.role = role;
    this.shared = shared ?? { ctxPromise: null, model: null, llama: null };
    const identity = createHash("sha256")
      .update(`${this.modelPath}\0${this.dim}\0${this.documentTemplate}\0${this.queryTemplate}`)
      .digest("hex")
      .slice(0, 20);
    this.fingerprint = `gguf:${identity}:${this.dim}`;
  }

  static createPair(opts: GgufEmbedderOptions): { document: GgufEmbedder; query: GgufEmbedder } {
    const shared: SharedState = { ctxPromise: null, model: null, llama: null };
    return {
      document: new GgufEmbedder(opts, "document", shared),
      query: new GgufEmbedder(opts, "query", shared),
    };
  }

  private async context(): Promise<LlamaEmbeddingContext> {
    if (!this.shared.ctxPromise) {
      this.shared.ctxPromise = (async () => {
        const { getLlama } = await import("node-llama-cpp");
        const llama = await getLlama({ gpu: this.gpu });
        const model = await llama.loadModel({ modelPath: this.modelPath });
        this.shared.llama = llama as unknown as Disposable;
        this.shared.model = model as unknown as Disposable;
        return (await model.createEmbeddingContext()) as unknown as LlamaEmbeddingContext;
      })().catch((error) => {
        this.shared.ctxPromise = null;
        throw error;
      });
    }
    return this.shared.ctxPromise;
  }

  async embed(texts: string[], signal?: AbortSignal): Promise<number[][]> {
    if (texts.length === 0) return [];
    const context = await this.context();
    const template = this.role === "query" ? this.queryTemplate : this.documentTemplate;
    const output: number[][] = [];
    for (const text of texts) {
      signal?.throwIfAborted();
      const embedding = await context.getEmbeddingFor(template.replace("{text}", text));
      output.push(validateVector(embedding.vector, this.dim, "GGUF embedding"));
    }
    return output;
  }

  async dispose(): Promise<void> {
    const context = this.shared.ctxPromise ? await this.shared.ctxPromise.catch(() => null) : null;
    await context?.dispose?.();
    await this.shared.model?.dispose?.();
    await this.shared.llama?.dispose?.();
    this.shared.ctxPromise = null;
    this.shared.model = null;
    this.shared.llama = null;
  }
}
