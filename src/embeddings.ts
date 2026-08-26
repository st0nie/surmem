/** Pluggable embedding backends and validated similarity helpers. */

import { ValidationError } from "./errors";

export interface Embedder {
  readonly dim: number;
  readonly fingerprint: string;
  embed(texts: string[], signal?: AbortSignal): number[][] | Promise<number[][]>;
  dispose?(): Promise<void> | void;
}

const TOKEN_RE = /[\p{L}\p{N}_]+/gu;

export function tokenize(text: string): string[] {
  const raw = text.normalize("NFKC").toLowerCase().match(TOKEN_RE) ?? [];
  const output: string[] = [];
  for (const token of raw) {
    if (/^[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+$/u.test(token)) {
      const chars = [...token];
      output.push(...chars);
      for (let index = 0; index < chars.length - 1; index++) output.push(chars[index] + chars[index + 1]);
    } else {
      output.push(token);
    }
  }
  return output;
}

export function tokenCount(text: string): number {
  return tokenize(text).length;
}

function fnv1a(s: string): bigint {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let i = 0; i < s.length; i++) {
    h ^= BigInt(s.charCodeAt(i));
    h = (h * prime) & mask;
  }
  return h;
}

export class HashEmbedder implements Embedder {
  readonly fingerprint: string;
  constructor(readonly dim = 512) {
    if (!Number.isInteger(dim) || dim < 16 || dim > 65_536)
      throw new ValidationError("HashEmbedder dim must be an integer from 16 to 65536.");
    this.fingerprint = `hash-fnv1a-bigram:v2:${dim}`;
  }

  private features(text: string): string[] {
    const tokens = tokenize(text);
    const features = [...tokens];
    for (let i = 0; i < tokens.length - 1; i++) features.push(`${tokens[i]}~${tokens[i + 1]}`);
    return features;
  }

  embed(texts: string[]): number[][] {
    return texts.map((text) => {
      const vector = new Array<number>(this.dim).fill(0);
      for (const token of this.features(text)) {
        const hash = fnv1a(token);
        const index = Number(hash % BigInt(this.dim));
        vector[index] += hash >> 63n === 0n ? 1 : -1;
      }
      const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
      return vector.map((value) => value / norm);
    });
  }
}

export interface OpenAIEmbedderOptions {
  apiKey: string;
  model?: string;
  baseUrl?: string;
  dim?: number;
  timeoutMs?: number;
  maxRetries?: number;
}

function abortSignal(
  parent: AbortSignal | undefined,
  timeoutMs: number,
): { signal: AbortSignal; dispose: () => void } {
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(new Error(`Request timed out after ${timeoutMs}ms.`)),
    timeoutMs,
  );
  const onAbort = () => controller.abort(parent?.reason);
  parent?.addEventListener("abort", onAbort, { once: true });
  return {
    signal: controller.signal,
    dispose: () => {
      clearTimeout(timer);
      parent?.removeEventListener("abort", onAbort);
    },
  };
}

export class OpenAIEmbedder implements Embedder {
  readonly dim: number;
  readonly fingerprint: string;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;

  constructor(opts: OpenAIEmbedderOptions) {
    if (!opts.apiKey) throw new ValidationError("OpenAIEmbedder apiKey is required.");
    this.apiKey = opts.apiKey;
    this.model = opts.model ?? "text-embedding-3-small";
    this.baseUrl = (opts.baseUrl ?? "https://api.openai.com/v1").replace(/\/$/, "");
    this.dim = opts.dim ?? 1536;
    this.timeoutMs = opts.timeoutMs ?? 30_000;
    this.maxRetries = opts.maxRetries ?? 2;
    if (!Number.isInteger(this.dim) || this.dim < 1 || this.dim > 65_536)
      throw new ValidationError("OpenAIEmbedder dim is invalid.");
    this.fingerprint = `openai:${this.baseUrl}:${this.model}:${this.dim}`;
  }

  async embed(texts: string[], signal?: AbortSignal): Promise<number[][]> {
    if (texts.length === 0) return [];
    let lastError: unknown;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      const scoped = abortSignal(signal, this.timeoutMs);
      try {
        const response = await fetch(`${this.baseUrl}/embeddings`, {
          method: "POST",
          headers: { "content-type": "application/json", authorization: `Bearer ${this.apiKey}` },
          body: JSON.stringify({ model: this.model, input: texts }),
          signal: scoped.signal,
        });
        if (!response.ok) {
          const body = (await response.text()).slice(0, 2000);
          const error = new Error(`Embedding request failed: ${response.status} ${body}`);
          if (response.status < 500 && response.status !== 429) throw error;
          lastError = error;
        } else {
          const payload = (await response.json()) as {
            data?: Array<{ embedding?: unknown; index?: unknown }>;
          };
          if (!Array.isArray(payload.data) || payload.data.length !== texts.length)
            throw new ValidationError("Embedding response has an invalid data array.");
          const vectors = [...payload.data]
            .sort((a, b) => Number(a.index) - Number(b.index))
            .map((entry, index) => validateVector(entry.embedding, this.dim, `embedding[${index}]`));
          return vectors;
        }
      } catch (error) {
        if (error instanceof ValidationError || (signal?.aborted ?? false)) throw error;
        lastError = error;
      } finally {
        scoped.dispose();
      }
      if (attempt < this.maxRetries) await new Promise((resolve) => setTimeout(resolve, 200 * 2 ** attempt));
    }
    throw new Error("Embedding request failed after retries.", { cause: lastError });
  }
}

export function validateVector(value: unknown, dim: number, label = "vector"): number[] {
  if (!Array.isArray(value) && !(value instanceof Float32Array) && !(value instanceof Float64Array)) {
    throw new ValidationError(`${label} is not an array.`);
  }
  const vector = Array.from(value as ArrayLike<number>);
  if (vector.length !== dim || !vector.every(Number.isFinite)) {
    throw new ValidationError(`${label} must contain exactly ${dim} finite numbers.`);
  }
  const norm = Math.sqrt(vector.reduce((sum, item) => sum + item * item, 0));
  if (norm === 0) throw new ValidationError(`${label} must not be a zero vector.`);
  return vector;
}

export function cosine(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : Math.max(-1, Math.min(1, dot / denominator));
}
