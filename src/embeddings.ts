/**
 * Embedding backends: pluggable interface plus two implementations.
 *
 * - HashEmbedder: zero-dependency hashing embedder, works offline, good for
 *   demos and as a fallback.
 * - OpenAIEmbedder: production backend against any OpenAI-compatible
 *   /embeddings endpoint (OpenAI, Azure, vLLM, Ollama, etc.).
 *
 * embed() may be sync or async; callers must always `await` it.
 */

export interface Embedder {
  readonly dim: number;
  embed(texts: string[]): number[][] | Promise<number[][]>;
}

const TOKEN_RE = /[A-Za-z0-9_]+|[一-鿿]/g;

/** Tokenize text (words for Latin scripts, characters for CJK). */
export function tokenize(text: string): string[] {
  return text.toLowerCase().match(TOKEN_RE) ?? [];
}

/** Count of meaningful tokens; used by the gate's triviality filter. */
export function tokenCount(text: string): number {
  return tokenize(text).length;
}

/** FNV-1a 64-bit hash; no external dependency required. */
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

/** Zero-dependency hashing embedder (bag of words + bigrams, L2-normalized). */
export class HashEmbedder implements Embedder {
  readonly dim: number;

  constructor(dim = 512) {
    this.dim = dim;
  }

  private tokens(text: string): string[] {
    const toks = tokenize(text);
    const feats = [...toks];
    for (let i = 0; i < toks.length - 1; i++) feats.push(toks[i] + "~" + toks[i + 1]);
    return feats;
  }

  embed(texts: string[]): number[][] {
    return texts.map((text) => {
      const vec = new Array<number>(this.dim).fill(0);
      for (const tok of this.tokens(text)) {
        const h = fnv1a(tok);
        const idx = Number(h % BigInt(this.dim));
        const sign = (h >> 63n) === 0n ? 1 : -1;
        vec[idx] += sign;
      }
      const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
      return vec.map((v) => v / norm);
    });
  }
}

export interface OpenAIEmbedderOptions {
  apiKey: string;
  model?: string; // default "text-embedding-3-small"
  baseUrl?: string; // default "https://api.openai.com/v1"
  dim?: number; // must match the model output dimension
}

/** Production embedder for any OpenAI-compatible embeddings endpoint. */
export class OpenAIEmbedder implements Embedder {
  readonly dim: number;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl: string;

  constructor(opts: OpenAIEmbedderOptions) {
    this.apiKey = opts.apiKey;
    this.model = opts.model ?? "text-embedding-3-small";
    this.baseUrl = (opts.baseUrl ?? "https://api.openai.com/v1").replace(/\/$/, "");
    this.dim = opts.dim ?? 1536;
  }

  async embed(texts: string[]): Promise<number[][]> {
    const res = await fetch(`${this.baseUrl}/embeddings`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ model: this.model, input: texts }),
    });
    if (!res.ok) {
      throw new Error(`Embedding request failed: ${res.status} ${await res.text()}`);
    }
    const payload = (await res.json()) as {
      data: Array<{ embedding: number[]; index: number }>;
    };
    return payload.data
      .sort((a, b) => a.index - b.index)
      .map((d) => d.embedding);
  }
}

/** Cosine similarity, robust to unnormalized inputs (e.g. raw GGUF vectors). */
export function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}
