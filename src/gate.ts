/**
 * SurpriseGate: the write-side policy of SurMem.
 *
 * Inspired by Titans' surprise-driven memory updates, but computed externally so
 * it works with any agent and any LLM:
 *
 *   novelty = 1 - max cosine(new_embedding, existing_memories)
 *
 * Verdicts (by similarity to the nearest active memory):
 *   sim >= dupSim                 -> REINFORCE (near-duplicate: spacing effect)
 *   conflictSim <= sim < dupSim   -> UPDATE    (same topic, different content:
 *                                               supersede or judge-arbitrate)
 *   novelty > tauAdd              -> ADD       (genuinely new information)
 *   otherwise                     -> NOOP      (routine, low-value)
 *
 * A Titans-style momentum term accumulates novelty over a sliding window, so a
 * *stream* of mildly-novel events about one topic can cross the write threshold.
 * An optional LLMJudge arbitrates the conflict zone (paraphrase vs contradiction).
 */

import { cosine, tokenCount } from "./embeddings";
import type { MemoryRecord } from "./types";
import { WriteVerdict } from "./types";

/** Optional arbitrator for the conflict zone. */
export interface LLMJudge {
  /**
   * Given the new text and the most similar existing memory, return
   * "ADD" (new fact), "UPDATE" (contradicts/replaces), "REINFORCE" or "NOOP".
   */
  arbitrate(newText: string, nearestText: string): string | Promise<string>;
}

export interface GateOptions {
  tauAdd?: number; // momentum-adjusted novelty above this -> ADD (default 0.45)
  dupSim?: number; // similarity above this -> REINFORCE (default 0.85)
  conflictSim?: number; // similarity above this -> UPDATE zone (default 0.6)
  minTokens?: number; // observations shorter than this -> NOOP (default 3)
  momentumDecay?: number; // per-second decay of novelty momentum (default 0.8)
  momentumWindowS?: number; // sliding window for momentum (default 300)
  judge?: LLMJudge;
}

export interface GateDecision {
  verdict: WriteVerdict;
  surprise: number;
  nearest: MemoryRecord | null;
}

export class SurpriseGate {
  private readonly tauAdd: number;
  private readonly dupSim: number;
  private readonly conflictSim: number;
  private readonly minTokens: number;
  private readonly momentumDecay: number;
  private readonly momentumWindowS: number;
  private readonly judge?: LLMJudge;
  private noveltyWindow: Array<{ ts: number; novelty: number }> = [];

  constructor(opts: GateOptions = {}) {
    this.tauAdd = opts.tauAdd ?? 0.45;
    this.dupSim = opts.dupSim ?? 0.85;
    this.conflictSim = opts.conflictSim ?? 0.6;
    this.minTokens = opts.minTokens ?? 3;
    this.momentumDecay = opts.momentumDecay ?? 0.8;
    this.momentumWindowS = opts.momentumWindowS ?? 300;
    this.judge = opts.judge;
  }

  /** Titans-style momentum: surprise now + decayed surprise of recent past. */
  private momentum(novelty: number): number {
    const now = Date.now() / 1000;
    this.noveltyWindow = this.noveltyWindow.filter(
      (e) => now - e.ts <= this.momentumWindowS,
    );
    const past = this.noveltyWindow.reduce(
      (s, e) => s + e.novelty * this.momentumDecay ** Math.max(0, Math.floor(now - e.ts)),
      0,
    );
    this.noveltyWindow.push({ ts: now, novelty });
    // Normalize so a single isolated event returns roughly its own novelty.
    return Math.min(1, novelty + 0.25 * past);
  }

  async decide(
    text: string,
    vector: number[],
    existing: MemoryRecord[],
  ): Promise<GateDecision> {
    // Triviality filter: no meaningful content, no memory.
    if (tokenCount(text) < this.minTokens) {
      return { verdict: WriteVerdict.NOOP, surprise: 0, nearest: null };
    }

    const active = existing.filter((m) => m.supersededBy === null);
    if (active.length === 0) {
      return { verdict: WriteVerdict.ADD, surprise: 1.0, nearest: null };
    }

    const nearest = active.reduce((best, m) =>
      cosine(vector, m.vector) > cosine(vector, best.vector) ? m : best,
    );
    const sim = cosine(vector, nearest.vector);
    const novelty = this.momentum(1 - sim);

    if (sim >= this.dupSim) {
      return { verdict: WriteVerdict.REINFORCE, surprise: novelty, nearest };
    }

    // Conflict zone: same topic, different content. Without a judge we assume
    // the newer statement supersedes the older one (recency wins).
    if (sim >= this.conflictSim) {
      if (this.judge) {
        const v = await this.judge.arbitrate(text, nearest.text);
        return { verdict: v as WriteVerdict, surprise: novelty, nearest };
      }
      return { verdict: WriteVerdict.UPDATE, surprise: novelty, nearest };
    }

    if (novelty > this.tauAdd) {
      return { verdict: WriteVerdict.ADD, surprise: novelty, nearest };
    }

    return { verdict: WriteVerdict.NOOP, surprise: novelty, nearest };
  }
}
