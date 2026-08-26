/** Surprise-gated write policy with strict verdict validation. */

import { cosine, tokenCount } from "./embeddings";
import { ValidationError } from "./errors";
import type { MemoryRecord } from "./types";
import { WriteVerdict } from "./types";

export interface LLMJudgeDecision {
  verdict: WriteVerdict | string;
  confidence?: number;
  reason?: string;
}

export interface LLMJudge {
  arbitrate(
    newText: string,
    nearestText: string,
    signal?: AbortSignal,
  ): string | LLMJudgeDecision | Promise<string | LLMJudgeDecision>;
}

export interface GateOptions {
  tauAdd?: number;
  dupSim?: number;
  conflictSim?: number;
  minTokens?: number;
  momentumDecay?: number;
  momentumWindowS?: number;
  minJudgeConfidence?: number;
  judge?: LLMJudge;
}

export interface GateDecision {
  verdict: WriteVerdict;
  surprise: number;
  nearest: MemoryRecord | null;
  reason?: string;
}

function unitInterval(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0 || value > 1)
    throw new ValidationError(`${name} must be between 0 and 1.`);
  return value;
}

function positive(value: number, name: string, allowZero = false): number {
  if (!Number.isFinite(value) || (allowZero ? value < 0 : value <= 0))
    throw new ValidationError(`${name} must be ${allowZero ? "non-negative" : "positive"}.`);
  return value;
}

function strictVerdict(value: unknown): WriteVerdict | null {
  const raw =
    typeof value === "string"
      ? value
      : value && typeof value === "object" && "verdict" in value
        ? String((value as { verdict: unknown }).verdict)
        : "";
  return Object.values(WriteVerdict).includes(raw as WriteVerdict) ? (raw as WriteVerdict) : null;
}

export class SurpriseGate {
  private tauAdd: number;
  private dupSim: number;
  private conflictSim: number;
  private minTokens: number;
  private momentumDecay: number;
  private momentumWindowS: number;
  private minJudgeConfidence: number;
  private judge?: LLMJudge;
  private noveltyWindow: Array<{ ts: number; novelty: number }> = [];

  constructor(opts: GateOptions = {}) {
    this.tauAdd = unitInterval(opts.tauAdd ?? 0.45, "tauAdd");
    this.dupSim = unitInterval(opts.dupSim ?? 0.85, "dupSim");
    this.conflictSim = unitInterval(opts.conflictSim ?? 0.6, "conflictSim");
    this.minTokens = Math.floor(positive(opts.minTokens ?? 3, "minTokens"));
    this.momentumDecay = unitInterval(opts.momentumDecay ?? 0.8, "momentumDecay");
    this.momentumWindowS = positive(opts.momentumWindowS ?? 300, "momentumWindowS");
    this.minJudgeConfidence = unitInterval(opts.minJudgeConfidence ?? 0.65, "minJudgeConfidence");
    this.judge = opts.judge;
    this.assertThresholds();
  }

  private assertThresholds(): void {
    if (this.conflictSim >= this.dupSim) throw new ValidationError("conflictSim must be lower than dupSim.");
  }

  get config() {
    return {
      tauAdd: this.tauAdd,
      dupSim: this.dupSim,
      conflictSim: this.conflictSim,
      minTokens: this.minTokens,
      momentumDecay: this.momentumDecay,
      momentumWindowS: this.momentumWindowS,
      minJudgeConfidence: this.minJudgeConfidence,
    };
  }

  configure(opts: Partial<GateOptions>): void {
    const previous = this.config;
    try {
      if (opts.tauAdd !== undefined) this.tauAdd = unitInterval(opts.tauAdd, "tauAdd");
      if (opts.dupSim !== undefined) this.dupSim = unitInterval(opts.dupSim, "dupSim");
      if (opts.conflictSim !== undefined) this.conflictSim = unitInterval(opts.conflictSim, "conflictSim");
      if (opts.minTokens !== undefined) this.minTokens = Math.floor(positive(opts.minTokens, "minTokens"));
      if (opts.momentumDecay !== undefined)
        this.momentumDecay = unitInterval(opts.momentumDecay, "momentumDecay");
      if (opts.momentumWindowS !== undefined)
        this.momentumWindowS = positive(opts.momentumWindowS, "momentumWindowS");
      if (opts.minJudgeConfidence !== undefined)
        this.minJudgeConfidence = unitInterval(opts.minJudgeConfidence, "minJudgeConfidence");
      if (opts.judge !== undefined) this.judge = opts.judge;
      this.assertThresholds();
    } catch (error) {
      Object.assign(this, previous);
      throw error;
    }
  }

  private momentum(novelty: number): number {
    const now = Date.now() / 1000;
    this.noveltyWindow = this.noveltyWindow.filter((entry) => now - entry.ts <= this.momentumWindowS);
    const past = this.noveltyWindow.reduce(
      (sum, entry) => sum + entry.novelty * this.momentumDecay ** Math.max(0, Math.floor(now - entry.ts)),
      0,
    );
    this.noveltyWindow.push({ ts: now, novelty });
    return Math.min(1, novelty + 0.25 * past);
  }

  async decide(
    text: string,
    vector: number[],
    existing: MemoryRecord[],
    signal?: AbortSignal,
  ): Promise<GateDecision> {
    if (tokenCount(text) < this.minTokens)
      return { verdict: WriteVerdict.NOOP, surprise: 0, nearest: null, reason: "too-short" };
    const active = existing.filter((memory) => memory.supersededBy === null);
    if (active.length === 0)
      return { verdict: WriteVerdict.ADD, surprise: 1, nearest: null, reason: "first-memory" };

    let nearest = active[0];
    let similarity = cosine(vector, nearest.vector);
    for (const memory of active.slice(1)) {
      const candidate = cosine(vector, memory.vector);
      if (candidate > similarity) {
        nearest = memory;
        similarity = candidate;
      }
    }
    const surprise = this.momentum(Math.max(0, 1 - similarity));
    if (similarity >= this.dupSim)
      return { verdict: WriteVerdict.REINFORCE, surprise, nearest, reason: "near-duplicate" };

    if (similarity >= this.conflictSim && this.judge) {
      const raw = await this.judge.arbitrate(text, nearest.text, signal);
      const verdict = strictVerdict(raw);
      const confidence =
        typeof raw === "object" && raw && "confidence" in raw
          ? Number((raw as LLMJudgeDecision).confidence ?? 1)
          : 1;
      const reason =
        typeof raw === "object" && raw && "reason" in raw
          ? String((raw as LLMJudgeDecision).reason ?? "judge")
          : "judge";
      if (verdict && Number.isFinite(confidence) && confidence >= this.minJudgeConfidence) {
        return { verdict, surprise, nearest, reason };
      }
      return {
        verdict: surprise > this.tauAdd ? WriteVerdict.ADD : WriteVerdict.NOOP,
        surprise,
        nearest,
        reason: "uncertain-judge",
      };
    }

    // Similarity alone cannot prove a contradiction. Without a judge, preserve
    // related facts independently rather than destructively superseding one.
    if (surprise > this.tauAdd) return { verdict: WriteVerdict.ADD, surprise, nearest, reason: "novel" };
    return { verdict: WriteVerdict.NOOP, surprise, nearest, reason: "low-surprise" };
  }
}
