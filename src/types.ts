/** Data structures: memory records, scopes, and write-gate verdicts. */

import { randomUUID } from "node:crypto";

export enum Kind {
  EPISODIC = "episodic",
  SEMANTIC = "semantic",
  PROCEDURAL = "procedural",
}

export type MemoryScope = "global" | "project";

export enum WriteVerdict {
  ADD = "ADD",
  UPDATE = "UPDATE",
  REINFORCE = "REINFORCE",
  NOOP = "NOOP",
}

export interface MemoryRecord {
  id: string;
  text: string;
  vector: number[];
  kind: Kind;
  createdAt: number;
  updatedAt: number;
  lastAccessed: number;
  baseStrength: number;
  accessCount: number;
  surpriseAtWrite: number;
  supersededBy: string | null;
  sourceIds: string[];
  metadata: Record<string, unknown>;
}

export interface MemoryTombstone {
  id: string;
  deletedAt: number;
}

export function nowSeconds(): number {
  return Date.now() / 1000;
}

export function createRecord(
  partial: Pick<MemoryRecord, "text" | "vector"> & Partial<MemoryRecord>,
): MemoryRecord {
  const now = nowSeconds();
  return {
    id: randomUUID(),
    kind: Kind.EPISODIC,
    createdAt: now,
    updatedAt: now,
    lastAccessed: now,
    baseStrength: 1,
    accessCount: 0,
    surpriseAtWrite: 0,
    supersededBy: null,
    sourceIds: [],
    metadata: {},
    ...partial,
  };
}

export function touch(rec: MemoryRecord, strengthDelta = 0): void {
  const now = nowSeconds();
  rec.lastAccessed = now;
  rec.updatedAt = now;
  rec.accessCount += 1;
  rec.baseStrength += strengthDelta;
}
