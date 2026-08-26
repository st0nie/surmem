/** Data structures: memory records and write-gate verdicts. */

export enum Kind {
  EPISODIC = "episodic", // raw events, subject to decay
  SEMANTIC = "semantic", // consolidated facts, stable and long-lived
  PROCEDURAL = "procedural", // skills / workflows (reserved)
}

export enum WriteVerdict {
  ADD = "ADD", // high surprise: store as new memory
  UPDATE = "UPDATE", // conflicts with an existing memory: supersede it
  REINFORCE = "REINFORCE", // repeated content: strengthen existing memory
  NOOP = "NOOP", // low-value routine: discard
}

export interface MemoryRecord {
  id: string;
  text: string;
  vector: number[];
  kind: Kind;
  createdAt: number; // epoch seconds
  lastAccessed: number; // epoch seconds
  baseStrength: number; // initial strength, driven by surprise at write
  accessCount: number; // retrieval/reinforcement hits (spacing effect)
  surpriseAtWrite: number; // surprise score when written (for auditing)
  supersededBy: string | null; // id of the UPDATE memory that replaced this one
  sourceIds: string[]; // episodic sources of a semantic memory
  metadata: Record<string, unknown>;
}

export function createRecord(
  partial: Pick<MemoryRecord, "text" | "vector"> & Partial<MemoryRecord>,
): MemoryRecord {
  const now = Date.now() / 1000;
  return {
    id: crypto.randomUUID().slice(0, 12),
    kind: Kind.EPISODIC,
    createdAt: now,
    lastAccessed: now,
    baseStrength: 1.0,
    accessCount: 0,
    surpriseAtWrite: 0,
    supersededBy: null,
    sourceIds: [],
    metadata: {},
    ...partial,
  };
}

export function touch(rec: MemoryRecord): void {
  rec.lastAccessed = Date.now() / 1000;
  rec.accessCount += 1;
}
