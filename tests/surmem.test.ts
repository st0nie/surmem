/** Unit tests for SurMem core: gate verdicts, decay, consolidation, persistence. */

import { describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { Kind, SqlitePersister, SurpriseMemory, WriteVerdict } from "../src/index";

function makeMem(extra: Record<string, unknown> = {}): SurpriseMemory {
  return new SurpriseMemory({
    gate: { tauAdd: 0.45, dupSim: 0.85, conflictSim: 0.55 },
    consolidation: { clusterSim: 0.3 },
    ...extra,
  });
}

describe("surprise gate", () => {
  test("novel information is ADDed", async () => {
    const mem = makeMem();
    const r = await mem.observe("The user moved from Beijing to Shanghai.");
    expect(r.verdict).toBe(WriteVerdict.ADD);
    expect(r.record).not.toBeNull();
    expect(mem.store.active()).toHaveLength(1);
  });

  test("exact duplicates REINFORCE the existing memory", async () => {
    const mem = makeMem();
    await mem.observe("The user loves spicy Sichuan food.");
    const before = mem.store.active()[0].accessCount;
    const r = await mem.observe("The user loves spicy Sichuan food.");
    expect(r.verdict).toBe(WriteVerdict.REINFORCE);
    expect(mem.store.active()).toHaveLength(1);
    expect(mem.store.active()[0].accessCount).toBe(before + 1);
    expect(mem.store.active()[0].baseStrength).toBeGreaterThan(1);
  });

  test("related facts are not destructively UPDATEd without a judge", async () => {
    const mem = makeMem();
    await mem.observe("The user just moved from Beijing to Shanghai.");
    const r = await mem.observe("The user moved from Beijing back to Shanghai again.");
    expect(r.verdict).not.toBe(WriteVerdict.UPDATE);
    expect(mem.store.all()[0].supersededBy).toBeNull();
  });

  test("a high-confidence judge can UPDATE and supersede an old memory", async () => {
    const mem = makeMem({
      gate: {
        tauAdd: 0.45,
        dupSim: 0.85,
        conflictSim: 0.55,
        judge: { arbitrate: async () => ({ verdict: "UPDATE", confidence: 0.95 }) },
      },
    });
    await mem.observe("The user just moved from Beijing to Shanghai.");
    const r = await mem.observe("The user moved from Beijing back to Shanghai again.");
    expect(r.verdict).toBe(WriteVerdict.UPDATE);
    expect(r.superseded).not.toBeNull();
    expect(mem.store.active().map((memory) => memory.text)).not.toContain(
      "The user just moved from Beijing to Shanghai.",
    );
  });

  test("trivial input is NOOPed", async () => {
    const mem = makeMem();
    const r = await mem.observe("ok");
    expect(r.verdict).toBe(WriteVerdict.NOOP);
    expect(mem.store.active()).toHaveLength(0);
  });

  test("an LLM judge can arbitrate the conflict zone", async () => {
    const mem = makeMem({
      gate: {
        tauAdd: 0.45,
        dupSim: 0.85,
        conflictSim: 0.55,
        judge: { arbitrate: async () => "REINFORCE" },
      },
    });
    await mem.observe("The user just moved from Beijing to Shanghai.");
    const r = await mem.observe("The user moved from Beijing back to Shanghai again.");
    expect(r.verdict).toBe(WriteVerdict.REINFORCE);
  });
});

describe("explicit supersede", () => {
  test("supersedes performs an UPDATE even when the gate would NOOP", async () => {
    const mem = makeMem();
    const first = await mem.observe("The user just moved from Beijing to Shanghai.");
    const r = await mem.observe("The user moved from Beijing back to Shanghai again.", {
      supersedes: first.record?.id,
    });
    expect(r.verdict).toBe(WriteVerdict.UPDATE);
    expect(r.reason).toBe("explicit-supersede");
    expect(r.superseded?.id).toBe(first.record?.id);
    expect(r.record?.sourceIds).toContain(first.record?.id);
    const old = mem.store.get(first.record?.id ?? "");
    expect(old?.supersededBy).toBe(r.record?.id);
    expect(mem.store.active().map((memory) => memory.text)).not.toContain(
      "The user just moved from Beijing to Shanghai.",
    );
  });

  test("supersedes with an unknown ID is rejected", async () => {
    const mem = makeMem();
    await expect(
      mem.observe("The user moved from Beijing to Shanghai.", { supersedes: crypto.randomUUID() }),
    ).rejects.toThrow(/supersedes target .* was not found/);
  });

  test("supersedes of an already superseded record is rejected", async () => {
    const mem = makeMem();
    const first = await mem.observe("The user just moved from Beijing to Shanghai.");
    await mem.observe("The user moved from Beijing back to Shanghai again.", {
      supersedes: first.record?.id,
    });
    await expect(
      mem.observe("The user moved from Beijing back to Shanghai once more.", {
        supersedes: first.record?.id,
      }),
    ).rejects.toThrow(/already superseded/);
  });

  test("supersedes cannot cross scopes", async () => {
    const mem = makeMem();
    const globalFact = await mem.observe("The user prefers bun over npm everywhere.", {
      scope: "global",
    });
    await expect(
      mem.observe("The user prefers bun over npm in this repository.", {
        scope: "project",
        supersedes: globalFact.record?.id,
      }),
    ).rejects.toThrow(/belongs to scope/);
  });

  test("NOOP results expose the nearest blocking memory", async () => {
    const mem = makeMem();
    const first = await mem.observe("The user just moved from Beijing to Shanghai.");
    const r = await mem.observe("The user moved from Beijing back to Shanghai again.");
    expect(r.verdict).toBe(WriteVerdict.NOOP);
    expect(r.record).toBeNull();
    expect(r.nearest?.id).toBe(first.record?.id);
  });
});

describe("retrieval", () => {
  test("recall ranks the most relevant memory first", async () => {
    const mem = makeMem();
    await mem.observe("The user loves spicy Sichuan food.");
    await mem.observe("The deploy script lives in scripts/deploy.sh.");
    const hits = await mem.recall("What food does the user like?");
    expect(hits[0].record.text).toContain("Sichuan");
  });

  test("superseded memories are never recalled", async () => {
    const mem = makeMem({
      gate: {
        tauAdd: 0.45,
        dupSim: 0.85,
        conflictSim: 0.55,
        judge: { arbitrate: async () => ({ verdict: "UPDATE", confidence: 1 }) },
      },
    });
    await mem.observe("The user just moved from Beijing to Shanghai.");
    await mem.observe("The user moved from Beijing back to Shanghai again.");
    const hits = await mem.recall("where does the user live");
    for (const h of hits) expect(h.record.supersededBy).toBeNull();
  });
});

describe("decay and forgetting", () => {
  test("effective strength decays over time", async () => {
    const mem = makeMem();
    const r = await mem.observe("The user loves spicy Sichuan food.");
    const rec = r.record;
    expect(rec).not.toBeNull();
    if (!rec) throw new Error("expected a record");
    const s0 = mem.store.effectiveStrength(rec);
    rec.lastAccessed -= 5 * 24 * 3600; // age 5 days
    const s1 = mem.store.effectiveStrength(rec);
    expect(s1).toBeLessThan(s0);
  });

  test("forgetPass prunes weak memories but keeps reinforced and semantic ones", async () => {
    const mem = makeMem({
      store: { decayRatePerHour: 0.02, forgetThreshold: 0.15 },
    });
    // A reinforced memory.
    await mem.observe("The user loves spicy Sichuan food.");
    await mem.observe("The user loves spicy Sichuan food.");
    // A one-off memory.
    await mem.observe("The taxi driver wore a blue hat today.");
    // A consolidatable cluster -> semantic memory.
    await mem.observe("The SurMem framework uses surprise-gated writes.");
    await mem.observe("The SurMem framework has episodic and semantic layers.");
    await mem.reflect();

    // Age everything by 5 days, then re-reinforce the food fact.
    for (const rec of mem.store.all()) {
      rec.lastAccessed -= 5 * 24 * 3600;
      rec.createdAt -= 5 * 24 * 3600;
    }
    await mem.observe("The user loves spicy Sichuan food.");

    const forgotten = mem.forgetPass();
    const forgottenTexts = forgotten.map((m) => m.text);
    expect(forgottenTexts).toContain("The taxi driver wore a blue hat today.");

    const remaining = mem.store.active();
    expect(remaining.some((m) => m.text.includes("Sichuan"))).toBe(true);
    expect(remaining.some((m) => m.kind === Kind.SEMANTIC)).toBe(true);
  });

  test("semantic memories decay much slower than episodic ones", () => {
    const mem = makeMem();
    const fiveDaysAgo = Date.now() / 1000 - 5 * 24 * 3600;
    const episodic = {
      lastAccessed: fiveDaysAgo,
      baseStrength: 1,
      accessCount: 0,
      kind: Kind.EPISODIC,
    } as never;
    const semantic = {
      lastAccessed: fiveDaysAgo,
      baseStrength: 1,
      accessCount: 0,
      kind: Kind.SEMANTIC,
    } as never;
    const se = mem.store.effectiveStrength(episodic);
    const ss = mem.store.effectiveStrength(semantic);
    expect(ss).toBeGreaterThan(se);
  });
});

describe("consolidation", () => {
  test("clusters of related episodic memories become semantic memories", async () => {
    const mem = makeMem();
    await mem.observe("The SurMem framework uses surprise-gated writes.");
    await mem.observe("The SurMem framework has episodic and semantic layers.");
    const result = await mem.reflect();
    expect(result.created.length).toBeGreaterThan(0);
    expect(result.created[0].kind).toBe(Kind.SEMANTIC);
    expect(result.created[0].sourceIds.length).toBeGreaterThanOrEqual(2);
  });

  test("reflect() is idempotent: sources are consolidated only once", async () => {
    const mem = makeMem();
    await mem.observe("The SurMem framework uses surprise-gated writes.");
    await mem.observe("The SurMem framework has episodic and semantic layers.");
    const first = await mem.reflect();
    expect(first.created.length).toBeGreaterThan(0);
    const second = await mem.reflect();
    expect(second.created).toHaveLength(0); // nothing new to consolidate
    const third = await mem.reflect();
    expect(third.created).toHaveLength(0);
  });

  test("near-duplicate consolidations reinforce instead of duplicating", async () => {
    const mem = makeMem();
    await mem.observe("The SurMem framework uses surprise-gated writes.");
    await mem.observe("The SurMem framework has episodic and semantic layers.");
    await mem.reflect();
    const semanticCount = mem.store.byKind(Kind.SEMANTIC).length;
    // Simulate a second consolidation pass over the same material (e.g. after
    // new related episodes arrived) by un-marking the sources.
    for (const rec of mem.store.all()) delete rec.metadata.consolidatedInto;
    await mem.reflect();
    expect(mem.store.byKind(Kind.SEMANTIC)).toHaveLength(semanticCount);
  });
});

describe("persistence", () => {
  test("save/load round-trips all memories", async () => {
    const dir = await mkdtemp(join(tmpdir(), "surmem-"));
    const path = join(dir, "memory.json");
    try {
      const mem1 = new SurpriseMemory({ store: { persistPath: path } });
      await mem1.observe("The user loves spicy Sichuan food.");
      await mem1.observe("The deploy script lives in scripts/deploy.sh.");
      await mem1.save();

      const mem2 = new SurpriseMemory({ store: { persistPath: path } });
      await mem2.load();
      const texts = mem2.store.all().map((m) => m.text);
      expect(texts).toContain("The user loves spicy Sichuan food.");
      expect(texts).toContain("The deploy script lives in scripts/deploy.sh.");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("loading from a missing file starts empty", async () => {
    const mem = new SurpriseMemory({
      store: { persistPath: join(tmpdir(), "surmem-nonexistent-dir", "m.json") },
    });
    await mem.load();
    expect(mem.store.all()).toHaveLength(0);
  });

  test("sqlite persister round-trips all memories", async () => {
    const dir = await mkdtemp(join(tmpdir(), "surmem-sqlite-"));
    const path = join(dir, "memory.sqlite");
    try {
      const mem1 = new SurpriseMemory({
        store: { persister: new SqlitePersister(path) },
      });
      await mem1.observe("The user loves spicy Sichuan food.");
      await mem1.observe("The deploy script lives in scripts/deploy.sh.");
      await mem1.save();

      const mem2 = new SurpriseMemory({
        store: { persister: new SqlitePersister(path) },
      });
      await mem2.load();
      const texts = mem2.store.all().map((m) => m.text);
      expect(texts).toContain("The user loves spicy Sichuan food.");
      expect(texts).toContain("The deploy script lives in scripts/deploy.sh.");

      // Vectors must survive the BLOB round-trip bit-accurately enough to
      // preserve similarity rankings.
      const hits = await mem2.recall("What food does the user like?");
      expect(hits[0].record.text).toContain("Sichuan");
      await mem1.close();
      await mem2.close();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
