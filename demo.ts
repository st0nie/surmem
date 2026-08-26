/**
 * Demo: exercises the full surprise-gated memory lifecycle.
 *
 *   1. ADD       - a novel fact is stored
 *   2. UPDATE    - a contradicting fact supersedes the old one
 *   3. REINFORCE - a near-duplicate strengthens the existing trace
 *   4. NOOP      - trivial filler is discarded
 *   5. recall    - hybrid retrieval for context injection
 *   6. reflect   - consolidation: episodic -> semantic
 *   7. forget    - decay-based pruning (simulated time travel)
 *
 * Run: bun run demo.ts
 */

import { SurpriseMemory } from "./src/index";

function show(title: string): void {
  console.log("\n" + "=".repeat(60));
  console.log(title);
  console.log("=".repeat(60));
}

const mem = new SurpriseMemory({
  gate: { tauAdd: 0.45, dupSim: 0.85, conflictSim: 0.55 },
  consolidation: { clusterSim: 0.3 }, // tuned for the built-in hash embedder
  store: { decayRatePerHour: 0.02, forgetThreshold: 0.15 },
});

async function observe(text: string): Promise<void> {
  const r = await mem.observe(text);
  console.log(
    `observe("${text}")\n  -> ${r.verdict} (surprise=${r.surprise.toFixed(2)})` +
      (r.superseded ? `  [supersedes: "${r.superseded.text}"]` : ""),
  );
}

// --- 1-4: the four write verdicts -------------------------------------------
show("1. Write path: surprise gate in action");

await observe("The user just moved from Beijing to Shanghai."); // ADD (novel)
await observe("The user just moved from Beijing to Shanghai."); // REINFORCE (exact dup)
await observe("The user loves spicy Sichuan food."); // ADD (new topic)
await observe("The user moved from Beijing back to Shanghai again."); // UPDATE (conflict)
await observe("ok"); // NOOP (trivial)

// --- 5: retrieval ------------------------------------------------------------
show("2. Read path: recall for context injection");

console.log(await mem.recallAsContext("Where does the user live now?"));
console.log();
console.log(await mem.recallAsContext("What food does the user like?"));

// --- 6: consolidation --------------------------------------------------------
show("3. reflect(): episodic -> semantic consolidation");

await observe("The user is building a memory framework called SurMem.");
await observe("The SurMem memory framework uses surprise-gated writes.");
await observe("The SurMem memory framework has episodic and semantic layers.");

const result = await mem.reflect();
console.log(`Consolidated ${result.clusters} cluster(s) ->`);
for (const rec of result.created) {
  console.log(`  [semantic] ${rec.text}  (sources: ${rec.sourceIds.length})`);
}

// --- 7: forgetting -----------------------------------------------------------
show("4. forgetPass(): Ebbinghaus decay (simulated +5 days)");

// Simulate time travel: age every record by 5 days.
for (const rec of mem.store.all()) {
  rec.lastAccessed -= 5 * 24 * 3600;
  rec.createdAt -= 5 * 24 * 3600;
}

// A recent interaction re-touches the food fact: reinforcement rescues it
// from decay (spaced repetition in action).
await observe("The user loves spicy Sichuan food.");

const forgotten = mem.forgetPass();
console.log(`Forgot ${forgotten.length} weak memories:`);
for (const rec of forgotten) console.log(`  [forgotten] ${rec.text}`);

console.log("\nRemaining memories after decay:");
for (const rec of mem.store.active()) {
  console.log(
    `  [${rec.kind} | strength=${mem.store.effectiveStrength(rec).toFixed(2)}] ${rec.text}`,
  );
}
