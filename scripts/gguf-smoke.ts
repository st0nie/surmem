/**
 * Smoke test for GgufEmbedder with the qmd-cached embeddinggemma-300M model.
 *
 * Verifies:
 *   1. The model loads and produces 768-dim vectors.
 *   2. Semantic similarity ordering is sane: a food query should be closest
 *      to the food memory, not to the deployment or relocation memories.
 *   3. Paraphrases score high (unlike the hash embedder).
 *
 * Run: bun run scripts/gguf-smoke.ts
 */

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { cosine, GgufEmbedder } from "../src/index";

const MODEL_PATH =
  process.env.SURMEM_GGUF_MODEL_PATH ??
  join(homedir(), ".cache/qmd/models/hf_ggml-org_embeddinggemma-300M-Q8_0.gguf");

if (!existsSync(MODEL_PATH)) {
  console.error(`Model not found at ${MODEL_PATH}`);
  process.exit(1);
}

const pair = GgufEmbedder.createPair({ modelPath: MODEL_PATH, dim: 768 });

const docs = [
  "The user loves spicy Sichuan food.",
  "The deploy script lives in scripts/deploy.sh.",
  "The user moved from Beijing to Shanghai.",
];

console.log("Embedding documents...");
const docVecs = await pair.document.embed(docs);
console.log(`  got ${docVecs.length} vectors of dim ${docVecs[0].length}`);

console.log("Embedding query...");
const [qFood] = await pair.query.embed(["What kind of cuisine does the user enjoy?"]);
const [qDeploy] = await pair.query.embed(["How do I deploy the app?"]);

console.log("\nSimilarity of 'What kind of cuisine does the user enjoy?' to docs:");
docs.forEach((d, i) => {
  console.log(`  ${cosine(qFood, docVecs[i]).toFixed(3)}  ${d}`);
});

console.log("\nSimilarity of 'How do I deploy the app?' to docs:");
docs.forEach((d, i) => {
  console.log(`  ${cosine(qDeploy, docVecs[i]).toFixed(3)}  ${d}`);
});

// Paraphrase check: the hash embedder fails this; a real model should not.
const [paraA] = await pair.document.embed(["The user relocated from Beijing to Shanghai."]);
const simPara = cosine(paraA, docVecs[2]);
console.log(`\nParaphrase similarity (relocated vs moved): ${simPara.toFixed(3)}`);

const foodBest =
  cosine(qFood, docVecs[0]) > cosine(qFood, docVecs[1]) &&
  cosine(qFood, docVecs[0]) > cosine(qFood, docVecs[2]);
const deployBest =
  cosine(qDeploy, docVecs[1]) > cosine(qDeploy, docVecs[0]) &&
  cosine(qDeploy, docVecs[1]) > cosine(qDeploy, docVecs[2]);

console.log(`\n${foodBest && deployBest && simPara > 0.8 ? "SMOKE TEST PASSED" : "SMOKE TEST FAILED"}`);

await (pair.document as GgufEmbedder).dispose();
