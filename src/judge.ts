/**
 * LLM-backed implementations of the gate's LLMJudge and the consolidator's
 * Summarizer, against any OpenAI-compatible chat completions endpoint.
 *
 * Both are optional: without them, the gate resolves the conflict zone by
 * recency (new supersedes old) and consolidation promotes the strongest
 * cluster member. With them, the framework distinguishes paraphrase from
 * contradiction and produces distilled semantic facts.
 */

import type { LLMJudge } from "./gate";
import type { Summarizer } from "./consolidation";

export interface OpenAIChatOptions {
  apiKey: string;
  model: string; // e.g. "gpt-4o-mini"
  baseUrl?: string; // default "https://api.openai.com/v1"
}

async function chat(opts: OpenAIChatOptions, prompt: string): Promise<string> {
  const baseUrl = (opts.baseUrl ?? "https://api.openai.com/v1").replace(/\/$/, "");
  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${opts.apiKey}`,
    },
    body: JSON.stringify({
      model: opts.model,
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
    }),
  });
  if (!res.ok) {
    throw new Error(`Chat request failed: ${res.status} ${await res.text()}`);
  }
  const payload = (await res.json()) as {
    choices: Array<{ message: { content: string } }>;
  };
  return payload.choices[0]?.message.content.trim() ?? "";
}

/** Arbitrates the conflict zone: paraphrase vs contradiction vs new fact. */
export class OpenAIJudge implements LLMJudge {
  constructor(private opts: OpenAIChatOptions) {}

  async arbitrate(newText: string, nearestText: string): Promise<string> {
    const answer = await chat(
      this.opts,
      `You are a memory gate for an AI agent. An old memory and a new observation are given.

Old memory: "${nearestText}"
New observation: "${newText}"

Decide what to do with the new observation. Reply with exactly one word:
- REINFORCE: it restates the old memory (same meaning, maybe reworded)
- UPDATE: it contradicts or replaces the old memory (the new one is fresher)
- ADD: it is related but contains genuinely new information worth a separate memory
- NOOP: it is trivial or not worth remembering

One word only:`,
    );
    const v = answer.toUpperCase();
    if (v.includes("REINFORCE")) return "REINFORCE";
    if (v.includes("UPDATE")) return "UPDATE";
    if (v.includes("ADD")) return "ADD";
    return "NOOP";
  }
}

/** Distills a cluster of episodic memories into one semantic fact. */
export class OpenAISummarizer implements Summarizer {
  constructor(private opts: OpenAIChatOptions) {}

  async summarize(texts: string[]): Promise<string> {
    const list = texts.map((t, i) => `${i + 1}. ${t}`).join("\n");
    return chat(
      this.opts,
      `Distill the following related observations into ONE concise, self-contained fact (one sentence). Preserve concrete details (names, places, decisions). Output only the fact.

${list}`,
    );
  }
}
