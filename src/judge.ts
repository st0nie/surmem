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

/**
 * Memorability judge: decides whether a raw conversation message contains a
 * fact worth long-term memory, and distills it into one self-contained
 * sentence. Returns null when the message is a question, command, or chatter.
 */
export interface MemorabilityJudge {
  assess(text: string): Promise<string | null>;
}

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

/** Chat-completion-backed memorability judge. */
export class OpenAIMemorabilityJudge implements MemorabilityJudge {
  constructor(private opts: OpenAIChatOptions) {}

  async assess(text: string): Promise<string | null> {
    const answer = await chat(this.opts, memorabilityPrompt(text));
    if (!answer || answer.toUpperCase().startsWith("NONE")) return null;
    return answer;
  }
}

/** Shared prompt for memorability assessment. */
export function memorabilityPrompt(text: string): string {
  return `You are the memory gate of an AI agent. Decide whether the following message contains a durable fact, preference, or decision worth long-term memory.

Message: """${text}"""

Rules:
- Questions, commands, requests, greetings, and task instructions are NOT memories.
- Facts about the user, stable preferences, project decisions, and hard-won lessons ARE memories.
- If it is a memory, rewrite it as ONE self-contained sentence (third person, no pronouns without referents).
- If it is not a memory, reply with exactly: NONE

Reply with the single sentence or NONE. Nothing else.`;
}

export interface GgufJudgeOptions {
  /** Absolute path to a generative GGUF model (e.g. qmd's cached query-expansion model). */
  modelPath: string;
  /** GPU backend for llama.cpp. Defaults to SURMEM_GGUF_GPU env, else false (CPU). */
  gpu?: "auto" | "metal" | "cuda" | "vulkan" | false;
}

interface LlamaChatSessionLike {
  prompt(text: string): Promise<string>;
  setChatHistory?(history: unknown[]): void;
  dispose?(): Promise<void> | void;
}

interface LlamaContextLike {
  getSequence(): unknown;
  dispose?(): Promise<void> | void;
}

/**
 * Fully local memorability judge via node-llama-cpp and a small generative
 * GGUF model. Quality is lower than a frontier model, but it is offline and
 * free. node-llama-cpp is only imported when this class is instantiated.
 */
export class GgufMemorabilityJudge implements MemorabilityJudge {
  private sessionPromise: Promise<LlamaChatSessionLike> | null = null;
  private context: LlamaContextLike | null = null;

  constructor(private opts: GgufJudgeOptions) {}

  private async session(): Promise<LlamaChatSessionLike> {
    if (!this.sessionPromise) {
      this.sessionPromise = (async () => {
        const llamaCpp = await import("node-llama-cpp");
        const llama = await llamaCpp.getLlama({
          gpu:
            this.opts.gpu ??
            (process.env.SURMEM_GGUF_GPU as GgufJudgeOptions["gpu"]) ??
            false,
        });
        const model = await llama.loadModel({ modelPath: this.opts.modelPath });
        this.context =
          (await model.createContext()) as unknown as LlamaContextLike;
        return new llamaCpp.LlamaChatSession({
          contextSequence: this.context.getSequence() as never,
        }) as unknown as LlamaChatSessionLike;
      })();
    }
    return this.sessionPromise;
  }

  async assess(text: string): Promise<string | null> {
    try {
      const session = await this.session();
      // Reset chat history so assessments do not accumulate context.
      session.setChatHistory?.([]);
      const answer = (await session.prompt(memorabilityPrompt(text))).trim();
      if (!answer || answer.toUpperCase().startsWith("NONE")) return null;
      return answer;
    } catch {
      return null; // judge failure must never break the agent loop
    }
  }

  async dispose(): Promise<void> {
    if (this.sessionPromise) {
      const session = await this.sessionPromise;
      await session.dispose?.();
      await this.context?.dispose?.();
      this.sessionPromise = null;
      this.context = null;
    }
  }
}
