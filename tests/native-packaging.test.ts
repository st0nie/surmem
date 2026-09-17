import { expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { z } from "zod";

test("Codex discovers the shared plugin root through its native manifest", async () => {
  const path = resolve(".codex-plugin/plugin.json");
  expect(await Bun.file(path).exists()).toBe(true);
  const manifest = z
    .object({
      name: z.literal("surmem"),
      version: z.string(),
      skills: z.literal("./skills/"),
      mcpServers: z.object({
        surmem: z.object({
          command: z.literal("bun"),
          args: z.array(z.string()),
          cwd: z.literal("."),
          env_vars: z.array(z.string()),
        }),
      }),
    })
    .parse(JSON.parse(await readFile(path, "utf8")));
  expect(manifest.mcpServers.surmem.args).toEqual(["run", "--no-install", "src/codex/mcp.ts"]);
  expect(manifest.mcpServers.surmem.env_vars).toContain("SURMEM_PROJECT");
  expect(await Bun.file(resolve("src/codex/mcp.ts")).exists()).toBe(true);
  expect(await Bun.file(resolve(manifest.skills, "memory/SKILL.md")).exists()).toBe(true);
});

test("native marketplace installs the repository root rather than an incomplete subdirectory", async () => {
  const path = resolve(".claude-plugin/marketplace.json");
  expect(await Bun.file(path).exists()).toBe(true);
  const marketplace = z
    .object({
      name: z.literal("surmem"),
      plugins: z.array(z.object({ name: z.string(), source: z.string() })),
    })
    .parse(JSON.parse(await readFile(path, "utf8")));
  expect(marketplace.plugins).toEqual([{ name: "surmem", source: "./" }]);
});
