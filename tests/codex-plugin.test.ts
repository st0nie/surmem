import { afterEach, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { cp, mkdir, mkdtemp, readFile, rm, stat, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const pluginRoot = resolve(".");
const entry = "src/codex/mcp.ts";
const roots: string[] = [];
const clients: Client[] = [];

afterEach(async () => {
  await Promise.all(clients.splice(0).map((client) => client.close()));
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "surmem-codex-"));
  roots.push(root);
  const project = join(root, "host project ' $literal");
  await mkdir(project);
  return { root, project, storage: join(root, "storage") };
}

function environment(root: string, storage: string): Record<string, string> {
  return {
    PATH: process.env.PATH ?? "",
    HOME: root,
    PI_CODING_AGENT_DIR: join(root, "agent"),
    SURMEM_DIR: storage,
    SURMEM_CONFIG_PATH: join(storage, "config.json"),
    SURMEM_EMBEDDER: "hash",
    SURMEM_JUDGE_MODE: "heuristic",
  };
}

async function launch(cwd: string, env: Record<string, string>, args: string[] = []) {
  const child = Bun.spawn([process.execPath, "run", "--no-install", entry, ...args], {
    cwd,
    env,
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
    timeout: 5000,
  });
  const [code, stdout, stderr] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ]);
  return { code, stdout, stderr };
}

async function connect(cwd: string, env: Record<string, string>) {
  const client = new Client({ name: "codex-plugin-test", version: "1" });
  clients.push(client);
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: ["run", "--no-install", entry],
    cwd,
    env,
    stderr: "pipe",
  });
  let stderr = "";
  transport.stderr?.on("data", (chunk) => {
    stderr += chunk.toString();
  });
  try {
    await client.connect(transport, { timeout: 5000 });
  } catch (error) {
    throw new Error(`Codex MCP connection failed: ${stderr}`, { cause: error });
  }
  return client;
}

async function call(client: Client, name: string, args: Record<string, unknown> = {}) {
  const result = await client.callTool({ name: `surmem_${name}`, arguments: args }, undefined, {
    timeout: 5000,
  });
  expect(result.isError).not.toBe(true);
  return result.structuredContent as Record<string, any>;
}

async function cachedPlugin(root: string) {
  const cached = join(root, "cached plugin ' $literal");
  await mkdir(cached);
  for (const path of ["src", "hooks", "package.json", "bun.lock"]) {
    await cp(join(pluginRoot, path), join(cached, path), { recursive: true });
  }
  return cached;
}

test("Codex missing project guard fails before any storage writes and never falls back to cwd", async () => {
  const { root, project, storage } = await fixture();
  const result = await launch(pluginRoot, {
    ...environment(root, storage),
    CLAUDE_PROJECT_DIR: project,
    SURMEM_PROJECT_DIR: project,
  });
  expect(result.code).toBe(1);
  expect(result.stdout).toBe("");
  expect(result.stderr).toContain("SURMEM_PROJECT");
  expect(await stat(storage).catch((error: NodeJS.ErrnoException) => error.code)).toBe("ENOENT");
  expect(await stat(join(root, "agent")).catch((error: NodeJS.ErrnoException) => error.code)).toBe("ENOENT");
});

test("Codex rejects empty, relative, missing and non-directory projects before creating storage", async () => {
  const { root, project, storage } = await fixture();
  const file = join(root, "not-a-directory");
  await writeFile(file, "project-boundary-sentinel");
  for (const value of ["", ".", "relative/project", join(root, "missing"), file]) {
    const result = await launch(pluginRoot, { ...environment(root, storage), SURMEM_PROJECT: value });
    expect(result.code).toBe(1);
    expect(result.stdout).toBe("");
    expect(result.stderr).toContain("SURMEM_PROJECT");
  }
  const args = await launch(pluginRoot, { ...environment(root, storage), SURMEM_PROJECT: project }, [
    "--project",
    pluginRoot,
  ]);
  expect(args.code).toBe(1);
  expect(args.stdout).toBe("");
  expect(await readFile(file, "utf8")).toBe("project-boundary-sentinel");
  expect(await stat(storage).catch((error: NodeJS.ErrnoException) => error.code)).toBe("ENOENT");
  expect(await stat(join(root, "agent")).catch((error: NodeJS.ErrnoException) => error.code)).toBe("ENOENT");
});

test("Codex stdio tools bind the explicit canonical project rather than the plugin cwd", async () => {
  const { root, project, storage } = await fixture();
  const alias = join(root, "project-alias");
  await symlink(project, alias, "dir");
  const env = { ...environment(root, storage), SURMEM_PROJECT: alias };
  const client = await connect(pluginRoot, env);
  expect((await client.listTools()).tools.map((tool) => tool.name)).toContain("surmem_remember");
  expect(await Bun.file(join(storage, "config.json")).exists()).toBe(false);
  const remembered = await call(client, "remember", {
    text: "Codex project releases use cobalt verification channels",
    scope: "project",
  });
  expect(remembered.verdict).toBe("ADD");
  const status = await call(client, "status");
  const key = createHash("sha256").update(project).digest("hex").slice(0, 20);
  expect(status.projectPath).toBe(project);
  expect(status.projectKey).toBe(key);
  expect((await stat(join(storage, "projects", `${key}.sqlite`))).isFile()).toBe(true);
  await client.close();
  const reopened = await connect(pluginRoot, env);
  const recalled = await call(reopened, "recall", {
    query: "cobalt verification channels",
    scope: "project",
  });
  expect(recalled.memories.map((memory: any) => memory.id)).toContain(remembered.record.id);
  const other = join(root, "other-project");
  await mkdir(other);
  const isolated = await connect(pluginRoot, { ...environment(root, storage), SURMEM_PROJECT: other });
  expect((await call(isolated, "list", { scope: "project" })).memories).toEqual([]);
});

test("Codex cached launch fails clearly without dependencies and serves tools after explicit setup", async () => {
  const { root, project, storage } = await fixture();
  const cached = await cachedPlugin(root);
  const env = { ...environment(root, storage), SURMEM_PROJECT: project };
  const missing = await launch(cached, env);
  expect(missing.code).toBe(1);
  expect(missing.stdout).toBe("");
  expect(missing.stderr).toContain(cached);
  expect(await stat(storage).catch((error: NodeJS.ErrnoException) => error.code)).toBe("ENOENT");
  expect(await stat(join(cached, "node_modules")).catch((error: NodeJS.ErrnoException) => error.code)).toBe(
    "ENOENT",
  );
  const noProject = await launch(cached, environment(root, storage));
  expect(noProject.code).toBe(1);
  expect(noProject.stdout).toBe("");
  expect(noProject.stderr).toContain("SURMEM_PROJECT");
  // Supply the real installed dependency tree deterministically, without an installer or network.
  await symlink(join(pluginRoot, "node_modules"), join(cached, "node_modules"), "dir");
  const client = await connect(cached, env);
  const remembered = await call(client, "remember", {
    text: "Cached Codex plugins preserve host project memory scope",
    scope: "project",
  });
  expect(remembered.verdict).toBe("ADD");
  expect((await call(client, "status")).projectPath).toBe(project);
  expect(
    (await call(client, "recall", { query: "host project memory scope", scope: "project" })).memories.map(
      (memory: any) => memory.id,
    ),
  ).toContain(remembered.record.id);
});

test("shared SessionStart hook executes with Codex shell-command semantics from a cached plugin", async () => {
  const { root, project, storage } = await fixture();
  const cached = await cachedPlugin(root);
  const hooks = JSON.parse(await readFile(join(cached, "hooks/hooks.json"), "utf8"));
  const hook = hooks.hooks.SessionStart[0].hooks[0];
  // Codex executes command via a shell; an exec-form args field is not consumed.
  const child = Bun.spawn(["/bin/sh", "-c", hook.command], {
    cwd: project,
    env: {
      ...environment(root, storage),
      CLAUDE_PLUGIN_ROOT: cached,
      SURMEM_EMBEDDER: "gguf",
      SURMEM_GGUF_MODEL_PATH: join(root, "no-model.gguf"),
      SURMEM_JUDGE_GGUF: join(root, "no-judge.gguf"),
    },
    stdin: new Blob([JSON.stringify({ hook_event_name: "SessionStart", cwd: project })]),
    stdout: "pipe",
    stderr: "pipe",
    timeout: 5000,
  });
  const [code, stdout, stderr] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ]);
  expect(code).toBe(0);
  expect(stderr).toBe("");
  const result = JSON.parse(stdout);
  expect(result.hookSpecificOutput.hookEventName).toBe("SessionStart");
  expect(typeof result.hookSpecificOutput.additionalContext).toBe("string");
  expect(Buffer.byteLength(result.hookSpecificOutput.additionalContext)).toBeLessThanOrEqual(12 * 1024);
  expect(await stat(storage).catch((error: NodeJS.ErrnoException) => error.code)).toBe("ENOENT");
  expect(await stat(join(cached, "node_modules")).catch((error: NodeJS.ErrnoException) => error.code)).toBe(
    "ENOENT",
  );
});
