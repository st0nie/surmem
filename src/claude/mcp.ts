/** Native plugin launcher. Keep the host project separate from the installed plugin directory. */
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseArgs } from "../mcp/config";

async function main(): Promise<void> {
  if (process.argv.length > 2) {
    throw new Error(
      "The native launcher accepts no arguments; set SURMEM_PROJECT_DIR and SURMEM_DIR instead.",
    );
  }
  const project = process.env.SURMEM_PROJECT_DIR ?? process.env.CLAUDE_PROJECT_DIR ?? process.cwd();
  const config = parseArgs(["--project", project]);
  const root = fileURLToPath(new URL("../../", import.meta.url));
  for (const dependency of ["@modelcontextprotocol/sdk", "zod"]) {
    if (!existsSync(new URL(`../../node_modules/${dependency}/package.json`, import.meta.url))) {
      const quoted = `'${root.replaceAll("'", "'\\''")}'`;
      throw new Error(
        `Missing installed runtime dependencies in ${root}. Native plugin caches do not include node_modules.\nRun: bun install --cwd ${quoted} --production --frozen-lockfile\nThen restart the plugin MCP server. Repeat after plugin updates.`,
      );
    }
  }
  // Delegate transport, shutdown, argument validation and all tools to the existing MCP entry point.
  process.argv = [
    process.execPath,
    fileURLToPath(new URL("../mcp/cli.ts", import.meta.url)),
    "--project",
    config.projectPath,
    "--storage-dir",
    config.storageDir,
  ];
  await import("../mcp/cli");
}

main().catch((error: unknown) => {
  process.stderr.write(`[surmem-plugin] ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
