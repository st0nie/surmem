/** Codex runs this launcher from the installed plugin root, never the host project. */
import { parseArgs } from "../mcp/config";

async function main(): Promise<void> {
  if (process.argv.length > 2) {
    throw new Error("The Codex launcher accepts no arguments; set SURMEM_PROJECT and SURMEM_DIR instead.");
  }
  const project = process.env.SURMEM_PROJECT;
  if (!project) {
    throw new Error(
      'SURMEM_PROJECT is required. Start Codex from your project with SURMEM_PROJECT="$PWD" codex.',
    );
  }
  try {
    // Validate before dependency loading or storage access; never infer the project from cwd.
    process.env.SURMEM_PROJECT_DIR = parseArgs(["--project", project]).projectPath;
  } catch (error) {
    throw new Error(
      `SURMEM_PROJECT must name an existing absolute project directory: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error },
    );
  }
  // Reuse the bundled dependency check and existing MCP CLI transport/tools/shutdown.
  await import("../claude/mcp");
}

main().catch((error: unknown) => {
  process.stderr.write(`[surmem-codex] ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
