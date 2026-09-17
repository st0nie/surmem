#!/usr/bin/env bun
/** Local stdio entry point. Stdout belongs exclusively to the MCP transport. */
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { parseArgs } from "./config";
import { createMcpServer } from "./server";

async function main(): Promise<void> {
  if (process.argv.slice(2).length === 1 && process.argv[2] === "--help") {
    process.stderr.write(
      "Usage: surmem-mcp --project /absolute/project [--storage-dir /path]\nRequires Bun. Storage defaults to SURMEM_DIR or $PI_CODING_AGENT_DIR/surmem (~/.pi/agent/surmem).\n",
    );
    return;
  }
  const { server, backend } = createMcpServer(parseArgs(process.argv.slice(2)));
  const transport = new StdioServerTransport(process.stdin, process.stdout, { maxBufferSize: 256 * 1024 });
  let stopping = false;
  let finish!: () => void;
  const done = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const stop = () => {
    if (stopping) return;
    stopping = true;
    const deadline = setTimeout(() => {
      process.stderr.write("[surmem-mcp] Shutdown exceeded 10 seconds; pending operation was interrupted.\n");
      process.exit(1);
    }, 10_000);
    void (async () => {
      const results = await Promise.allSettled([backend.close(), server.close()]);
      for (const result of results) {
        if (result.status === "rejected") {
          process.stderr.write(`[surmem-mcp] Shutdown failed: ${String(result.reason)}\n`);
          process.exitCode = 1;
        }
      }
      clearTimeout(deadline);
      process.off("SIGINT", stop);
      process.off("SIGTERM", stop);
      process.stdin.off("end", stop);
      process.stdin.pause();
      finish();
    })();
  };
  server.server.onclose = stop;
  server.server.onerror = (error) => {
    process.stderr.write(`[surmem-mcp] Protocol error: ${error.message.slice(0, 1000)}\n`);
    process.exitCode = 1;
    stop();
  };
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  process.stdin.once("end", stop);
  await server.connect(transport);
  await done;
}

main().catch((error: unknown) => {
  process.stderr.write(`[surmem-mcp] ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
