/**
 * Pluggable persistence backends for the memory store.
 *
 * - JsonPersister: single JSON file, zero dependencies, human-readable.
 * - SqlitePersister: single SQLite file via the runtime's built-in SQLite
 *   (bun:sqlite under Bun, node:sqlite under Node >= 22.5). Transactional
 *   writes; vectors stored as Float64 blobs.
 *
 * Note on scale: retrieval currently does a brute-force cosine scan over
 * in-memory vectors, which is fine up to tens of thousands of memories
 * (~ms). For larger corpora, add sqlite-vec or an external vector DB behind
 * the same Persister + retrieval interfaces.
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import type { MemoryRecord } from "./types";

export interface Persister {
  load(): Promise<MemoryRecord[]>;
  save(records: MemoryRecord[]): Promise<void>;
}

/** Single JSON file persistence. */
export class JsonPersister implements Persister {
  constructor(private readonly path: string) {}

  async load(): Promise<MemoryRecord[]> {
    try {
      return JSON.parse(await readFile(this.path, "utf8")) as MemoryRecord[];
    } catch {
      return []; // missing or unreadable file: start empty
    }
  }

  async save(records: MemoryRecord[]): Promise<void> {
    await mkdir(dirname(this.path), { recursive: true });
    await writeFile(this.path, JSON.stringify(records, null, 2), "utf8");
  }
}

type SqliteStatement = {
  run(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
};

type SqliteDatabase = {
  exec(sql: string): unknown;
  prepare(sql: string): SqliteStatement;
  close(): void;
};

/** Resolve the runtime's built-in SQLite driver (Bun or Node). */
async function openSqlite(path: string): Promise<SqliteDatabase> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (typeof (globalThis as any).Bun !== "undefined") {
    const { Database } = await import("bun:sqlite");
    return new Database(path) as unknown as SqliteDatabase;
  }
  try {
    // Node >= 22.5 (experimental) / >= 23.4 (unflagged)
    const { DatabaseSync } = await import("node:sqlite");
    return new DatabaseSync(path) as unknown as SqliteDatabase;
  } catch {
    throw new Error(
      "SqlitePersister requires Bun or Node >= 22.5 with built-in SQLite support.",
    );
  }
}

/** SQLite file persistence with transactional writes. */
export class SqlitePersister implements Persister {
  private db: SqliteDatabase | null = null;

  constructor(private readonly path: string) {}

  private async open(): Promise<SqliteDatabase> {
    if (this.db) return this.db;
    await mkdir(dirname(this.path), { recursive: true });
    this.db = await openSqlite(this.path);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        text TEXT NOT NULL,
        vector BLOB NOT NULL,
        created_at REAL NOT NULL,
        last_accessed REAL NOT NULL,
        base_strength REAL NOT NULL,
        access_count INTEGER NOT NULL,
        surprise_at_write REAL NOT NULL,
        superseded_by TEXT,
        source_ids TEXT NOT NULL,
        metadata TEXT NOT NULL
      )
    `);
    return this.db;
  }

  async load(): Promise<MemoryRecord[]> {
    try {
      const db = await this.open();
      const rows = db.prepare("SELECT * FROM memories").all() as Array<Record<string, unknown>>;
      return rows.map((row) => this.rowToRecord(row));
    } catch {
      return []; // missing/corrupt DB: start empty
    }
  }

  async save(records: MemoryRecord[]): Promise<void> {
    const db = await this.open();
    db.exec("BEGIN");
    try {
      db.exec("DELETE FROM memories");
      const insert = db.prepare(`
        INSERT INTO memories (
          id, kind, text, vector, created_at, last_accessed, base_strength,
          access_count, surprise_at_write, superseded_by, source_ids, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);
      for (const rec of records) {
        insert.run(
          rec.id,
          rec.kind,
          rec.text,
          Buffer.from(new Float64Array(rec.vector).buffer),
          rec.createdAt,
          rec.lastAccessed,
          rec.baseStrength,
          rec.accessCount,
          rec.surpriseAtWrite,
          rec.supersededBy,
          JSON.stringify(rec.sourceIds),
          JSON.stringify(rec.metadata),
        );
      }
      db.exec("COMMIT");
    } catch (err) {
      db.exec("ROLLBACK");
      throw err;
    }
  }

  private rowToRecord(row: Record<string, unknown>): MemoryRecord {
    const blob = row.vector as Uint8Array;
    const floats = new Float64Array(blob.buffer, blob.byteOffset, blob.byteLength / 8);
    return {
      id: row.id as string,
      kind: row.kind as MemoryRecord["kind"],
      text: row.text as string,
      vector: Array.from(floats),
      createdAt: row.created_at as number,
      lastAccessed: row.last_accessed as number,
      baseStrength: row.base_strength as number,
      accessCount: row.access_count as number,
      surpriseAtWrite: row.surprise_at_write as number,
      supersededBy: (row.superseded_by as string | null) ?? null,
      sourceIds: JSON.parse(row.source_ids as string) as string[],
      metadata: JSON.parse(row.metadata as string) as Record<string, unknown>,
    };
  }
}
