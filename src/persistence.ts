/** Versioned, validated, concurrency-safe persistence backends. */

import { randomUUID } from "node:crypto";
import { constants as fsConstants } from "node:fs";
import { chmod, mkdir, open, readFile, rename, stat, unlink, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import { PersistenceConflictError, PersistenceError, ValidationError } from "./errors";
import { Kind, type MemoryRecord, type MemoryTombstone } from "./types";

export const STORE_SCHEMA_VERSION = 2;

export interface PersistenceSnapshot {
  schemaVersion: number;
  revision: number;
  embeddingFingerprint: string;
  updatedAt: number;
  records: MemoryRecord[];
  tombstones: MemoryTombstone[];
}

export interface PersistenceHealth {
  backend: "memory" | "json" | "sqlite" | "custom";
  path?: string;
  revision?: number;
  schemaVersion?: number;
}

export interface Persister {
  load(): Promise<PersistenceSnapshot | null>;
  save(snapshot: PersistenceSnapshot, expectedRevision: number): Promise<number>;
  close?(): Promise<void> | void;
  health?(): Promise<PersistenceHealth> | PersistenceHealth;
}

function finiteNumber(value: unknown, name: string, min = -Infinity): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min) {
    throw new ValidationError(`Invalid ${name} in persisted memory.`);
  }
  return value;
}

function isKind(value: unknown): value is Kind {
  return value === Kind.EPISODIC || value === Kind.SEMANTIC || value === Kind.PROCEDURAL;
}

export function normalizeRecord(value: unknown): MemoryRecord {
  if (!value || typeof value !== "object") throw new ValidationError("Invalid memory record.");
  const raw = value as Partial<MemoryRecord>;
  if (typeof raw.id !== "string" || raw.id.length < 1 || raw.id.length > 128)
    throw new ValidationError("Invalid memory id.");
  if (typeof raw.text !== "string" || raw.text.length < 1 || raw.text.length > 100_000)
    throw new ValidationError(`Invalid text for memory ${raw.id}.`);
  if (
    !Array.isArray(raw.vector) ||
    raw.vector.length < 1 ||
    raw.vector.length > 65_536 ||
    !raw.vector.every(Number.isFinite)
  ) {
    throw new ValidationError(`Invalid vector for memory ${raw.id}.`);
  }
  if (!isKind(raw.kind)) throw new ValidationError(`Invalid kind for memory ${raw.id}.`);
  const createdAt = finiteNumber(raw.createdAt, "createdAt", 0);
  const updatedAt = raw.updatedAt === undefined ? createdAt : finiteNumber(raw.updatedAt, "updatedAt", 0);
  const metadata =
    raw.metadata && typeof raw.metadata === "object" && !Array.isArray(raw.metadata)
      ? { ...raw.metadata }
      : {};
  return {
    id: raw.id,
    text: raw.text,
    vector: [...raw.vector],
    kind: raw.kind,
    createdAt,
    updatedAt,
    lastAccessed: finiteNumber(raw.lastAccessed, "lastAccessed", 0),
    baseStrength: finiteNumber(raw.baseStrength, "baseStrength", 0),
    accessCount: Math.floor(finiteNumber(raw.accessCount, "accessCount", 0)),
    surpriseAtWrite: finiteNumber(raw.surpriseAtWrite, "surpriseAtWrite", 0),
    supersededBy: raw.supersededBy == null ? null : String(raw.supersededBy),
    sourceIds: Array.isArray(raw.sourceIds)
      ? raw.sourceIds.filter((id): id is string => typeof id === "string")
      : [],
    metadata,
  };
}

export function normalizeSnapshot(value: unknown): PersistenceSnapshot {
  if (Array.isArray(value)) {
    const records = value.map(normalizeRecord);
    return {
      schemaVersion: STORE_SCHEMA_VERSION,
      revision: 0,
      embeddingFingerprint: "legacy:unknown",
      updatedAt: 0,
      records,
      tombstones: [],
    };
  }
  if (!value || typeof value !== "object") throw new ValidationError("Memory store root must be an object.");
  const raw = value as Partial<PersistenceSnapshot>;
  const schemaVersion = finiteNumber(raw.schemaVersion, "schemaVersion", 1);
  if (schemaVersion > STORE_SCHEMA_VERSION) {
    throw new ValidationError(
      `Memory schema v${schemaVersion} is newer than supported v${STORE_SCHEMA_VERSION}.`,
    );
  }
  if (!Array.isArray(raw.records)) throw new ValidationError("Memory store records must be an array.");
  const tombstones = Array.isArray(raw.tombstones)
    ? raw.tombstones.map((entry) => {
        if (!entry || typeof entry !== "object") throw new ValidationError("Invalid tombstone.");
        const row = entry as Partial<MemoryTombstone>;
        if (typeof row.id !== "string") throw new ValidationError("Invalid tombstone id.");
        return { id: row.id, deletedAt: finiteNumber(row.deletedAt, "deletedAt", 0) };
      })
    : [];
  return {
    schemaVersion: STORE_SCHEMA_VERSION,
    revision: Math.floor(finiteNumber(raw.revision ?? 0, "revision", 0)),
    embeddingFingerprint:
      typeof raw.embeddingFingerprint === "string" ? raw.embeddingFingerprint : "legacy:unknown",
    updatedAt: finiteNumber(raw.updatedAt ?? 0, "updatedAt", 0),
    records: raw.records.map(normalizeRecord),
    tombstones,
  };
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function withFileLock<T>(path: string, fn: () => Promise<T>): Promise<T> {
  const lockPath = `${path}.lock`;
  const token = `${process.pid}:${randomUUID()}`;
  const deadline = Date.now() + 5000;
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  while (true) {
    try {
      const handle = await open(
        lockPath,
        fsConstants.O_CREAT | fsConstants.O_EXCL | fsConstants.O_WRONLY,
        0o600,
      );
      await handle.writeFile(token, "utf8");
      await handle.close();
      break;
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code !== "EEXIST")
        throw new PersistenceError(`Cannot acquire memory lock ${lockPath}.`, { cause: error });
      try {
        const info = await stat(lockPath);
        if (Date.now() - info.mtimeMs > 30_000) await unlink(lockPath);
      } catch {}
      if (Date.now() >= deadline)
        throw new PersistenceError(`Timed out waiting for memory lock ${lockPath}.`);
      await new Promise((resolve) => setTimeout(resolve, 25 + Math.floor(Math.random() * 50)));
    }
  }
  try {
    return await fn();
  } finally {
    try {
      if ((await readFile(lockPath, "utf8")) === token) await unlink(lockPath);
    } catch {}
  }
}

/** Atomic JSON snapshots with optimistic revision checks and 0600 permissions. */
export class JsonPersister implements Persister {
  constructor(readonly path: string) {}

  async load(): Promise<PersistenceSnapshot | null> {
    if (!(await pathExists(this.path))) return null;
    try {
      const info = await stat(this.path);
      if (!info.isFile()) throw new PersistenceError(`Memory path is not a regular file: ${this.path}`);
      if (info.size > 256 * 1024 * 1024)
        throw new PersistenceError(`Memory file exceeds the 256 MiB safety limit: ${this.path}`);
      return normalizeSnapshot(JSON.parse(await readFile(this.path, "utf8")));
    } catch (error) {
      if (error instanceof PersistenceError || error instanceof ValidationError) throw error;
      throw new PersistenceError(`Failed to read memory store ${this.path}; the file was preserved.`, {
        cause: error,
      });
    }
  }

  async save(snapshot: PersistenceSnapshot, expectedRevision: number): Promise<number> {
    return withFileLock(this.path, async () => {
      const current = await this.load();
      const actual = current?.revision ?? 0;
      if (actual !== expectedRevision) {
        throw new PersistenceConflictError(
          `Memory store changed concurrently at ${this.path}.`,
          expectedRevision,
          actual,
        );
      }
      const revision = actual + 1;
      const next = normalizeSnapshot({
        ...snapshot,
        schemaVersion: STORE_SCHEMA_VERSION,
        revision,
        updatedAt: Date.now() / 1000,
      });
      const temp = `${this.path}.tmp-${process.pid}-${randomUUID()}`;
      await mkdir(dirname(this.path), { recursive: true, mode: 0o700 });
      try {
        await writeFile(temp, `${JSON.stringify(next, null, 2)}\n`, {
          encoding: "utf8",
          mode: 0o600,
          flag: "wx",
        });
        await rename(temp, this.path);
        await chmod(this.path, 0o600);
      } finally {
        try {
          await unlink(temp);
        } catch {}
      }
      return revision;
    });
  }

  health(): PersistenceHealth {
    return { backend: "json", path: this.path, schemaVersion: STORE_SCHEMA_VERSION };
  }
}

type SqliteStatement = {
  run(...params: unknown[]): { changes?: number } | unknown;
  all(...params: unknown[]): unknown[];
  get(...params: unknown[]): unknown;
};
type SqliteDatabase = { exec(sql: string): unknown; prepare(sql: string): SqliteStatement; close(): void };

async function openSqlite(path: string): Promise<SqliteDatabase> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (typeof (globalThis as any).Bun !== "undefined") {
    const { Database } = await import("bun:sqlite");
    return new Database(path) as unknown as SqliteDatabase;
  }
  try {
    const { DatabaseSync } = await import("node:sqlite");
    return new DatabaseSync(path) as unknown as SqliteDatabase;
  } catch (error) {
    throw new PersistenceError("SQLite persistence requires Bun or Node >= 22.5.", { cause: error });
  }
}

/** Built-in SQLite backend: WAL, optimistic revisions, upserts, and tombstones. */
export class SqlitePersister implements Persister {
  private db: SqliteDatabase | null = null;
  constructor(readonly path: string) {}

  private async open(): Promise<SqliteDatabase> {
    if (this.db) return this.db;
    await mkdir(dirname(this.path), { recursive: true, mode: 0o700 });
    try {
      const db = await openSqlite(this.path);
      db.exec(`
        PRAGMA busy_timeout = 5000;
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS memories (
          id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tombstones (
          id TEXT PRIMARY KEY, deleted_at REAL NOT NULL
        );
      `);
      this.migrateLegacySchema(db);
      db.prepare("INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version', ?)").run(
        String(STORE_SCHEMA_VERSION),
      );
      db.prepare("INSERT OR IGNORE INTO meta(key,value) VALUES('revision', '0')").run();
      db.prepare("INSERT OR IGNORE INTO meta(key,value) VALUES('embedding_fingerprint', '')").run();
      this.db = db;
      await chmod(this.path, 0o600);
      return db;
    } catch (error) {
      if (/malformed|not a database|SQLITE_CORRUPT|SQLITE_NOTADB/i.test(String(error))) {
        throw new PersistenceError(
          `SQLite memory database is corrupt: ${this.path}. The file was preserved.`,
          { cause: error },
        );
      }
      throw error;
    }
  }

  private migrateLegacySchema(db: SqliteDatabase): void {
    const columns = db.prepare("PRAGMA table_info(memories)").all() as Array<{ name: string }>;
    if (columns.some((column) => column.name === "payload")) return;
    if (!columns.some((column) => column.name === "text")) {
      throw new PersistenceError(`Unsupported legacy SQLite memory schema: ${this.path}`);
    }
    const rows = db.prepare("SELECT * FROM memories").all() as Array<Record<string, unknown>>;
    db.exec("BEGIN IMMEDIATE");
    try {
      db.exec("ALTER TABLE memories RENAME TO memories_legacy_v1");
      db.exec("CREATE TABLE memories (id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL)");
      const insert = db.prepare("INSERT INTO memories(id,payload,updated_at) VALUES(?,?,?)");
      for (const row of rows) {
        const blob = row.vector as Uint8Array;
        const bytes = blob instanceof Uint8Array ? blob.slice() : new Uint8Array();
        const vector = bytes.byteLength % 8 === 0 ? Array.from(new Float64Array(bytes.buffer)) : [];
        const createdAt = Number(row.created_at ?? 0);
        const record = normalizeRecord({
          id: String(row.id ?? ""),
          kind: row.kind,
          text: row.text,
          vector,
          createdAt,
          updatedAt: createdAt,
          lastAccessed: Number(row.last_accessed ?? createdAt),
          baseStrength: Number(row.base_strength ?? 1),
          accessCount: Number(row.access_count ?? 0),
          surpriseAtWrite: Number(row.surprise_at_write ?? 0),
          supersededBy: row.superseded_by ?? null,
          sourceIds: typeof row.source_ids === "string" ? JSON.parse(row.source_ids) : [],
          metadata: typeof row.metadata === "string" ? JSON.parse(row.metadata) : {},
        });
        insert.run(record.id, JSON.stringify(record), record.updatedAt);
      }
      db.exec("DROP TABLE memories_legacy_v1");
      db.prepare(
        "INSERT INTO meta(key,value) VALUES('embedding_fingerprint','legacy:unknown') ON CONFLICT(key) DO UPDATE SET value=excluded.value",
      ).run();
      db.exec("COMMIT");
    } catch (error) {
      try {
        db.exec("ROLLBACK");
      } catch {}
      throw new PersistenceError(`Failed to migrate legacy SQLite memory database ${this.path}.`, {
        cause: error,
      });
    }
  }

  async load(): Promise<PersistenceSnapshot | null> {
    const db = await this.open();
    try {
      const metaRows = db.prepare("SELECT key,value FROM meta").all() as Array<{
        key: string;
        value: string;
      }>;
      const meta = new Map(metaRows.map((row) => [row.key, row.value]));
      const records = (
        db.prepare("SELECT payload FROM memories ORDER BY id").all() as Array<{ payload: string }>
      ).map((row) => normalizeRecord(JSON.parse(row.payload)));
      const tombstones = (
        db.prepare("SELECT id,deleted_at FROM tombstones").all() as Array<{ id: string; deleted_at: number }>
      ).map((row) => ({ id: row.id, deletedAt: row.deleted_at }));
      return {
        schemaVersion: Number(meta.get("schema_version") ?? STORE_SCHEMA_VERSION),
        revision: Number(meta.get("revision") ?? 0),
        embeddingFingerprint: meta.get("embedding_fingerprint") ?? "",
        updatedAt: Number(meta.get("updated_at") ?? 0),
        records,
        tombstones,
      };
    } catch (error) {
      if (error instanceof ValidationError) throw error;
      throw new PersistenceError(`Failed to load SQLite memory database ${this.path}.`, { cause: error });
    }
  }

  async save(snapshot: PersistenceSnapshot, expectedRevision: number): Promise<number> {
    const db = await this.open();
    db.exec("BEGIN IMMEDIATE");
    try {
      const actual = Number(
        (db.prepare("SELECT value FROM meta WHERE key='revision'").get() as { value?: string } | undefined)
          ?.value ?? 0,
      );
      if (actual !== expectedRevision)
        throw new PersistenceConflictError(
          `Memory database changed concurrently at ${this.path}.`,
          expectedRevision,
          actual,
        );
      const revision = actual + 1;
      const upsert = db.prepare(
        "INSERT INTO memories(id,payload,updated_at) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
      );
      const removeTombstone = db.prepare("DELETE FROM tombstones WHERE id=?");
      for (const record of snapshot.records) {
        const normalized = normalizeRecord(record);
        upsert.run(normalized.id, JSON.stringify(normalized), normalized.updatedAt);
        removeTombstone.run(normalized.id);
      }
      const tombstone = db.prepare(
        "INSERT INTO tombstones(id,deleted_at) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET deleted_at=MAX(deleted_at,excluded.deleted_at)",
      );
      const removeRecord = db.prepare("DELETE FROM memories WHERE id=?");
      for (const deleted of snapshot.tombstones) {
        tombstone.run(deleted.id, deleted.deletedAt);
        removeRecord.run(deleted.id);
      }
      db.prepare(
        "INSERT INTO meta(key,value) VALUES('revision',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
      ).run(String(revision));
      db.prepare(
        "INSERT INTO meta(key,value) VALUES('schema_version',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
      ).run(String(STORE_SCHEMA_VERSION));
      db.prepare(
        "INSERT INTO meta(key,value) VALUES('embedding_fingerprint',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
      ).run(snapshot.embeddingFingerprint);
      db.prepare(
        "INSERT INTO meta(key,value) VALUES('updated_at',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
      ).run(String(Date.now() / 1000));
      db.exec("COMMIT");
      return revision;
    } catch (error) {
      try {
        db.exec("ROLLBACK");
      } catch {}
      throw error;
    }
  }

  async close(): Promise<void> {
    if (!this.db) return;
    try {
      this.db.exec("PRAGMA wal_checkpoint(TRUNCATE)");
    } catch {}
    this.db.close();
    this.db = null;
  }

  health(): PersistenceHealth {
    return { backend: "sqlite", path: this.path, schemaVersion: STORE_SCHEMA_VERSION };
  }
}
