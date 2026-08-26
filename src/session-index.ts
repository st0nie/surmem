import { chmod, mkdir, readdir, readFile, stat } from "node:fs/promises";
import { basename, dirname, extname, resolve } from "node:path";

export type SessionSearchResult = {
  sessionId: string;
  path: string;
  cwd: string | null;
  timestamp: string | null;
  role: "user" | "assistant" | "system";
  content: string;
  score: number;
};

export type IndexFileResult = {
  indexed: boolean;
  skipped: boolean;
  sessionId: string;
  messages: number;
  path: string;
};

export type IndexDirectoryOptions = { limit?: number };
export type SearchOptions = { limit?: number; cwd?: string };

export type SessionIndexStats = {
  sessions: number;
  files: number;
  messages: number;
  tokenizer: string;
  schemaVersion: number;
};

type SqliteStatement = {
  run: (...params: unknown[]) => unknown;
  all: (...params: unknown[]) => unknown[];
  get: (...params: unknown[]) => unknown;
};

type SqliteDatabase = {
  exec(sql: string): unknown;
  prepare?(sql: string): SqliteStatement;
  query?(sql: string): SqliteStatement;
  close(): void;
};

type MessageRow = {
  sessionId: string;
  path: string;
  entryId: string;
  timestamp: string | null;
  role: "user" | "assistant" | "system";
  content: string;
};

type ParsedSession = {
  sessionId: string;
  cwd: string | null;
  timestamp: string | null;
  messages: MessageRow[];
};

const SCHEMA_VERSION = 1;
const DEFAULT_DIRECTORY_LIMIT = 100;
const DEFAULT_SEARCH_LIMIT = 20;
const MAX_CONTENT_LENGTH = 1200;
const BUSY_TIMEOUT_MS = 5000;

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function sqliteError(path: string, error: unknown): Error {
  const message = errorMessage(error);
  return new Error(`SQLite database error for ${path}: ${message}`, { cause: error });
}

function valueString(value: unknown): string | null {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "bigint") return String(value);
  return null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function textFromContent(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        const object = asRecord(item);
        if (!object) return textFromContent(item);
        if (typeof object.text === "string") return object.text;
        if (object.type === "text" || object.type === "input_text") {
          return textFromContent(object.content);
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  const object = asRecord(value);
  if (object && typeof object.text === "string") return object.text;
  return "";
}

function parseSessionJsonl(input: string, filePath: string): ParsedSession {
  const absolutePath = resolve(filePath);
  let sessionId: string | null = null;
  let cwd: string | null = null;
  let sessionTimestamp: string | null = null;
  const messages: MessageRow[] = [];
  let lineNumber = 0;

  for (const line of input.split(/\r?\n/)) {
    lineNumber += 1;
    if (!line.trim()) continue;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      // A damaged line must not make an otherwise usable session disappear.
      continue;
    }
    const record = asRecord(value);
    if (!record) continue;

    const type = typeof record.type === "string" ? record.type : "";
    const isHeader =
      type === "session" ||
      type === "session_header" ||
      (typeof record.cwd === "string" && !record.message && !record.role);
    if (isHeader) {
      sessionId = valueString(record.id) ?? valueString(record.sessionId) ?? sessionId;
      cwd = valueString(record.cwd) ?? cwd;
      sessionTimestamp = valueString(record.timestamp) ?? sessionTimestamp;
    }

    const message = asRecord(record.message) ?? record;
    const role = valueString(message.role);
    if (role !== "user" && role !== "assistant" && role !== "system") continue;
    if (type === "tool_result" || type === "toolResult") continue;
    const content = textFromContent(message.content);
    if (!content) continue;
    const timestamp = valueString(message.timestamp) ?? valueString(record.timestamp) ?? sessionTimestamp;
    const entryId = valueString(record.id) ?? valueString(message.id) ?? `${lineNumber}`;
    messages.push({
      sessionId: sessionId ?? "",
      path: absolutePath,
      entryId,
      timestamp,
      role,
      content,
    });
  }

  const finalSessionId = sessionId ?? basename(filePath, extname(filePath));
  for (const message of messages) message.sessionId = finalSessionId;
  return { sessionId: finalSessionId, cwd, timestamp: sessionTimestamp, messages };
}

function clampSearchLimit(value: number | undefined): number {
  if (value === undefined) return DEFAULT_SEARCH_LIMIT;
  if (!Number.isFinite(value)) throw new RangeError("search limit must be a finite number");
  return Math.min(20, Math.max(1, Math.floor(value)));
}

function tokenizeQuery(query: string): string[] {
  const tokens: string[] = [];
  const pattern = /"([^"\\]*(?:\\.[^"\\]*)*)"|([^\s]+)/gu;
  for (const match of query.matchAll(pattern)) {
    const raw = match[1] ?? match[2] ?? "";
    const unescaped = raw.replace(/\\(["\\])/g, "$1");
    const words = unescaped.match(/[\p{L}\p{N}_]+/gu) ?? [];
    if (words.length) tokens.push(words.join(" "));
  }
  return tokens;
}

function quoteFtsToken(token: string): string {
  return `"${token.replaceAll('"', '""')}"`;
}

function normalizedFtsQuery(query: string, joiner: "AND" | "OR"): string {
  return tokenizeQuery(query).map(quoteFtsToken).join(` ${joiner} `);
}

function isShortCjk(query: string): boolean {
  const compact = query.trim();
  if (!compact || /["*:?()[\]]/u.test(compact)) return false;
  const chars = [...compact].filter((char) => /\p{Script=Han}/u.test(char));
  return chars.length >= 1 && chars.length <= 2 && chars.join("") === compact;
}

export class SessionIndex {
  private dbPromise: Promise<SqliteDatabase> | null = null;
  private closed = false;
  private tokenizer = "unicode61";
  private readonly dbPath: string;

  constructor(dbPath: string) {
    if (!dbPath) throw new TypeError("dbPath is required");
    this.dbPath = dbPath;
  }

  private async open(): Promise<SqliteDatabase> {
    if (this.closed) {
      this.closed = false;
      this.dbPromise = null;
    }
    if (!this.dbPromise) this.dbPromise = this.openDatabase();
    return this.dbPromise;
  }

  private async openDatabase(): Promise<SqliteDatabase> {
    const path = resolve(this.dbPath);
    try {
      await mkdir(dirname(path), { recursive: true, mode: 0o700 });
      let Constructor: new (path: string) => SqliteDatabase;
      if ("Bun" in globalThis) {
        const module = await import("bun:sqlite");
        Constructor = module.Database as unknown as new (path: string) => SqliteDatabase;
      } else {
        const module = await import("node:sqlite");
        Constructor = module.DatabaseSync as unknown as new (path: string) => SqliteDatabase;
      }
      const db = new Constructor(path);
      await chmod(path, 0o600);
      db.exec(`PRAGMA busy_timeout=${BUSY_TIMEOUT_MS}; PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;`);
      this.initializeSchema(db, path);
      return db;
    } catch (error) {
      throw sqliteError(path, error);
    }
  }

  private statement(db: SqliteDatabase, sql: string): SqliteStatement {
    if (db.prepare) return db.prepare(sql);
    if (db.query) return db.query(sql);
    throw new Error("SQLite driver has no prepare/query API");
  }

  private run(db: SqliteDatabase, sql: string, ...params: unknown[]): unknown {
    return this.statement(db, sql).run(...params);
  }

  private all<T = Record<string, unknown>>(db: SqliteDatabase, sql: string, ...params: unknown[]): T[] {
    return this.statement(db, sql).all(...params) as T[];
  }

  private get<T = Record<string, unknown>>(
    db: SqliteDatabase,
    sql: string,
    ...params: unknown[]
  ): T | undefined {
    return this.statement(db, sql).get(...params) as T | undefined;
  }

  private initializeSchema(db: SqliteDatabase, path: string): void {
    try {
      db.exec(`
        CREATE TABLE IF NOT EXISTS metadata (
          key TEXT PRIMARY KEY NOT NULL,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
          session_id TEXT PRIMARY KEY NOT NULL,
          path TEXT NOT NULL UNIQUE,
          cwd TEXT,
          timestamp TEXT,
          size INTEGER NOT NULL,
          mtime REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS session_files (
          path TEXT PRIMARY KEY NOT NULL,
          session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
          size INTEGER NOT NULL,
          mtime REAL NOT NULL,
          indexed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
          path TEXT NOT NULL,
          entry_id TEXT NOT NULL,
          timestamp TEXT,
          role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
          content TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id);
        CREATE INDEX IF NOT EXISTS messages_timestamp_idx ON messages(timestamp);
      `);
      const version = this.get<{ value?: string }>(
        db,
        "SELECT value FROM metadata WHERE key='schema_version'",
      )?.value;
      if (version !== undefined && Number(version) !== SCHEMA_VERSION) {
        throw new Error(`unsupported schema version ${version}`);
      }
      this.run(
        db,
        "INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?)",
        String(SCHEMA_VERSION),
      );

      const existing = this.get<{ name?: string }>(
        db,
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'",
      );
      if (!existing) {
        try {
          db.exec(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(session_id UNINDEXED, path UNINDEXED, timestamp UNINDEXED, role UNINDEXED, content, tokenize='trigram')",
          );
          this.tokenizer = "trigram";
        } catch (error) {
          const message = errorMessage(error).toLowerCase();
          if (!message.includes("trigram") && !message.includes("tokenizer") && !message.includes("fts5"))
            throw error;
          db.exec(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(session_id UNINDEXED, path UNINDEXED, timestamp UNINDEXED, role UNINDEXED, content, tokenize='unicode61')",
          );
          this.tokenizer = "unicode61";
        }
      } else {
        const stored = this.get<{ value?: string }>(
          db,
          "SELECT value FROM metadata WHERE key='tokenizer'",
        )?.value;
        this.tokenizer = stored === "trigram" ? "trigram" : "unicode61";
      }
      this.run(db, "INSERT OR REPLACE INTO metadata(key,value) VALUES('tokenizer',?)", this.tokenizer);
    } catch (error) {
      throw sqliteError(path, error);
    }
  }

  async indexFile(sessionJsonlPath: string): Promise<IndexFileResult> {
    const path = resolve(sessionJsonlPath);
    const db = await this.open();
    let fileStat: Awaited<ReturnType<typeof stat>>;
    try {
      fileStat = await stat(path);
    } catch (error) {
      throw new Error(`Unable to stat session file ${path}: ${errorMessage(error)}`, { cause: error });
    }
    const size = fileStat.size;
    const mtime = fileStat.mtimeMs;
    let existing: { session_id?: string } | undefined;
    try {
      existing = this.get<{ session_id?: string }>(
        db,
        "SELECT session_id FROM session_files WHERE path=? AND size=? AND mtime=?",
        path,
        size,
        mtime,
      );
    } catch (error) {
      throw sqliteError(path, error);
    }
    if (existing?.session_id) {
      return { indexed: false, skipped: true, sessionId: existing.session_id, messages: 0, path };
    }

    let input: string;
    try {
      input = await readFile(path, "utf8");
    } catch (error) {
      throw new Error(`Unable to read session file ${path}: ${errorMessage(error)}`, { cause: error });
    }
    const parsed = parseSessionJsonl(input, path);
    try {
      db.exec("BEGIN IMMEDIATE");
      this.run(
        db,
        "DELETE FROM messages_fts WHERE session_id IN (SELECT session_id FROM sessions WHERE path=?)",
        path,
      );
      this.run(db, "DELETE FROM session_files WHERE path=?", path);
      this.run(db, "DELETE FROM sessions WHERE path=?", path);
      this.run(
        db,
        "INSERT INTO sessions(session_id,path,cwd,timestamp,size,mtime) VALUES(?,?,?,?,?,?)",
        parsed.sessionId,
        path,
        parsed.cwd,
        parsed.timestamp,
        size,
        mtime,
      );
      this.run(
        db,
        "INSERT INTO session_files(path,session_id,size,mtime,indexed_at) VALUES(?,?,?,?,?)",
        path,
        parsed.sessionId,
        size,
        mtime,
        new Date().toISOString(),
      );
      for (let index = 0; index < parsed.messages.length; index += 1) {
        const message = parsed.messages[index];
        this.run(
          db,
          "INSERT INTO messages(session_id,path,entry_id,timestamp,role,content) VALUES(?,?,?,?,?,?)",
          message.sessionId,
          path,
          message.entryId,
          message.timestamp,
          message.role,
          message.content,
        );
        const row = this.get<{ id?: number }>(db, "SELECT last_insert_rowid() AS id");
        this.run(
          db,
          "INSERT INTO messages_fts(rowid,session_id,path,timestamp,role,content) VALUES(?,?,?,?,?,?)",
          row?.id,
          message.sessionId,
          path,
          message.timestamp,
          message.role,
          message.content,
        );
      }
      db.exec("COMMIT");
    } catch (error) {
      try {
        db.exec("ROLLBACK");
      } catch {
        /* preserve original database error */
      }
      throw sqliteError(path, error);
    }
    return {
      indexed: true,
      skipped: false,
      sessionId: parsed.sessionId,
      messages: parsed.messages.length,
      path,
    };
  }

  async indexDirectory(root: string, options: IndexDirectoryOptions = {}): Promise<IndexFileResult[]> {
    const limitValue = options.limit ?? DEFAULT_DIRECTORY_LIMIT;
    if (!Number.isFinite(limitValue) || limitValue < 0)
      throw new RangeError("directory limit must be a non-negative finite number");
    const limit = Math.floor(limitValue);
    const files: Array<{ path: string; mtimeMs: number }> = [];
    const walk = async (directory: string): Promise<void> => {
      for (const entry of await readdir(directory, { withFileTypes: true })) {
        const path = resolve(directory, entry.name);
        if (entry.isDirectory()) await walk(path);
        else if (entry.isFile() && extname(entry.name).toLowerCase() === ".jsonl") {
          try {
            files.push({ path, mtimeMs: (await stat(path)).mtimeMs });
          } catch (error) {
            throw new Error(`Unable to stat session file ${path}: ${errorMessage(error)}`, { cause: error });
          }
        }
      }
    };
    try {
      await walk(resolve(root));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw new Error(`Unable to read session directory ${resolve(root)}: ${errorMessage(error)}`, {
        cause: error,
      });
    }
    files.sort((a, b) => b.mtimeMs - a.mtimeMs || a.path.localeCompare(b.path));
    const results: IndexFileResult[] = [];
    for (const file of files.slice(0, limit)) results.push(await this.indexFile(file.path));
    return results;
  }

  async search(query: string, options: SearchOptions = {}): Promise<SessionSearchResult[]> {
    const db = await this.open();
    const limit = clampSearchLimit(options.limit);
    const text = query.trim();
    if (!text) return [];
    const cwd = options.cwd;
    const cwdClause = cwd === undefined ? "" : " AND s.cwd=?";
    const cwdParams = cwd === undefined ? [] : [cwd];
    const select = `SELECT f.session_id AS sessionId, f.path AS path, s.cwd AS cwd, f.timestamp AS timestamp, f.role AS role, f.content AS content, -bm25(messages_fts) AS score FROM messages_fts f JOIN sessions s ON s.session_id=f.session_id WHERE f.messages_fts MATCH ?${cwdClause} ORDER BY score DESC, f.timestamp DESC LIMIT ?`;
    const likeSelect = `SELECT m.session_id AS sessionId, m.path AS path, s.cwd AS cwd, m.timestamp AS timestamp, m.role AS role, m.content AS content, 0.0 AS score FROM messages m JOIN sessions s ON s.session_id=m.session_id WHERE m.content LIKE ?${cwdClause} ORDER BY m.timestamp DESC LIMIT ?`;
    const mapRows = (rows: Array<Record<string, unknown>>): SessionSearchResult[] =>
      rows.map((row) => ({
        sessionId: String(row.sessionId),
        path: String(row.path),
        cwd: valueString(row.cwd),
        timestamp: valueString(row.timestamp),
        role: row.role as SessionSearchResult["role"],
        content: String(row.content).slice(0, MAX_CONTENT_LENGTH),
        score: Number(row.score) || 0,
      }));
    try {
      if (isShortCjk(text)) return mapRows(this.all(db, likeSelect, `%${text}%`, ...cwdParams, limit));
      const ftsQuery = normalizedFtsQuery(text, "AND");
      if (!ftsQuery) return [];
      return mapRows(this.all(db, select, ftsQuery, ...cwdParams, limit));
    } catch {
      try {
        const fallback = normalizedFtsQuery(text, "OR");
        if (!fallback) return [];
        return mapRows(this.all(db, select, fallback, ...cwdParams, limit));
      } catch (error) {
        throw sqliteError(resolve(this.dbPath), error);
      }
    }
  }

  async stats(): Promise<SessionIndexStats> {
    const db = await this.open();
    try {
      const sessions = this.get<{ count?: number }>(db, "SELECT count(*) AS count FROM sessions")?.count ?? 0;
      const files =
        this.get<{ count?: number }>(db, "SELECT count(*) AS count FROM session_files")?.count ?? 0;
      const messages = this.get<{ count?: number }>(db, "SELECT count(*) AS count FROM messages")?.count ?? 0;
      const tokenizer =
        this.get<{ value?: string }>(db, "SELECT value FROM metadata WHERE key='tokenizer'")?.value ??
        this.tokenizer;
      return {
        sessions: Number(sessions),
        files: Number(files),
        messages: Number(messages),
        tokenizer,
        schemaVersion: SCHEMA_VERSION,
      };
    } catch (error) {
      throw sqliteError(resolve(this.dbPath), error);
    }
  }

  async close(): Promise<void> {
    if (!this.dbPromise) {
      this.closed = true;
      return;
    }
    const promise = this.dbPromise;
    this.dbPromise = null;
    this.closed = true;
    const db = await promise;
    try {
      db.exec("PRAGMA wal_checkpoint(TRUNCATE)");
      db.close();
    } catch (error) {
      throw sqliteError(resolve(this.dbPath), error);
    }
  }
}
