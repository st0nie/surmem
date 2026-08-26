/** Strict, atomic configuration handling for the Pi extension. */

import { randomUUID } from "node:crypto";
import { constants } from "node:fs";
import {
  chmod,
  type FileHandle,
  mkdir,
  open,
  readFile,
  rename,
  stat,
  unlink,
  writeFile,
} from "node:fs/promises";
import { dirname } from "node:path";
import { ValidationError } from "./errors";

export interface ExtensionConfig {
  tauAdd: number;
  dupSim: number;
  conflictSim: number;
  minTokens: number;
  decayRatePerHour: number;
  semanticDecayRatePerHour: number;
  forgetThreshold: number;
  snapshotSize: number;
  autoCandidates: boolean;
  autoMaintenance: boolean;
  sessionSearch: boolean;
}

export const DEFAULT_EXTENSION_CONFIG: ExtensionConfig = {
  tauAdd: 0.45,
  dupSim: 0.85,
  conflictSim: 0.55,
  minTokens: 3,
  decayRatePerHour: 0.02,
  semanticDecayRatePerHour: 0.002,
  forgetThreshold: 0.1,
  snapshotSize: 8,
  autoCandidates: true,
  autoMaintenance: true,
  sessionSearch: true,
};

function numberIn(
  value: unknown,
  fallback: number,
  min: number,
  max: number,
  name: string,
  integer = false,
): number {
  if (value === undefined) return fallback;
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < min ||
    value > max ||
    (integer && !Number.isInteger(value))
  ) {
    throw new ValidationError(
      `${name} must be ${integer ? "an integer" : "a number"} between ${min} and ${max}.`,
    );
  }
  return value;
}

export function normalizeExtensionConfig(value: unknown): ExtensionConfig {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    if (value == null) return { ...DEFAULT_EXTENSION_CONFIG };
    throw new ValidationError("SurMem config must be a JSON object.");
  }
  const raw = value as Partial<ExtensionConfig>;
  const config: ExtensionConfig = {
    tauAdd: numberIn(raw.tauAdd, DEFAULT_EXTENSION_CONFIG.tauAdd, 0, 1, "tauAdd"),
    dupSim: numberIn(raw.dupSim, DEFAULT_EXTENSION_CONFIG.dupSim, 0, 1, "dupSim"),
    conflictSim: numberIn(raw.conflictSim, DEFAULT_EXTENSION_CONFIG.conflictSim, 0, 1, "conflictSim"),
    minTokens: numberIn(raw.minTokens, DEFAULT_EXTENSION_CONFIG.minTokens, 1, 100, "minTokens", true),
    decayRatePerHour: numberIn(
      raw.decayRatePerHour,
      DEFAULT_EXTENSION_CONFIG.decayRatePerHour,
      0,
      10,
      "decayRatePerHour",
    ),
    semanticDecayRatePerHour: numberIn(
      raw.semanticDecayRatePerHour,
      DEFAULT_EXTENSION_CONFIG.semanticDecayRatePerHour,
      0,
      10,
      "semanticDecayRatePerHour",
    ),
    forgetThreshold: numberIn(
      raw.forgetThreshold,
      DEFAULT_EXTENSION_CONFIG.forgetThreshold,
      0,
      100,
      "forgetThreshold",
    ),
    snapshotSize: numberIn(
      raw.snapshotSize,
      DEFAULT_EXTENSION_CONFIG.snapshotSize,
      0,
      50,
      "snapshotSize",
      true,
    ),
    autoCandidates: raw.autoCandidates ?? DEFAULT_EXTENSION_CONFIG.autoCandidates,
    autoMaintenance: raw.autoMaintenance ?? DEFAULT_EXTENSION_CONFIG.autoMaintenance,
    sessionSearch: raw.sessionSearch ?? DEFAULT_EXTENSION_CONFIG.sessionSearch,
  };
  if (
    typeof config.autoCandidates !== "boolean" ||
    typeof config.autoMaintenance !== "boolean" ||
    typeof config.sessionSearch !== "boolean"
  ) {
    throw new ValidationError("Boolean SurMem config values are invalid.");
  }
  if (config.conflictSim >= config.dupSim)
    throw new ValidationError("conflictSim must be lower than dupSim.");
  return config;
}

export async function loadExtensionConfig(path: string): Promise<ExtensionConfig> {
  try {
    const info = await stat(path);
    if (!info.isFile()) throw new ValidationError(`SurMem config is not a regular file: ${path}`);
    if (info.size > 64 * 1024) throw new ValidationError("SurMem config exceeds 64 KiB.");
    return normalizeExtensionConfig(JSON.parse(await readFile(path, "utf8")));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { ...DEFAULT_EXTENSION_CONFIG };
    throw error;
  }
}

export async function saveExtensionConfig(path: string, config: ExtensionConfig): Promise<void> {
  const normalized = normalizeExtensionConfig(config);
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const lockPath = `${path}.lock`;
  let handle: FileHandle | undefined;
  const deadline = Date.now() + 5000;
  while (!handle) {
    try {
      handle = await open(lockPath, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY, 0o600);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") {
        throw new Error(`Unable to lock SurMem config: ${path}`, { cause: error });
      }
      try {
        const info = await stat(lockPath);
        if (Date.now() - info.mtimeMs > 5 * 60_000) await unlink(lockPath);
      } catch {}
      if (Date.now() >= deadline) {
        throw new Error(`SurMem config is being modified by another process: ${path}`, { cause: error });
      }
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
  }
  const temp = `${path}.tmp-${process.pid}-${randomUUID()}`;
  try {
    await handle.writeFile(String(process.pid));
    await handle.close();
    handle = undefined;
    await writeFile(temp, `${JSON.stringify(normalized, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600,
      flag: "wx",
    });
    await rename(temp, path);
    await chmod(path, 0o600);
  } finally {
    await handle?.close().catch(() => {});
    await unlink(temp).catch(() => {});
    await unlink(lockPath).catch(() => {});
  }
}
