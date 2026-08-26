/** Typed errors surfaced by SurMem instead of silently discarding data. */

export class SurMemError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}

export class ValidationError extends SurMemError {}
export class PersistenceError extends SurMemError {}
export class PersistenceConflictError extends PersistenceError {
  constructor(
    message: string,
    readonly expectedRevision: number,
    readonly actualRevision: number,
  ) {
    super(message);
  }
}
export class EmbeddingMismatchError extends SurMemError {}
export class SensitiveContentError extends SurMemError {
  constructor(
    message: string,
    readonly findings: string[],
  ) {
    super(message);
  }
}
