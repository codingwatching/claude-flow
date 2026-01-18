/**
 * Memory Backend Interface
 *
 * All custom memory backends must implement this interface.
 * This allows swapping storage implementations without changing the interceptor.
 */

export interface MemoryEntry {
  key: string;
  value: unknown;
  metadata?: Record<string, unknown>;
  timestamp: number;
  ttl?: number;
  embedding?: number[];
}

export interface SearchResult {
  key: string;
  value: unknown;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface MemoryStats {
  totalEntries: number;
  totalSizeBytes: number;
  oldestEntry?: number;
  newestEntry?: number;
  namespaces?: string[];
}

export interface MemoryBackend {
  /** Backend name for logging */
  readonly name: string;

  /** Initialize the backend */
  init(): Promise<void>;

  /** Close connections and cleanup */
  close(): Promise<void>;

  /** Store a value */
  store(key: string, value: unknown, metadata?: Record<string, unknown>): Promise<void>;

  /** Retrieve a value */
  retrieve(key: string): Promise<MemoryEntry | null>;

  /** Delete a value */
  delete(key: string): Promise<boolean>;

  /** List entries with optional pagination */
  list(options?: { limit?: number; offset?: number; prefix?: string }): Promise<MemoryEntry[]>;

  /** Search by keyword or semantic similarity */
  search(query: string, options?: { limit?: number; threshold?: number }): Promise<SearchResult[]>;

  /** Get storage statistics */
  stats(): Promise<MemoryStats>;

  /** Clear all entries (optional namespace) */
  clear(namespace?: string): Promise<void>;

  /** Check if backend is healthy */
  health(): Promise<boolean>;
}

export interface MemoryBackendConfig {
  type: 'sqlite' | 'redis' | 'postgres' | 'custom';
  connectionString?: string;
  options?: Record<string, unknown>;
}
