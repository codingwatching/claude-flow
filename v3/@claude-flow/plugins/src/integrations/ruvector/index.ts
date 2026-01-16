/**
 * RuVector PostgreSQL Bridge
 *
 * Integration module for RuVector - a PostgreSQL extension providing
 * advanced vector search, attention mechanisms, graph neural networks,
 * and hyperbolic embeddings.
 *
 * @module @claude-flow/plugins/integrations/ruvector
 */

// Export the main bridge plugin
export { RuVectorBridge, createRuVectorBridge } from './ruvector-bridge.js';
export { default as RuVectorBridgeDefault } from './ruvector-bridge.js';

// Export all types
export * from './types.js';

// Export GNN module
export * from './gnn.js';

// Export hyperbolic embeddings module
export * from './hyperbolic.js';

// Re-export commonly used types for convenience
export type {
  // Configuration
  RuVectorConfig,
  RuVectorClientOptions,
  SSLConfig,
  PoolConfig,
  RetryConfig,

  // Vector Operations
  VectorSearchOptions,
  VectorSearchResult,
  VectorInsertOptions,
  VectorUpdateOptions,
  VectorIndexOptions,
  BatchVectorOptions,
  DistanceMetric,
  VectorIndexType,

  // Attention
  AttentionMechanism,
  AttentionConfig,
  AttentionInput,
  AttentionOutput,
  AttentionParams,

  // GNN
  GNNLayerType,
  GNNLayer,
  GNNLayerParams,
  GraphData,
  GNNOutput,
  GNNAggregation,

  // Hyperbolic
  HyperbolicModel,
  HyperbolicEmbedding,
  HyperbolicInput,
  HyperbolicOutput,
  HyperbolicOperation,

  // Events
  RuVectorEventType,
  RuVectorEvent,
  RuVectorEventHandler,
  RuVectorEventEmitter,

  // Results
  Result,
  AsyncResult,
  ConnectionResult,
  QueryResult,
  BatchResult,
  TransactionResult,
  BulkSearchResult,
  EmbeddingResult,

  // Client Interface
  IRuVectorClient,
  IRuVectorTransaction,

  // Utilities
  RuVectorStats,
  HealthStatus,
  IndexStats,
} from './types.js';

// Export type guards
export {
  isDistanceMetric,
  isAttentionMechanism,
  isGNNLayerType,
  isHyperbolicModel,
  isVectorIndexType,
  isSuccess,
  isError,
} from './types.js';

// Export namespace
export { RuVector } from './types.js';

// Export GNN module classes and utilities
export {
  // Core classes
  GNNLayerRegistry,
  BaseGNNLayer,
  GraphOperations,
  GNNSQLGenerator,
  GNNEmbeddingCache,

  // Layer implementations
  GCNLayer,
  GATLayer,
  GATv2Layer,
  GraphSAGELayer,
  GINLayer,
  MPNNLayer,
  EdgeConvLayer,
  PointConvLayer,
  GraphTransformerLayer,
  PNALayer,
  FiLMLayer,
  RGCNLayer,
  HGTLayer,
  HANLayer,
  MetaPathLayer,

  // Factory functions
  createGNNLayerRegistry,
  createGNNLayer,
  createGraphOperations,

  // Constants
  GNN_DEFAULTS,
  GNN_SQL_FUNCTIONS,

  // Types
  type NodeId,
  type NodeFeatures,
  type EdgeFeatures,
  type Message,
  type AggregationMethod,
  type Path,
  type Community,
  type PageRankOptions,
  type CommunityOptions,
  type GNNLayerConfig,
  type GNNLayerFactory,
  type IGNNLayer,
  type SQLGenerationOptions,
} from './gnn.js';

// Export hyperbolic module classes and utilities
export {
  // Core class
  HyperbolicSpace,

  // SQL generation
  HyperbolicSQL,

  // Batch processing
  HyperbolicBatchProcessor,

  // Use case implementations
  HierarchyEmbedder,
  ASTEmbedder,
  DependencyGraphEmbedder,

  // Factory functions
  createHyperbolicSpace,
  fromEmbeddingConfig,
  validatePoint,

  // Types
  type HyperbolicSpaceConfig,
  type HyperbolicDistanceResult,
  type HyperbolicSearchResult,
  type HyperbolicBatchOptions,
  type HyperbolicBatchResult,
  type ASTNode,
  type TreeNode,
} from './hyperbolic.js';

// Export self-learning optimization module
export * from './self-learning.js';

// Export self-learning classes and utilities
export {
  // Core classes
  QueryOptimizer,
  IndexTuner,
  PatternRecognizer,
  LearningLoop,

  // Factory function
  createSelfLearningSystem,

  // Configuration presets
  DEFAULT_LEARNING_CONFIG,
  HIGH_PERF_LEARNING_CONFIG,
  HIGH_ACCURACY_LEARNING_CONFIG,

  // Types
  type QueryAnalysis,
  type QueryType,
  type VectorOperation,
  type IndexHint,
  type Bottleneck,
  type Optimization,
  type OptimizationType,
  type QueryExecutionStats,
  type WorkloadAnalysis,
  type QueryPattern,
  type TableAccess,
  type IndexUsageSummary,
  type WorkloadCharacteristics,
  type WorkloadRecommendation,
  type IndexSuggestion,
  type HNSWParams,
  type QueryHistory,
  type Pattern,
  type PatternType,
  type TemporalPattern,
  type PerformancePattern,
  type Context,
  type Anomaly,
  type AnomalyType,
  type LearningConfig,
  type LearningStats,
  type EWCState,
} from './self-learning.js';

// ============================================================================
// Attention Module Exports
// ============================================================================

// Export attention module
export * from './attention.js';
export * from './attention-mechanisms.js';
export * from './attention-advanced.js';
export * from './attention-executor.js';

// Core attention exports
export {
  AttentionRegistry,
  BaseAttentionMechanism,
  type IAttentionMechanism,
  type AttentionOptions,
  type AttentionCategory,
  MultiHeadAttention,
  SelfAttention,
  CrossAttention,
  CausalAttention,
  BidirectionalAttention,
  LocalAttention,
  GlobalAttention,
  FlashAttention,
  FlashAttentionV2,
  MemoryEfficientAttention,
  ChunkAttention,
  SlidingWindowAttention,
  DilatedAttention,
} from './attention.js';

// Additional attention mechanisms
export {
  SparseAttention,
  BlockSparseAttention,
  LinearAttention,
  PerformerAttention,
  LinformerAttention,
  ReformerAttention,
  RelativePositionAttention,
  RotaryPositionAttention,
  ALiBiAttention,
  AxialAttention,
} from './attention-mechanisms.js';

// Advanced attention mechanisms
export {
  GraphAttention,
  HyperbolicAttention,
  SphericalAttention,
  ToroidalAttention,
  TemporalAttention,
  RecurrentAttention,
  StateSpaceAttention,
  CrossModalAttention,
  PerceiverAttention,
  FlamingoAttention,
  RetrievalAttention,
  KNNAttention,
  MemoryAugmentedAttention,
  SynthesizerAttention,
  RoutingAttention,
  MixtureOfExpertsAttention,
} from './attention-advanced.js';

// Executor and factory
export {
  AttentionExecutor,
  AttentionFactory,
  AttentionStack,
  AttentionSQLBuilder,
  createDefaultRegistry,
  type ExecutionOptions,
  type ExecutionResult,
} from './attention-executor.js';
