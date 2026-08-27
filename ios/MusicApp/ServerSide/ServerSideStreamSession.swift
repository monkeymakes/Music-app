import Foundation
import UIKit

/// One item in a server-side batch inference response.
public struct ServerSideBatchItem: Decodable, Sendable {
  public var index: Int
  public var inferenceId: String
  public var latencyMs: Double
  public var labels: [ServerSideLabel]

  enum CodingKeys: String, CodingKey {
    case index, labels
    case inferenceId = "inference_id"
    case latencyMs = "latency_ms"
  }
}

/// Response from `/v1/infer/batch`.
public struct ServerSideBatchResponse: Decodable, Sendable {
  public var batchId: String
  public var model: String
  public var channel: String
  public var version: String
  public var architecture: String
  public var deviceClass: String
  public var runtime: String
  public var engineMode: String
  public var count: Int
  public var totalLatencyMs: Double
  public var results: [ServerSideBatchItem]

  enum CodingKeys: String, CodingKey {
    case model, channel, version, architecture, runtime, count, results
    case batchId = "batch_id"
    case deviceClass = "device_class"
    case engineMode = "engine_mode"
    case totalLatencyMs = "total_latency_ms"
  }
}

/// Server stream session metadata from `/v1/stream/sessions`.
public struct ServerSideStreamSessionInfo: Decodable, Sendable {
  public var sessionId: String
  public var model: String
  public var channel: String
  public var version: String?
  public var topK: Int
  public var batchSize: Int
  public var maxWaitMs: Int
  public var idleTtlMs: Int
  public var samplingMode: String
  public var sampleWindow: Int
  public var pendingFrames: Int
  public var totalFramesReceived: Int
  public var totalBatchesFlushed: Int
  public var framesSinceFlush: Int

  enum CodingKeys: String, CodingKey {
    case model, channel, version
    case sessionId = "session_id"
    case topK = "top_k"
    case batchSize = "batch_size"
    case maxWaitMs = "max_wait_ms"
    case idleTtlMs = "idle_ttl_ms"
    case samplingMode = "sampling_mode"
    case sampleWindow = "sample_window"
    case pendingFrames = "pending_frames"
    case totalFramesReceived = "total_frames_received"
    case totalBatchesFlushed = "total_batches_flushed"
    case framesSinceFlush = "frames_since_flush"
  }
}

/// How the server selects frames from a high-frequency camera stream.
public enum StreamSamplingMode: String, Sendable {
  /// Keep frames in arrival order until `batchSize` (legacy behavior).
  case fifo
  /// Algorithm A-Res weighted reservoir of capacity `batchSize`.
  case weightedReservoir = "weighted_reservoir"
}

/// One frame result inside a stream flush payload.
public struct ServerSideStreamFrameResult: Decodable, Sendable {
  public var frameId: String
  public var clientIndex: Int?
  public var index: Int
  public var latencyMs: Double
  public var sampleWeight: Double
  public var labels: [ServerSideLabel]

  enum CodingKeys: String, CodingKey {
    case index, labels
    case frameId = "frame_id"
    case clientIndex = "client_index"
    case latencyMs = "latency_ms"
    case sampleWeight = "sample_weight"
  }
}

/// Batch flush result from a stream session.
public struct ServerSideStreamFlushResponse: Decodable, Sendable {
  public var batchId: String
  public var sessionId: String
  public var flushedCount: Int
  public var pendingFrames: Int
  public var totalLatencyMs: Double
  public var engineMode: String
  public var samplingMode: String
  public var framesConsidered: Int
  public var results: [ServerSideStreamFrameResult]

  enum CodingKeys: String, CodingKey {
    case results
    case batchId = "batch_id"
    case sessionId = "session_id"
    case flushedCount = "flushed_count"
    case pendingFrames = "pending_frames"
    case totalLatencyMs = "total_latency_ms"
    case engineMode = "engine_mode"
    case samplingMode = "sampling_mode"
    case framesConsidered = "frames_considered"
  }
}

/// Response after pushing one frame to a stream session.
public struct ServerSideStreamFrameResponse: Decodable, Sendable {
  public var frameId: String
  public var sessionId: String
  public var pendingFrames: Int
  public var selected: Bool
  public var sampleWeight: Double
  public var autoFlushed: Bool
  public var flush: ServerSideStreamFlushResponse?

  enum CodingKeys: String, CodingKey {
    case flush, selected
    case frameId = "frame_id"
    case sessionId = "session_id"
    case pendingFrames = "pending_frames"
    case sampleWeight = "sample_weight"
    case autoFlushed = "auto_flushed"
  }
}

/// Manages a server-side stream session that batches camera frames before inference.
public final class ServerSideStreamSession: @unchecked Sendable {
  public private(set) var sessionInfo: ServerSideStreamSessionInfo?
  public private(set) var isOpen = false

  private let client: ServerSideInferenceClient
  private var clientFrameIndex = 0

  public init(client: ServerSideInferenceClient) {
    self.client = client
  }

  /// Opens a stream session on the inference server.
  @discardableResult
  public func open(
    batchSize: Int = 8,
    maxWaitMs: Int = 500,
    idleTtlMs: Int = 60_000,
    samplingMode: StreamSamplingMode = .fifo,
    sampleWindow: Int = 0
  ) async throws -> ServerSideStreamSessionInfo {
    if !client.isReady {
      _ = try await client.prepare()
    }
    let info = try await client.createStreamSession(
      batchSize: batchSize,
      maxWaitMs: maxWaitMs,
      idleTtlMs: idleTtlMs,
      samplingMode: samplingMode,
      sampleWindow: sampleWindow
    )
    sessionInfo = info
    isOpen = true
    clientFrameIndex = 0
    return info
  }

  /// Pushes one JPEG frame; returns flush results when the server auto-batches.
  /// - Parameter sampleWeight: Relative importance for weighted reservoir sessions (`> 0`).
  public func pushFrame(
    _ image: UIImage,
    sampleWeight: Double = 1.0
  ) async throws -> ServerSideStreamFrameResponse {
    guard isOpen else { throw ServerSideInferenceError.streamNotOpen }
    let index = clientFrameIndex
    clientFrameIndex += 1
    let response = try await client.pushStreamFrame(
      image: image,
      clientIndex: index,
      sampleWeight: sampleWeight
    )
    if let flush = response.flush {
      ModelPerformanceAnalytics.logStreamBatch(
        flushedCount: flush.flushedCount,
        totalLatencyMs: flush.totalLatencyMs,
        engineMode: flush.engineMode
      )
    }
    return response
  }

  /// Forces the server to flush any pending frames in the session.
  public func flush() async throws -> ServerSideStreamFlushResponse {
    guard isOpen else { throw ServerSideInferenceError.streamNotOpen }
    let flush = try await client.flushStreamSession()
    ModelPerformanceAnalytics.logStreamBatch(
      flushedCount: flush.flushedCount,
      totalLatencyMs: flush.totalLatencyMs,
      engineMode: flush.engineMode
    )
    return flush
  }

  /// Closes the session on the server.
  public func close() async throws {
    guard isOpen else { return }
    try await client.deleteStreamSession()
    isOpen = false
    sessionInfo = nil
  }
}
