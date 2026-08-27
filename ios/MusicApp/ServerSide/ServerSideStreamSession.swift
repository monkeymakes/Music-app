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
  public var pendingFrames: Int
  public var totalFramesReceived: Int
  public var totalBatchesFlushed: Int

  enum CodingKeys: String, CodingKey {
    case model, channel, version
    case sessionId = "session_id"
    case topK = "top_k"
    case batchSize = "batch_size"
    case maxWaitMs = "max_wait_ms"
    case idleTtlMs = "idle_ttl_ms"
    case pendingFrames = "pending_frames"
    case totalFramesReceived = "total_frames_received"
    case totalBatchesFlushed = "total_batches_flushed"
  }
}

/// One frame result inside a stream flush payload.
public struct ServerSideStreamFrameResult: Decodable, Sendable {
  public var frameId: String
  public var clientIndex: Int?
  public var index: Int
  public var latencyMs: Double
  public var labels: [ServerSideLabel]

  enum CodingKeys: String, CodingKey {
    case index, labels
    case frameId = "frame_id"
    case clientIndex = "client_index"
    case latencyMs = "latency_ms"
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
  public var results: [ServerSideStreamFrameResult]

  enum CodingKeys: String, CodingKey {
    case results
    case batchId = "batch_id"
    case sessionId = "session_id"
    case flushedCount = "flushed_count"
    case pendingFrames = "pending_frames"
    case totalLatencyMs = "total_latency_ms"
    case engineMode = "engine_mode"
  }
}

/// Response after pushing one frame to a stream session.
public struct ServerSideStreamFrameResponse: Decodable, Sendable {
  public var frameId: String
  public var sessionId: String
  public var pendingFrames: Int
  public var autoFlushed: Bool
  public var flush: ServerSideStreamFlushResponse?

  enum CodingKeys: String, CodingKey {
    case flush
    case frameId = "frame_id"
    case sessionId = "session_id"
    case pendingFrames = "pending_frames"
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
    idleTtlMs: Int = 60_000
  ) async throws -> ServerSideStreamSessionInfo {
    if !client.isReady {
      _ = try await client.prepare()
    }
    let info = try await client.createStreamSession(
      batchSize: batchSize,
      maxWaitMs: maxWaitMs,
      idleTtlMs: idleTtlMs
    )
    sessionInfo = info
    isOpen = true
    clientFrameIndex = 0
    return info
  }

  /// Pushes one JPEG frame; returns flush results when the server auto-batches.
  public func pushFrame(_ image: UIImage) async throws -> ServerSideStreamFrameResponse {
    guard isOpen else { throw ServerSideInferenceError.streamNotOpen }
    let index = clientFrameIndex
    clientFrameIndex += 1
    let response = try await client.pushStreamFrame(image: image, clientIndex: index)
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
