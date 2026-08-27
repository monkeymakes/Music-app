import Foundation
import FirebaseAnalytics

/// Logs Music Vision model lifecycle and inference metrics to Firebase Analytics.
///
/// Register these custom definitions in the Firebase console
/// (Analytics → Custom definitions) so they appear in reports:
/// - Dimensions: `model_name`, `model_source`, `architecture`, `device_class`,
///   `runtime`, `top_label`, `success`, `batch_count`, `engine_mode`
/// - Metrics: `latency_ms`, `label_count`, `top_confidence`, `value`, `total_latency_ms`
public enum ModelPerformanceAnalytics {
  public static let modelName = "music_vision"
  private static let inferenceBatchQueue = DispatchQueue(label: "music_vision.analytics.batch")
  private static var pendingInferenceEvents: [[String: Any]] = []
  private static let defaultInferenceBatchSize = 10

  public enum ModelSource: String {
    case bundled
    case downloaded
    case serverSide = "server_side"
  }

  /// Fired when ML Kit or Core ML successfully prepares a model.
  public static func logPrepareSuccess(
    source: ModelSource,
    architecture: VisionModelArchitecture = .tinyML,
    deviceClass: VisionDeploymentTarget = .iOSPhone,
    runtime: VisionInferenceRuntime = .coreML
  ) {
    guard FirebaseBootstrap.configureIfNeeded() else { return }
    Analytics.logEvent("music_vision_prepare", parameters: [
      "model_name": modelName,
      "model_source": source.rawValue,
      "architecture": architecture.rawValue,
      "device_class": deviceClass.rawValue,
      "runtime": runtime.rawValue,
      "success": "true",
    ])
  }

  /// Fired when prepare fails (missing `.tflite` / `.mlpackage`, etc.).
  public static func logPrepareFailure(
    message: String,
    architecture: VisionModelArchitecture = .tinyML,
    deviceClass: VisionDeploymentTarget = .iOSPhone,
    runtime: VisionInferenceRuntime = .coreML
  ) {
    guard FirebaseBootstrap.configureIfNeeded() else { return }
    Analytics.logEvent("music_vision_prepare", parameters: [
      "model_name": modelName,
      "architecture": architecture.rawValue,
      "device_class": deviceClass.rawValue,
      "runtime": runtime.rawValue,
      "success": "false",
      "error_message": String(message.prefix(100)),
    ])
  }

  /// Fired after each inference with latency and top prediction quality signals.
  /// High-frequency paths can use `enqueueInference` to batch Firebase writes.
  public static func logInference(
    latencyMs: Double,
    labels: [MusicVisionLabel],
    success: Bool,
    errorMessage: String? = nil,
    architecture: VisionModelArchitecture = .tinyML,
    deviceClass: VisionDeploymentTarget = .iOSPhone,
    runtime: VisionInferenceRuntime = .coreML
  ) {
    enqueueInference(
      latencyMs: latencyMs,
      labels: labels,
      success: success,
      errorMessage: errorMessage,
      architecture: architecture,
      deviceClass: deviceClass,
      runtime: runtime,
      flushImmediately: true
    )
  }

  /// Queues inference analytics and flushes when the batch buffer fills.
  public static func enqueueInference(
    latencyMs: Double,
    labels: [MusicVisionLabel],
    success: Bool,
    errorMessage: String? = nil,
    architecture: VisionModelArchitecture = .tinyML,
    deviceClass: VisionDeploymentTarget = .iOSPhone,
    runtime: VisionInferenceRuntime = .coreML,
    flushImmediately: Bool = false
  ) {
    guard FirebaseBootstrap.configureIfNeeded() else { return }

    var parameters: [String: Any] = [
      "model_name": modelName,
      "architecture": architecture.rawValue,
      "device_class": deviceClass.rawValue,
      "runtime": runtime.rawValue,
      "latency_ms": Int(latencyMs.rounded()),
      "label_count": labels.count,
      "success": success ? "true" : "false",
      AnalyticsParameterValue: latencyMs,
    ]

    if let top = labels.max(by: { $0.confidence < $1.confidence }) {
      parameters["top_label"] = String(top.text.prefix(100))
      parameters["top_confidence"] = Double(top.confidence)
    }

    if let errorMessage {
      parameters["error_message"] = String(errorMessage.prefix(100))
    }

    inferenceBatchQueue.sync {
      pendingInferenceEvents.append(parameters)
      if flushImmediately || pendingInferenceEvents.count >= defaultInferenceBatchSize {
        flushPendingInferenceEventsLocked()
      }
    }
  }

  /// Sends any queued inference analytics events to Firebase.
  public static func flushInferenceBatch() {
    guard FirebaseBootstrap.configureIfNeeded() else { return }
    inferenceBatchQueue.sync {
      flushPendingInferenceEventsLocked()
    }
  }

  /// Fired after a server-side batch inference completes.
  public static func logBatchInference(
    count: Int,
    totalLatencyMs: Double,
    engineMode: String
  ) {
    guard FirebaseBootstrap.configureIfNeeded() else { return }
    Analytics.logEvent("music_vision_batch_inference", parameters: [
      "model_name": modelName,
      "batch_count": count,
      "total_latency_ms": Int(totalLatencyMs.rounded()),
      "engine_mode": engineMode,
      AnalyticsParameterValue: totalLatencyMs,
    ])
  }

  /// Fired when a stream session auto-flushes or is manually flushed.
  public static func logStreamBatch(
    flushedCount: Int,
    totalLatencyMs: Double,
    engineMode: String
  ) {
    guard FirebaseBootstrap.configureIfNeeded() else { return }
    Analytics.logEvent("music_vision_stream_batch", parameters: [
      "model_name": modelName,
      "batch_count": flushedCount,
      "total_latency_ms": Int(totalLatencyMs.rounded()),
      "engine_mode": engineMode,
      AnalyticsParameterValue: totalLatencyMs,
    ])
  }

  private static func flushPendingInferenceEventsLocked() {
    guard !pendingInferenceEvents.isEmpty else { return }
    let events = pendingInferenceEvents
    pendingInferenceEvents = []
    for parameters in events {
      Analytics.logEvent("music_vision_inference", parameters: parameters)
    }
  }

  /// Fired when a user submits feedback so Analytics can correlate with inference quality.
  public static func logFeedbackSubmitted(
    rating: ModelFeedbackRating,
    topLabel: String?,
    hasComment: Bool,
    architecture: VisionModelArchitecture = .tinyML,
    runtime: VisionInferenceRuntime = .coreML
  ) {
    guard FirebaseBootstrap.configureIfNeeded() else { return }
    var parameters: [String: Any] = [
      "model_name": modelName,
      "architecture": architecture.rawValue,
      "device_class": VisionDeploymentTarget.iOSPhone.rawValue,
      "runtime": runtime.rawValue,
      "rating": rating.rawValue,
      "has_comment": hasComment ? "true" : "false",
    ]
    if let topLabel {
      parameters["top_label"] = String(topLabel.prefix(100))
    }
    Analytics.logEvent("music_vision_feedback", parameters: parameters)
  }
}
