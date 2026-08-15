import Foundation
import FirebaseAnalytics

/// Logs Music Vision model lifecycle and inference metrics to Firebase Analytics.
///
/// Register these custom definitions in the Firebase console
/// (Analytics → Custom definitions) so they appear in reports:
/// - Dimensions: `model_name`, `model_source`, `top_label`, `success`
/// - Metrics: `latency_ms`, `label_count`, `top_confidence`, `value`
public enum ModelPerformanceAnalytics {
  public static let modelName = "music_vision"

  public enum ModelSource: String {
    case bundled
    case downloaded
  }

  /// Fired when the ML Kit bridge successfully prepares a model.
  public static func logPrepareSuccess(source: ModelSource) {
    guard FirebaseBootstrap.configureIfNeeded() else { return }
    Analytics.logEvent("music_vision_prepare", parameters: [
      "model_name": modelName,
      "model_source": source.rawValue,
      "success": "true",
    ])
  }

  /// Fired when prepare fails (missing `.tflite`, etc.).
  public static func logPrepareFailure(message: String) {
    guard FirebaseBootstrap.configureIfNeeded() else { return }
    Analytics.logEvent("music_vision_prepare", parameters: [
      "model_name": modelName,
      "success": "false",
      "error_message": String(message.prefix(100)),
    ])
  }

  /// Fired after each inference with latency and top prediction quality signals.
  public static func logInference(
    latencyMs: Double,
    labels: [MusicVisionLabel],
    success: Bool,
    errorMessage: String? = nil
  ) {
    guard FirebaseBootstrap.configureIfNeeded() else { return }

    var parameters: [String: Any] = [
      "model_name": modelName,
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

    Analytics.logEvent("music_vision_inference", parameters: parameters)
  }

  /// Fired when a user submits feedback so Analytics can correlate with inference quality.
  public static func logFeedbackSubmitted(
    rating: ModelFeedbackRating,
    topLabel: String?,
    hasComment: Bool
  ) {
    guard FirebaseBootstrap.configureIfNeeded() else { return }
    var parameters: [String: Any] = [
      "model_name": modelName,
      "rating": rating.rawValue,
      "has_comment": hasComment ? "true" : "false",
    ]
    if let topLabel {
      parameters["top_label"] = String(topLabel.prefix(100))
    }
    Analytics.logEvent("music_vision_feedback", parameters: parameters)
  }
}
