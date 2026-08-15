import Foundation
import FirebaseFirestore

/// User rating for a model prediction (customer feedback).
public enum ModelFeedbackRating: String, Sendable, Codable {
  case helpful
  case notHelpful = "not_helpful"
}

/// Structured customer feedback about a Music Vision prediction.
public struct ModelFeedback: Sendable, Equatable {
  public var inferenceId: String
  public var modelName: String
  public var rating: ModelFeedbackRating
  public var comment: String
  public var topLabel: String?
  public var topConfidence: Float?
  public var labels: [MusicVisionLabel]
  public var latencyMs: Double?

  public init(
    inferenceId: String = UUID().uuidString,
    modelName: String = ModelPerformanceAnalytics.modelName,
    rating: ModelFeedbackRating,
    comment: String = "",
    topLabel: String? = nil,
    topConfidence: Float? = nil,
    labels: [MusicVisionLabel] = [],
    latencyMs: Double? = nil
  ) {
    self.inferenceId = inferenceId
    self.modelName = modelName
    self.rating = rating
    self.comment = comment
    self.topLabel = topLabel
    self.topConfidence = topConfidence
    self.labels = labels
    self.latencyMs = latencyMs
  }
}

/// Persists customer feedback to Cloud Firestore for offline review of model quality.
///
/// Collection: `model_feedback`
/// Enable Firestore in the Firebase console and tighten security rules before production.
public final class ModelFeedbackService: Sendable {
  public static let shared = ModelFeedbackService()
  public let collectionName: String

  public init(collectionName: String = "model_feedback") {
    self.collectionName = collectionName
  }

  /// Writes feedback to Firestore and mirrors a summary event to Analytics.
  public func submit(_ feedback: ModelFeedback) async throws {
    guard FirebaseBootstrap.configureIfNeeded() else {
      throw ModelFeedbackError.firebaseNotConfigured
    }

    var payload: [String: Any] = [
      "inference_id": feedback.inferenceId,
      "model_name": feedback.modelName,
      "rating": feedback.rating.rawValue,
      "comment": String(feedback.comment.prefix(1000)),
      "labels": feedback.labels.map { label -> [String: Any] in
        [
          "text": label.text,
          "confidence": Double(label.confidence),
          "index": label.index,
        ]
      },
      "platform": "ios",
      "created_at": FieldValue.serverTimestamp(),
    ]

    if let topLabel = feedback.topLabel {
      payload["top_label"] = topLabel
    }
    if let topConfidence = feedback.topConfidence {
      payload["top_confidence"] = Double(topConfidence)
    }
    if let latencyMs = feedback.latencyMs {
      payload["latency_ms"] = latencyMs
    }

    let db = Firestore.firestore()
    try await db.collection(collectionName).document(feedback.inferenceId).setData(payload)

    ModelPerformanceAnalytics.logFeedbackSubmitted(
      rating: feedback.rating,
      topLabel: feedback.topLabel,
      hasComment: !feedback.comment.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    )
  }
}

public enum ModelFeedbackError: Error, LocalizedError {
  case firebaseNotConfigured

  public var errorDescription: String? {
    switch self {
    case .firebaseNotConfigured:
      return "Add GoogleService-Info.plist and call FirebaseBootstrap.configureIfNeeded()."
    }
  }
}
