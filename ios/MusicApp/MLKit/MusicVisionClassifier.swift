import Foundation
import UIKit
import AVFoundation

/// Music-app facing façade over `MLKitModelBridge`.
///
/// Logs Firebase Analytics for prepare/inference and keeps the last prediction
/// so the UI can submit customer feedback via `ModelFeedbackService`.
@MainActor
public final class MusicVisionClassifier: ObservableObject {
  @Published public private(set) var isPrepared = false
  @Published public private(set) var lastLabels: [MusicVisionLabel] = []
  @Published public private(set) var lastError: String?
  @Published public private(set) var lastInferenceId: String?
  @Published public private(set) var lastLatencyMs: Double?

  private let bridge: MLKitModelBridge
  private let modelSourceProvider: () -> ModelPerformanceAnalytics.ModelSource

  public init(
    bridge: MLKitModelBridge = MLKitModelBridge(),
    modelSourceProvider: @escaping () -> ModelPerformanceAnalytics.ModelSource = {
      RemoteModelDownloader().isCached ? .downloaded : .bundled
    }
  ) {
    self.bridge = bridge
    self.modelSourceProvider = modelSourceProvider
  }

  /// Loads the bundled (or downloaded) `.tflite` model into ML Kit.
  public func prepare() {
    _ = FirebaseBootstrap.configureIfNeeded()
    do {
      _ = try bridge.prepare()
      isPrepared = true
      lastError = nil
      ModelPerformanceAnalytics.logPrepareSuccess(source: modelSourceProvider())
    } catch {
      isPrepared = false
      lastError = error.localizedDescription
      ModelPerformanceAnalytics.logPrepareFailure(message: error.localizedDescription)
    }
  }

  /// Classifies a still image (async completion bridged onto the main actor).
  public func classify(image: UIImage) async throws -> [MusicVisionLabel] {
    let started = Date()
    let inferenceId = UUID().uuidString

    do {
      let labels: [MusicVisionLabel] = try await withCheckedThrowingContinuation { continuation in
        bridge.classify(image: image) { result in
          continuation.resume(with: result)
        }
      }

      let latencyMs = Date().timeIntervalSince(started) * 1000
      lastLabels = labels
      lastError = nil
      lastInferenceId = inferenceId
      lastLatencyMs = latencyMs
      ModelPerformanceAnalytics.logInference(
        latencyMs: latencyMs,
        labels: labels,
        success: true
      )
      return labels
    } catch {
      let latencyMs = Date().timeIntervalSince(started) * 1000
      lastError = error.localizedDescription
      lastInferenceId = inferenceId
      lastLatencyMs = latencyMs
      ModelPerformanceAnalytics.logInference(
        latencyMs: latencyMs,
        labels: [],
        success: false,
        errorMessage: error.localizedDescription
      )
      throw error
    }
  }

  /// Classifies a live camera frame synchronously (call off the main thread).
  /// Analytics for high-frequency frames are sampled (~1/30) to stay within quotas.
  nonisolated public func classifyFrame(
    sampleBuffer: CMSampleBuffer,
    cameraPosition: AVCaptureDevice.Position,
    sampleAnalytics: Bool = Int.random(in: 0..<30) == 0
  ) throws -> [MusicVisionLabel] {
    let started = Date()
    do {
      let labels = try bridge.classifySync(
        sampleBuffer: sampleBuffer,
        cameraPosition: cameraPosition
      )
      if sampleAnalytics {
        let latencyMs = Date().timeIntervalSince(started) * 1000
        ModelPerformanceAnalytics.logInference(
          latencyMs: latencyMs,
          labels: labels,
          success: true
        )
      }
      return labels
    } catch {
      if sampleAnalytics {
        let latencyMs = Date().timeIntervalSince(started) * 1000
        ModelPerformanceAnalytics.logInference(
          latencyMs: latencyMs,
          labels: [],
          success: false,
          errorMessage: error.localizedDescription
        )
      }
      throw error
    }
  }

  /// Builds feedback for the last still-image inference and writes it to Firestore.
  public func submitFeedback(
    rating: ModelFeedbackRating,
    comment: String = ""
  ) async throws {
    let top = lastLabels.max(by: { $0.confidence < $1.confidence })
    let feedback = ModelFeedback(
      inferenceId: lastInferenceId ?? UUID().uuidString,
      rating: rating,
      comment: comment,
      topLabel: top?.text,
      topConfidence: top?.confidence,
      labels: lastLabels,
      latencyMs: lastLatencyMs
    )
    try await ModelFeedbackService.shared.submit(feedback)
  }
}
