import Foundation
import UIKit
import AVFoundation

/// Music-app facing façade over `MLKitModelBridge`.
///
/// **iOS phones default to MobileNet** for fast, accessible on-device vision.
/// Logs Firebase Analytics and keeps the last prediction for customer feedback.
@MainActor
public final class MusicVisionClassifier: ObservableObject {
  @Published public private(set) var isPrepared = false
  @Published public private(set) var lastLabels: [MusicVisionLabel] = []
  @Published public private(set) var lastError: String?
  @Published public private(set) var lastInferenceId: String?
  @Published public private(set) var lastLatencyMs: Double?
  @Published public private(set) var accessibilitySummary: String = ""

  private let bridge: MLKitModelBridge
  private let deploymentTarget: VisionDeploymentTarget
  private let announceResults: Bool
  private let modelSourceProvider: () -> ModelPerformanceAnalytics.ModelSource

  public var architecture: VisionModelArchitecture { bridge.architecture }

  public init(
    deploymentTarget: VisionDeploymentTarget = .iOSPhone,
    announceResults: Bool = true,
    modelSourceProvider: (() -> ModelPerformanceAnalytics.ModelSource)? = nil
  ) {
    let bridge = MLKitModelBridge(
      configuration: .init(deploymentTarget: deploymentTarget)
    )
    self.bridge = bridge
    self.deploymentTarget = deploymentTarget
    self.announceResults = announceResults
    self.modelSourceProvider = modelSourceProvider ?? {
      RemoteModelDownloader(
        destinationFileName: bridge.architecture.downloadedModelFileName
      ).isCached ? .downloaded : .bundled
    }
  }

  public init(
    bridge: MLKitModelBridge,
    deploymentTarget: VisionDeploymentTarget = .iOSPhone,
    announceResults: Bool = true,
    modelSourceProvider: @escaping () -> ModelPerformanceAnalytics.ModelSource = {
      RemoteModelDownloader().isCached ? .downloaded : .bundled
    }
  ) {
    self.bridge = bridge
    self.deploymentTarget = deploymentTarget
    self.announceResults = announceResults
    self.modelSourceProvider = modelSourceProvider
  }

  /// Loads the MobileNet (iOS) `.tflite` model into ML Kit.
  public func prepare() {
    _ = FirebaseBootstrap.configureIfNeeded()
    do {
      _ = try bridge.prepare()
      isPrepared = true
      lastError = nil
      ModelPerformanceAnalytics.logPrepareSuccess(
        source: modelSourceProvider(),
        architecture: architecture,
        deviceClass: deploymentTarget
      )
    } catch {
      isPrepared = false
      lastError = error.localizedDescription
      ModelPerformanceAnalytics.logPrepareFailure(
        message: error.localizedDescription,
        architecture: architecture,
        deviceClass: deploymentTarget
      )
    }
  }

  /// Classifies a still image (async). Announces results for VoiceOver users.
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
      accessibilitySummary = MusicVisionAccessibility.announcement(
        for: labels,
        architecture: architecture
      )
      if announceResults {
        MusicVisionAccessibility.announce(labels, architecture: architecture)
      }
      ModelPerformanceAnalytics.logInference(
        latencyMs: latencyMs,
        labels: labels,
        success: true,
        architecture: architecture,
        deviceClass: deploymentTarget
      )
      return labels
    } catch {
      let latencyMs = Date().timeIntervalSince(started) * 1000
      lastError = error.localizedDescription
      lastInferenceId = inferenceId
      lastLatencyMs = latencyMs
      accessibilitySummary = "Vision failed. \(error.localizedDescription)"
      ModelPerformanceAnalytics.logInference(
        latencyMs: latencyMs,
        labels: [],
        success: false,
        errorMessage: error.localizedDescription,
        architecture: architecture,
        deviceClass: deploymentTarget
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
    let architecture = bridge.architecture
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
          success: true,
          architecture: architecture,
          deviceClass: .iOSPhone
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
          errorMessage: error.localizedDescription,
          architecture: architecture,
          deviceClass: .iOSPhone
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
    ModelPerformanceAnalytics.logFeedbackSubmitted(
      rating: rating,
      topLabel: top?.text,
      hasComment: !comment.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      architecture: architecture
    )
  }
}
