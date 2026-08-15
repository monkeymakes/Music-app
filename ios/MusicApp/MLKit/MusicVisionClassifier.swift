import Foundation
import UIKit
import AVFoundation

/// Music-app facing façade over ML Kit **or** Apple Core ML.
///
/// **iOS phones default to MobileNet** for fast, accessible on-device vision.
/// Choose `.mlKit` (TFLite) or `.coreML` (Create ML / Neural Engine).
/// Logs Firebase Analytics and keeps the last prediction for customer feedback.
@MainActor
public final class MusicVisionClassifier: ObservableObject {
  @Published public private(set) var isPrepared = false
  @Published public private(set) var lastLabels: [MusicVisionLabel] = []
  @Published public private(set) var lastError: String?
  @Published public private(set) var lastInferenceId: String?
  @Published public private(set) var lastLatencyMs: Double?
  @Published public private(set) var accessibilitySummary: String = ""

  public let runtime: VisionInferenceRuntime
  private let engine: MusicVisionInferencing
  private let mlKitBridge: MLKitModelBridge?
  private let coreMLBridge: CoreMLModelBridge?
  private let deploymentTarget: VisionDeploymentTarget
  private let announceResults: Bool
  private let modelSourceProvider: () -> ModelPerformanceAnalytics.ModelSource

  public var architecture: VisionModelArchitecture { engine.architecture }

  public init(
    runtime: VisionInferenceRuntime = .coreML,
    deploymentTarget: VisionDeploymentTarget = .iOSPhone,
    announceResults: Bool = true,
    modelSourceProvider: (() -> ModelPerformanceAnalytics.ModelSource)? = nil
  ) {
    self.runtime = runtime
    self.deploymentTarget = deploymentTarget
    self.announceResults = announceResults

    switch runtime {
    case .mlKit:
      let bridge = MLKitModelBridge(
        configuration: .init(deploymentTarget: deploymentTarget)
      )
      self.mlKitBridge = bridge
      self.coreMLBridge = nil
      self.engine = bridge
      self.modelSourceProvider = modelSourceProvider ?? {
        RemoteModelDownloader(
          destinationFileName: bridge.architecture.downloadedModelFileName
        ).isCached ? .downloaded : .bundled
      }
    case .coreML:
      let bridge = CoreMLModelBridge(
        configuration: .init(deploymentTarget: deploymentTarget)
      )
      self.coreMLBridge = bridge
      self.mlKitBridge = nil
      self.engine = bridge
      self.modelSourceProvider = modelSourceProvider ?? {
        RemoteModelDownloader(
          destinationFileName: bridge.architecture.coreMLDownloadedFileName
        ).isCached ? .downloaded : .bundled
      }
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
    self.runtime = .mlKit
    self.mlKitBridge = bridge
    self.coreMLBridge = nil
    self.engine = bridge
    self.deploymentTarget = deploymentTarget
    self.announceResults = announceResults
    self.modelSourceProvider = modelSourceProvider
  }

  public init(
    bridge: CoreMLModelBridge,
    deploymentTarget: VisionDeploymentTarget = .iOSPhone,
    announceResults: Bool = true,
    modelSourceProvider: @escaping () -> ModelPerformanceAnalytics.ModelSource = { .bundled }
  ) {
    self.runtime = .coreML
    self.coreMLBridge = bridge
    self.mlKitBridge = nil
    self.engine = bridge
    self.deploymentTarget = deploymentTarget
    self.announceResults = announceResults
    self.modelSourceProvider = modelSourceProvider
  }

  /// Loads the MobileNet model into ML Kit (`.tflite`) or Core ML (`.mlpackage`).
  public func prepare() {
    _ = FirebaseBootstrap.configureIfNeeded()
    do {
      _ = try engine.prepare()
      isPrepared = true
      lastError = nil
      ModelPerformanceAnalytics.logPrepareSuccess(
        source: modelSourceProvider(),
        architecture: architecture,
        deviceClass: deploymentTarget,
        runtime: runtime
      )
    } catch {
      isPrepared = false
      lastError = error.localizedDescription
      ModelPerformanceAnalytics.logPrepareFailure(
        message: error.localizedDescription,
        architecture: architecture,
        deviceClass: deploymentTarget,
        runtime: runtime
      )
    }
  }

  /// Classifies a still image (async). Announces results for VoiceOver users.
  public func classify(image: UIImage) async throws -> [MusicVisionLabel] {
    let started = Date()
    let inferenceId = UUID().uuidString

    do {
      let labels: [MusicVisionLabel] = try await withCheckedThrowingContinuation { continuation in
        engine.classify(image: image) { result in
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
        deviceClass: deploymentTarget,
        runtime: runtime
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
        deviceClass: deploymentTarget,
        runtime: runtime
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
    let activeRuntime: VisionInferenceRuntime = mlKitBridge != nil ? .mlKit : .coreML
    let activeArchitecture =
      mlKitBridge?.architecture ?? coreMLBridge?.architecture ?? .mobileNet
    do {
      let labels: [MusicVisionLabel]
      if let mlKitBridge {
        labels = try mlKitBridge.classifySync(
          sampleBuffer: sampleBuffer,
          cameraPosition: cameraPosition
        )
      } else if let coreMLBridge {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
          throw CoreMLBridgeError.unsupportedImage
        }
        let orientation: CGImagePropertyOrientation =
          cameraPosition == .front ? .leftMirrored : .right
        labels = try coreMLBridge.classifySync(
          pixelBuffer: pixelBuffer,
          orientation: orientation
        )
      } else {
        throw CoreMLBridgeError.modelNotReady
      }

      if sampleAnalytics {
        let latencyMs = Date().timeIntervalSince(started) * 1000
        ModelPerformanceAnalytics.logInference(
          latencyMs: latencyMs,
          labels: labels,
          success: true,
          architecture: activeArchitecture,
          deviceClass: .iOSPhone,
          runtime: activeRuntime
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
          architecture: activeArchitecture,
          deviceClass: .iOSPhone,
          runtime: activeRuntime
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
      architecture: architecture,
      runtime: runtime
    )
  }
}
