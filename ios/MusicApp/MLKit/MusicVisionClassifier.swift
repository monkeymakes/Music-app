import Foundation
import UIKit
import AVFoundation

/// Music-app facing façade over ML Kit, Apple Core ML, or **server-side** inference.
///
/// **iOS phones default to TinyML** on-device. Use `.serverSide` to call the
/// versioned registry API (`/v1/infer`) so model pins update without an App Store release.
/// Logs Firebase Analytics and keeps the last prediction for customer feedback.
@MainActor
public final class MusicVisionClassifier: ObservableObject {
  @Published public private(set) var isPrepared = false
  @Published public private(set) var lastLabels: [MusicVisionLabel] = []
  @Published public private(set) var lastError: String?
  @Published public private(set) var lastInferenceId: String?
  @Published public private(set) var lastLatencyMs: Double?
  @Published public private(set) var lastModelVersion: String?
  @Published public private(set) var accessibilitySummary: String = ""

  public let runtime: VisionInferenceRuntime
  private let engine: MusicVisionInferencing?
  private let mlKitBridge: MLKitModelBridge?
  private let coreMLBridge: CoreMLModelBridge?
  private let serverClient: ServerSideInferenceClient?
  private let deploymentTarget: VisionDeploymentTarget
  private let announceResults: Bool
  private let modelSourceProvider: () -> ModelPerformanceAnalytics.ModelSource

  public var architecture: VisionModelArchitecture {
    if let engine { return engine.architecture }
    if let serverClient, let arch = serverClient.activeVersion?.architecture,
       let parsed = VisionModelArchitecture(rawValue: arch) {
      return parsed
    }
    return deploymentTarget.architecture
  }

  public init(
    runtime: VisionInferenceRuntime = .coreML,
    deploymentTarget: VisionDeploymentTarget = .iOSPhone,
    announceResults: Bool = true,
    serverConfiguration: ServerSideInferenceClient.Configuration? = nil,
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
      self.serverClient = nil
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
      self.serverClient = nil
      self.engine = bridge
      self.modelSourceProvider = modelSourceProvider ?? {
        RemoteModelDownloader(
          destinationFileName: bridge.architecture.coreMLDownloadedFileName
        ).isCached ? .downloaded : .bundled
      }
    case .serverSide:
      let config = serverConfiguration ?? ServerSideInferenceClient.Configuration(
        baseURL: URL(string: "http://127.0.0.1:8080")!,
        modelName: deploymentTarget == .raspberryPi
          ? "music_vision_efficientnet"
          : "music_vision_tinyml",
        channel: "production"
      )
      self.serverClient = ServerSideInferenceClient(configuration: config)
      self.mlKitBridge = nil
      self.coreMLBridge = nil
      self.engine = nil
      self.modelSourceProvider = modelSourceProvider ?? { .serverSide }
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
    self.serverClient = nil
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
    self.serverClient = nil
    self.engine = bridge
    self.deploymentTarget = deploymentTarget
    self.announceResults = announceResults
    self.modelSourceProvider = modelSourceProvider
  }

  public init(
    serverClient: ServerSideInferenceClient,
    deploymentTarget: VisionDeploymentTarget = .iOSPhone,
    announceResults: Bool = true
  ) {
    self.runtime = .serverSide
    self.serverClient = serverClient
    self.mlKitBridge = nil
    self.coreMLBridge = nil
    self.engine = nil
    self.deploymentTarget = deploymentTarget
    self.announceResults = announceResults
    self.modelSourceProvider = { .serverSide }
  }

  /// Loads on-device model, or resolves the active server-side registry pin.
  public func prepare() {
    _ = FirebaseBootstrap.configureIfNeeded()
    if runtime == .serverSide {
      Task { await prepareServerSide() }
      return
    }
    do {
      guard let engine else { throw ServerSideInferenceError.notPrepared }
      _ = try engine.prepare()
      isPrepared = true
      lastError = nil
      lastModelVersion = nil
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

  /// Async prepare for server-side (resolves channel → version from registry).
  public func prepareServerSide() async {
    guard let serverClient else {
      isPrepared = false
      lastError = ServerSideInferenceError.notPrepared.localizedDescription
      return
    }
    do {
      let info = try await serverClient.prepare()
      isPrepared = true
      lastError = nil
      lastModelVersion = info.version
      ModelPerformanceAnalytics.logPrepareSuccess(
        source: .serverSide,
        architecture: architecture,
        deviceClass: deploymentTarget,
        runtime: .serverSide
      )
    } catch {
      isPrepared = false
      lastError = error.localizedDescription
      ModelPerformanceAnalytics.logPrepareFailure(
        message: error.localizedDescription,
        architecture: architecture,
        deviceClass: deploymentTarget,
        runtime: .serverSide
      )
    }
  }

  /// Classifies a still image (async). Announces results for VoiceOver users.
  public func classify(image: UIImage) async throws -> [MusicVisionLabel] {
    let started = Date()
    let inferenceId = UUID().uuidString

    do {
      let labels: [MusicVisionLabel]
      if let serverClient {
        if !serverClient.isReady {
          _ = try await serverClient.prepare()
          lastModelVersion = serverClient.activeVersion?.version
        }
        let response = try await serverClient.classify(image: image)
        lastModelVersion = response.version
        labels = response.labels.map {
          MusicVisionLabel(text: $0.text, confidence: $0.confidence, index: $0.index)
        }
      } else if let engine {
        labels = try await withCheckedThrowingContinuation { continuation in
          engine.classify(image: image) { result in
            continuation.resume(with: result)
          }
        }
      } else {
        throw ServerSideInferenceError.notPrepared
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

  /// Classifies a live camera frame synchronously (on-device only).
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
    var modelName = ModelPerformanceAnalytics.modelName
    if let version = lastModelVersion {
      modelName = "\(modelName)@\(version)"
    }
    let feedback = ModelFeedback(
      inferenceId: lastInferenceId ?? UUID().uuidString,
      modelName: modelName,
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
