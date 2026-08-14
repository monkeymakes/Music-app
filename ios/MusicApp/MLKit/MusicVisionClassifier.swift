import Foundation
import UIKit
import AVFoundation

/// Music-app facing façade over `MLKitModelBridge`.
///
/// Use this from SwiftUI / UIKit to classify album art, instruments, or
/// stage scenes with the bridged custom ML Kit model.
@MainActor
public final class MusicVisionClassifier: ObservableObject {
  @Published public private(set) var isPrepared = false
  @Published public private(set) var lastLabels: [MusicVisionLabel] = []
  @Published public private(set) var lastError: String?

  private let bridge: MLKitModelBridge

  public init(bridge: MLKitModelBridge = MLKitModelBridge()) {
    self.bridge = bridge
  }

  /// Loads the bundled (or downloaded) `.tflite` model into ML Kit.
  public func prepare() {
    do {
      _ = try bridge.prepare()
      isPrepared = true
      lastError = nil
    } catch {
      isPrepared = false
      lastError = error.localizedDescription
    }
  }

  /// Classifies a still image (async completion bridged onto the main actor).
  public func classify(image: UIImage) async throws -> [MusicVisionLabel] {
    try await withCheckedThrowingContinuation { continuation in
      bridge.classify(image: image) { result in
        Task { @MainActor in
          switch result {
          case .success(let labels):
            self.lastLabels = labels
            self.lastError = nil
            continuation.resume(returning: labels)
          case .failure(let error):
            self.lastError = error.localizedDescription
            continuation.resume(throwing: error)
          }
        }
      }
    }
  }

  /// Classifies a live camera frame synchronously (call off the main thread).
  nonisolated public func classifyFrame(
    sampleBuffer: CMSampleBuffer,
    cameraPosition: AVCaptureDevice.Position
  ) throws -> [MusicVisionLabel] {
    try bridge.classifySync(
      sampleBuffer: sampleBuffer,
      cameraPosition: cameraPosition
    )
  }
}
