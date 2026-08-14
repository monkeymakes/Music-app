import AVFoundation
import Foundation
import UIKit
import MLKitCommon
import MLKitImageLabelingCommon
import MLKitImageLabelingCustom
import MLKitVision

/// Errors raised while bridging a LiteRT / TFLite model into ML Kit on iOS.
public enum MLKitBridgeError: Error, LocalizedError, Equatable {
  case modelNotFound(resourceName: String)
  case labelerNotReady
  case inferenceFailed(String)

  public var errorDescription: String? {
    switch self {
    case .modelNotFound(let name):
      return "Bundled model '\(name).tflite' was not found in the app bundle."
    case .labelerNotReady:
      return "ML Kit image labeler is not ready. Call prepare() first."
    case .inferenceFailed(let message):
      return "ML Kit inference failed: \(message)"
    }
  }
}

/// A single classification result from the bridged custom model.
public struct MusicVisionLabel: Equatable, Sendable {
  public let text: String
  public let confidence: Float
  public let index: Int

  public init(text: String, confidence: Float, index: Int) {
    self.text = text
    self.confidence = confidence
    self.index = index
  }
}

/// Bridges a custom AutoML / TensorFlow Lite image model into ML Kit's
/// `ImageLabeler` so Music-app can run on-device vision on iOS.
///
/// Drop `music_vision.tflite` (with metadata) into the app target and ensure
/// it is listed under Copy Bundle Resources. Optionally place a downloaded
/// remote copy at the documents path configured in `Configuration`.
public final class MLKitModelBridge: @unchecked Sendable {
  public struct Configuration: Sendable {
    /// Bundled resource name without extension (default: `music_vision`).
    public var bundledModelName: String
    /// Optional on-disk filename for a model downloaded from Cloud Storage.
    public var downloadedModelFileName: String
    public var confidenceThreshold: Float
    public var maxResultCount: Int

    public init(
      bundledModelName: String = "music_vision",
      downloadedModelFileName: String = "music_vision_remote.tflite",
      confidenceThreshold: Float = 0.5,
      maxResultCount: Int = 5
    ) {
      self.bundledModelName = bundledModelName
      self.downloadedModelFileName = downloadedModelFileName
      self.confidenceThreshold = confidenceThreshold
      self.maxResultCount = maxResultCount
    }
  }

  private let configuration: Configuration
  private let lock = NSLock()
  private var imageLabeler: ImageLabeler?

  public init(configuration: Configuration = Configuration()) {
    self.configuration = configuration
  }

  /// Resolves a local model path (downloaded override → bundled) and builds
  /// the ML Kit custom image labeler.
  @discardableResult
  public func prepare() throws -> LocalModel {
    let localModel = try resolveLocalModel()
    let options = CustomImageLabelerOptions(localModel: localModel)
    options.confidenceThreshold = NSNumber(value: configuration.confidenceThreshold)
    options.maxResultCount = configuration.maxResultCount

    let labeler = ImageLabeler.imageLabeler(options: options)
    lock.lock()
    imageLabeler = labeler
    lock.unlock()
    return localModel
  }

  /// Returns whether the labeler has been prepared.
  public var isReady: Bool {
    lock.lock()
    defer { lock.unlock() }
    return imageLabeler != nil
  }

  /// Runs the bridged custom model on a UIImage asynchronously.
  public func classify(
    image: UIImage,
    completion: @escaping (Result<[MusicVisionLabel], Error>) -> Void
  ) {
    let labeler: ImageLabeler
    do {
      labeler = try requireLabeler()
    } catch {
      completion(.failure(error))
      return
    }

    let visionImage = VisionImage(image: image)
    visionImage.orientation = image.imageOrientation

    labeler.process(visionImage) { labels, error in
      if let error {
        completion(.failure(MLKitBridgeError.inferenceFailed(error.localizedDescription)))
        return
      }

      let mapped = (labels ?? []).map {
        MusicVisionLabel(text: $0.text, confidence: $0.confidence, index: $0.index)
      }
      completion(.success(mapped))
    }
  }

  /// Runs the bridged custom model on a UIImage using the synchronous ML Kit API
  /// (preferred for camera / real-time frames).
  public func classifySync(image: UIImage) throws -> [MusicVisionLabel] {
    let labeler = try requireLabeler()
    let visionImage = VisionImage(image: image)
    visionImage.orientation = image.imageOrientation

    do {
      let labels = try labeler.results(in: visionImage)
      return labels.map {
        MusicVisionLabel(text: $0.text, confidence: $0.confidence, index: $0.index)
      }
    } catch {
      throw MLKitBridgeError.inferenceFailed(error.localizedDescription)
    }
  }

  /// Runs the bridged model on a camera sample buffer (real-time path).
  public func classifySync(
    sampleBuffer: CMSampleBuffer,
    cameraPosition: AVCaptureDevice.Position,
    deviceOrientation: UIDeviceOrientation = UIDevice.current.orientation
  ) throws -> [MusicVisionLabel] {
    let labeler = try requireLabeler()
    let visionImage = VisionImage(buffer: sampleBuffer)
    visionImage.orientation = Self.imageOrientation(
      deviceOrientation: deviceOrientation,
      cameraPosition: cameraPosition
    )

    do {
      let labels = try labeler.results(in: visionImage)
      return labels.map {
        MusicVisionLabel(text: $0.text, confidence: $0.confidence, index: $0.index)
      }
    } catch {
      throw MLKitBridgeError.inferenceFailed(error.localizedDescription)
    }
  }

  // MARK: - Model resolution

  private func resolveLocalModel() throws -> LocalModel {
    if let downloaded = downloadedModelPath(),
       FileManager.default.fileExists(atPath: downloaded.path) {
      return LocalModel(path: downloaded.path)
    }

    guard let bundled = Bundle.main.path(
      forResource: configuration.bundledModelName,
      ofType: "tflite"
    ) else {
      throw MLKitBridgeError.modelNotFound(resourceName: configuration.bundledModelName)
    }

    return LocalModel(path: bundled)
  }

  private func downloadedModelPath() -> URL? {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
      .appendingPathComponent(configuration.downloadedModelFileName)
  }

  private func requireLabeler() throws -> ImageLabeler {
    lock.lock()
    defer { lock.unlock() }
    guard let imageLabeler else {
      throw MLKitBridgeError.labelerNotReady
    }
    return imageLabeler
  }

  /// Orientation helper from ML Kit's real-time guidance.
  public static func imageOrientation(
    deviceOrientation: UIDeviceOrientation,
    cameraPosition: AVCaptureDevice.Position
  ) -> UIImage.Orientation {
    switch deviceOrientation {
    case .portrait:
      return cameraPosition == .front ? .leftMirrored : .right
    case .landscapeLeft:
      return cameraPosition == .front ? .downMirrored : .up
    case .portraitUpsideDown:
      return cameraPosition == .front ? .rightMirrored : .left
    case .landscapeRight:
      return cameraPosition == .front ? .upMirrored : .down
    case .faceDown, .faceUp, .unknown:
      return .up
    @unknown default:
      return .up
    }
  }
}
