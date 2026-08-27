import CoreML
import Foundation
import UIKit
import Vision

/// Errors raised while loading or running a Create ML / Core ML model on iOS.
public enum CoreMLBridgeError: Error, LocalizedError, Equatable {
  case modelNotFound(resourceName: String)
  case modelNotReady
  case inferenceFailed(String)
  case unsupportedImage

  public var errorDescription: String? {
    switch self {
    case .modelNotFound(let name):
      return "Bundled Core ML model '\(name)' (.mlmodel / .mlpackage / .mlmodelc) was not found."
    case .modelNotReady:
      return "Core ML model is not ready. Call prepare() first."
    case .inferenceFailed(let message):
      return "Core ML inference failed: \(message)"
    case .unsupportedImage:
      return "Could not create a CGImage / pixel buffer from the input image."
    }
  }
}

/// Bridges a Create ML–exported (or coremltools-converted) MobileNet into
/// Apple Vision + Core ML for on-device Music Vision on iOS.
///
/// Prefer bundling `music_vision_mobilenet.mlpackage` (or compiled `.mlmodelc`)
/// in the MusicApp target. Xcode compiles `.mlmodel` / `.mlpackage` at build time.
public final class CoreMLModelBridge: MusicVisionInferencing, @unchecked Sendable {
  public struct Configuration: Sendable {
    /// Bundled resource name without extension (MobileNet for iOS phones).
    public var bundledModelName: String
    /// Optional Documents override downloaded from Cloud Storage.
    public var downloadedModelFileName: String
    public var architecture: VisionModelArchitecture
    public var confidenceThreshold: Float
    public var maxResultCount: Int

    public init(
      deploymentTarget: VisionDeploymentTarget = .iOSPhone,
      confidenceThreshold: Float = 0.5,
      maxResultCount: Int = 5
    ) {
      let architecture = deploymentTarget.architecture
      self.architecture = architecture
      self.bundledModelName = architecture.bundledModelName
      self.downloadedModelFileName = architecture.coreMLDownloadedFileName
      self.confidenceThreshold = confidenceThreshold
      self.maxResultCount = maxResultCount
    }

    public init(
      bundledModelName: String,
      downloadedModelFileName: String,
      architecture: VisionModelArchitecture = .tinyML,
      confidenceThreshold: Float = 0.5,
      maxResultCount: Int = 5
    ) {
      self.bundledModelName = bundledModelName
      self.downloadedModelFileName = downloadedModelFileName
      self.architecture = architecture
      self.confidenceThreshold = confidenceThreshold
      self.maxResultCount = maxResultCount
    }
  }

  private let configuration: Configuration
  private let lock = NSLock()
  private var visionModel: VNCoreMLModel?
  private var loadedModelDescription: String = ""

  public var architecture: VisionModelArchitecture { configuration.architecture }

  public init(configuration: Configuration = Configuration(deploymentTarget: .iOSPhone)) {
    self.configuration = configuration
  }

  /// Resolves a local Core ML model (downloaded override → bundled) and wraps it for Vision.
  @discardableResult
  public func prepare() throws -> String {
    let url = try resolveModelURL()
    let mlModel: MLModel
    do {
      let config = MLModelConfiguration()
      config.computeUnits = .all // CPU + GPU + Neural Engine when available
      mlModel = try MLModel(contentsOf: url, configuration: config)
    } catch {
      throw CoreMLBridgeError.inferenceFailed(error.localizedDescription)
    }

    let vnModel: VNCoreMLModel
    do {
      vnModel = try VNCoreMLModel(for: mlModel)
    } catch {
      throw CoreMLBridgeError.inferenceFailed(error.localizedDescription)
    }

    lock.lock()
    visionModel = vnModel
    loadedModelDescription = url.lastPathComponent
    lock.unlock()
    return loadedModelDescription
  }

  public var isReady: Bool {
    lock.lock()
    defer { lock.unlock() }
    return visionModel != nil
  }

  public func classify(
    image: UIImage,
    completion: @escaping (Result<[MusicVisionLabel], Error>) -> Void
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        let labels = try self.classifySync(image: image)
        completion(.success(labels))
      } catch {
        completion(.failure(error))
      }
    }
  }

  public func classifySync(image: UIImage) throws -> [MusicVisionLabel] {
    let model = try requireVisionModel()
    guard let cgImage = image.cgImage else {
      throw CoreMLBridgeError.unsupportedImage
    }

    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .centerCrop

    let handler = VNImageRequestHandler(
      cgImage: cgImage,
      orientation: Self.cgImageOrientation(from: image.imageOrientation),
      options: [:]
    )

    do {
      try handler.perform([request])
    } catch {
      throw CoreMLBridgeError.inferenceFailed(error.localizedDescription)
    }

    return mapObservations(request.results)
  }

  /// Runs Core ML on a camera pixel buffer (real-time path).
  public func classifySync(
    pixelBuffer: CVPixelBuffer,
    orientation: CGImagePropertyOrientation = .right
  ) throws -> [MusicVisionLabel] {
    let model = try requireVisionModel()
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .centerCrop

    let handler = VNImageRequestHandler(
      cvPixelBuffer: pixelBuffer,
      orientation: orientation,
      options: [:]
    )

    do {
      try handler.perform([request])
    } catch {
      throw CoreMLBridgeError.inferenceFailed(error.localizedDescription)
    }

    return mapObservations(request.results)
  }

  // MARK: - Model resolution

  private func resolveModelURL() throws -> URL {
    if let downloaded = downloadedModelURL(),
       FileManager.default.fileExists(atPath: downloaded.path) {
      return downloaded
    }

    let name = configuration.bundledModelName
    let extensions = ["mlmodelc", "mlpackage", "mlmodel"]
    for ext in extensions {
      if let url = Bundle.main.url(forResource: name, withExtension: ext) {
        return url
      }
    }

    throw CoreMLBridgeError.modelNotFound(resourceName: name)
  }

  private func downloadedModelURL() -> URL? {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
      .appendingPathComponent(configuration.downloadedModelFileName)
  }

  private func requireVisionModel() throws -> VNCoreMLModel {
    lock.lock()
    defer { lock.unlock() }
    guard let visionModel else {
      throw CoreMLBridgeError.modelNotReady
    }
    return visionModel
  }

  private func mapObservations(_ results: [Any]?) -> [MusicVisionLabel] {
    let threshold = configuration.confidenceThreshold
    let maxCount = configuration.maxResultCount

    let classifications = (results as? [VNClassificationObservation]) ?? []
    return classifications
      .filter { $0.confidence >= threshold }
      .prefix(maxCount)
      .enumerated()
      .map { index, observation in
        MusicVisionLabel(
          text: observation.identifier,
          confidence: observation.confidence,
          index: index
        )
      }
  }

  public static func cgImageOrientation(
    from orientation: UIImage.Orientation
  ) -> CGImagePropertyOrientation {
    switch orientation {
    case .up: return .up
    case .down: return .down
    case .left: return .left
    case .right: return .right
    case .upMirrored: return .upMirrored
    case .downMirrored: return .downMirrored
    case .leftMirrored: return .leftMirrored
    case .rightMirrored: return .rightMirrored
    @unknown default: return .up
    }
  }
}
