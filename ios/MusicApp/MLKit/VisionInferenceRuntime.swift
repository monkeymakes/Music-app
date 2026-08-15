import Foundation
import UIKit

/// Inference stack for Music Vision on iOS.
///
/// - **mlKit** — Google ML Kit custom TFLite ImageLabeler (`.tflite`)
/// - **coreML** — Apple Core ML + Vision (`VNCoreMLRequest`), typically from
///   Create ML or `coremltools` conversion (`.mlmodel` / `.mlpackage`)
/// - **serverSide** — HTTP `/v1/infer` against the versioned model registry
public enum VisionInferenceRuntime: String, Sendable, CaseIterable, Codable {
  case mlKit = "mlkit"
  case coreML = "coreml"
  case serverSide = "server_side"

  public var displayName: String {
    switch self {
    case .mlKit: return "ML Kit"
    case .coreML: return "Core ML"
    case .serverSide: return "Server"
    }
  }

  public var rationale: String {
    switch self {
    case .mlKit:
      return "Google ML Kit runs the MobileNet `.tflite` with ImageLabelingCustom."
    case .coreML:
      return "Apple Core ML uses the Neural Engine via Vision (`VNCoreMLRequest`) for a Create ML–exported MobileNet."
    case .serverSide:
      return "Server-side inference calls the versioned registry API so production/staging pins update without a new app build."
    }
  }
}

/// Shared prepare / classify surface so ML Kit and Core ML stay interchangeable.
public protocol MusicVisionInferencing: AnyObject {
  var architecture: VisionModelArchitecture { get }
  var isReady: Bool { get }
  @discardableResult
  func prepare() throws -> String
  func classify(
    image: UIImage,
    completion: @escaping (Result<[MusicVisionLabel], Error>) -> Void
  )
  func classifySync(image: UIImage) throws -> [MusicVisionLabel]
}
