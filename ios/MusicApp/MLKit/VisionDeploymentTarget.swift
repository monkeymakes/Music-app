import Foundation

/// Which neural architecture Music Vision should run, by device class.
///
/// - **iOS phones (default): TinyML** — MobileNetV3-Small / quantized — lowest
///   latency so VoiceOver and assistive flows stay responsive.
/// - **iOS phones (optional): MobileNet** — slightly larger full MobileNet.
/// - **Raspberry Pi:** EfficientNet — more accuracy on a plugged-in edge box.
public enum VisionModelArchitecture: String, Sendable, Codable, CaseIterable {
  case tinyML = "tinyml"
  case mobileNet = "mobilenet"
  case efficientNet = "efficientnet"

  public var displayName: String {
    switch self {
    case .tinyML: return "TinyML"
    case .mobileNet: return "MobileNet"
    case .efficientNet: return "EfficientNet"
    }
  }

  /// Recommended model resource name (no extension) for this architecture.
  public var bundledModelName: String {
    switch self {
    case .tinyML: return "music_vision_tinyml"
    case .mobileNet: return "music_vision_mobilenet"
    case .efficientNet: return "music_vision_efficientnet"
    }
  }

  public var downloadedModelFileName: String {
    "\(bundledModelName)_remote.tflite"
  }

  public var coreMLDownloadedFileName: String {
    "\(bundledModelName)_remote.mlpackage"
  }
}

/// Device deployment profile for Music Vision.
public enum VisionDeploymentTarget: String, Sendable, CaseIterable {
  case iOSPhone = "ios_phone"
  case raspberryPi = "raspberry_pi"

  /// Architecture chosen for this target.
  public var architecture: VisionModelArchitecture {
    switch self {
    case .iOSPhone: return .tinyML
    case .raspberryPi: return .efficientNet
    }
  }

  public var rationale: String {
    switch self {
    case .iOSPhone:
      return "TinyML (compact MobileNet) keeps inference fastest and battery-friendly so accessibility features (VoiceOver, larger type, reduced motion) stay smooth for handicapped users."
    case .raspberryPi:
      return "EfficientNet uses more compute for higher accuracy on always-on Raspberry Pi kiosks / edge cameras."
    }
  }
}
