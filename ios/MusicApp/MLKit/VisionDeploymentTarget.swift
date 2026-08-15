import Foundation

/// Which neural architecture Music Vision should run, by device class.
///
/// - **iOS phones:** MobileNet — low latency / low power so VoiceOver and
///   assistive flows stay responsive for users with disabilities.
/// - **Raspberry Pi:** EfficientNet — more accuracy on a plugged-in edge box
///   where battery and thermal limits are looser than a phone.
public enum VisionModelArchitecture: String, Sendable, Codable, CaseIterable {
  case mobileNet = "mobilenet"
  case efficientNet = "efficientnet"

  public var displayName: String {
    switch self {
    case .mobileNet: return "MobileNet"
    case .efficientNet: return "EfficientNet"
    }
  }

  /// Recommended TFLite resource name (no extension) for this architecture.
  public var bundledModelName: String {
    switch self {
    case .mobileNet: return "music_vision_mobilenet"
    case .efficientNet: return "music_vision_efficientnet"
    }
  }

  public var downloadedModelFileName: String {
    "\(bundledModelName)_remote.tflite"
  }
}

/// Device deployment profile for Music Vision.
public enum VisionDeploymentTarget: String, Sendable, CaseIterable {
  case iOSPhone = "ios_phone"
  case raspberryPi = "raspberry_pi"

  /// Architecture chosen for this target.
  public var architecture: VisionModelArchitecture {
    switch self {
    case .iOSPhone: return .mobileNet
    case .raspberryPi: return .efficientNet
    }
  }

  public var rationale: String {
    switch self {
    case .iOSPhone:
      return "MobileNet keeps inference fast and battery-friendly so accessibility features (VoiceOver, larger type, reduced motion) stay smooth for handicapped users."
    case .raspberryPi:
      return "EfficientNet uses more compute for higher accuracy on always-on Raspberry Pi kiosks / edge cameras."
    }
  }
}
