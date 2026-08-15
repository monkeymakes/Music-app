import Foundation
import UIKit

/// Accessibility helpers so MobileNet predictions are usable with VoiceOver
/// and other assistive technologies on iPhone.
public enum MusicVisionAccessibility {
  /// Spoken / accessibility summary of model labels, highest confidence first.
  public static func announcement(for labels: [MusicVisionLabel], architecture: VisionModelArchitecture = .mobileNet) -> String {
    guard let top = labels.max(by: { $0.confidence < $1.confidence }) else {
      return "No music scene detected."
    }

    let percent = Int((top.confidence * 100).rounded())
    var parts = [
      "\(architecture.displayName) result: \(friendlyLabel(top.text)), \(percent) percent confidence."
    ]

    let extras = labels
      .sorted { $0.confidence > $1.confidence }
      .dropFirst()
      .prefix(2)
      .map { "\(friendlyLabel($0.text)) \(Int(($0.confidence * 100).rounded())) percent" }

    if !extras.isEmpty {
      parts.append("Also possible: \(extras.joined(separator: ", ")).")
    }

    return parts.joined(separator: " ")
  }

  /// Posts a VoiceOver announcement after inference (no-op if VoiceOver is off).
  @MainActor
  public static func announce(_ labels: [MusicVisionLabel], architecture: VisionModelArchitecture = .mobileNet) {
    guard UIAccessibility.isVoiceOverRunning else { return }
    let message = announcement(for: labels, architecture: architecture)
    UIAccessibility.post(notification: .announcement, argument: message)
  }

  /// Human-friendly wording for class ids like `album_cover`.
  public static func friendlyLabel(_ raw: String) -> String {
    raw
      .replacingOccurrences(of: "_", with: " ")
      .trimmingCharacters(in: .whitespacesAndNewlines)
  }

  /// Large Dynamic Type–friendly accessibility traits for result rows.
  public static func resultAccessibilityLabel(for label: MusicVisionLabel) -> String {
    let percent = Int((label.confidence * 100).rounded())
    return "\(friendlyLabel(label.text)), \(percent) percent"
  }
}
