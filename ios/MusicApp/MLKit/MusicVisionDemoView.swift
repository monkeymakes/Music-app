import SwiftUI

/// Minimal demo surface for the ML Kit bridge. Wire this into your app target
/// after `pod install` and adding `music_vision.tflite` to the bundle.
struct MusicVisionDemoView: View {
  @StateObject private var classifier = MusicVisionClassifier()
  @State private var status = "Tap Prepare, then pick album art."

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      Text("Music Vision")
        .font(.largeTitle.bold())

      Text(status)
        .foregroundStyle(.secondary)

      Button("Prepare ML Kit bridge") {
        classifier.prepare()
        status = classifier.isPrepared
          ? "Bridge ready."
          : (classifier.lastError ?? "Prepare failed.")
      }
      .buttonStyle(.borderedProminent)

      if !classifier.lastLabels.isEmpty {
        ForEach(Array(classifier.lastLabels.enumerated()), id: \.offset) { _, label in
          HStack {
            Text(label.text)
            Spacer()
            Text(String(format: "%.0f%%", label.confidence * 100))
              .monospacedDigit()
          }
        }
      }
    }
    .padding()
  }
}
