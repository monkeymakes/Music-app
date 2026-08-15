import SwiftUI

/// Accessible MobileNet demo: large controls, VoiceOver labels, feedback buttons.
struct MusicVisionDemoView: View {
  @StateObject private var classifier = MusicVisionClassifier(deploymentTarget: .iOSPhone)
  @State private var status = "Prepare MobileNet, then run a prediction and rate it."
  @State private var comment = ""
  @State private var isSubmittingFeedback = false
  @Environment(\.dynamicTypeSize) private var dynamicTypeSize

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 20) {
        Text("Music Vision")
          .font(.largeTitle.bold())
          .accessibilityAddTraits(.isHeader)

        Text("iPhone uses MobileNet so results stay fast for assistive technologies.")
          .font(.body)
          .foregroundStyle(.secondary)
          .accessibilityLabel(
            "iPhone uses Mobile Net so results stay fast for assistive technologies."
          )

        Text(status)
          .font(.body)
          .foregroundStyle(.secondary)
          .accessibilityLabel(status)

        Button("Prepare MobileNet bridge") {
          classifier.prepare()
          status = classifier.isPrepared
            ? "MobileNet ready. VoiceOver will announce predictions."
            : (classifier.lastError ?? "Prepare failed.")
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .accessibilityHint("Loads the MobileNet model for on-device music scene detection.")

        if !classifier.lastLabels.isEmpty {
          Text(classifier.accessibilitySummary)
            .font(.title3)
            .accessibilityLabel(classifier.accessibilitySummary)

          ForEach(Array(classifier.lastLabels.enumerated()), id: \.offset) { _, label in
            HStack {
              Text(MusicVisionAccessibility.friendlyLabel(label.text))
                .font(.title3)
              Spacer()
              Text(String(format: "%.0f%%", label.confidence * 100))
                .font(.title3.monospacedDigit())
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(MusicVisionAccessibility.resultAccessibilityLabel(for: label))
          }

          TextField("Optional feedback comment", text: $comment, axis: .vertical)
            .textFieldStyle(.roundedBorder)
            .lineLimit(2...4)
            .font(.body)
            .accessibilityLabel("Optional feedback comment about the prediction")

          HStack(spacing: 16) {
            Button("Helpful") {
              Task { await sendFeedback(.helpful) }
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .disabled(isSubmittingFeedback)
            .accessibilityHint("Marks this MobileNet prediction as correct.")

            Button("Not helpful") {
              Task { await sendFeedback(.notHelpful) }
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .disabled(isSubmittingFeedback)
            .accessibilityHint("Marks this MobileNet prediction as incorrect.")
          }
        }
      }
      .padding()
      .frame(maxWidth: dynamicTypeSize.isAccessibilitySize ? .infinity : 560, alignment: .leading)
    }
  }

  private func sendFeedback(_ rating: ModelFeedbackRating) async {
    isSubmittingFeedback = true
    defer { isSubmittingFeedback = false }
    do {
      try await classifier.submitFeedback(rating: rating, comment: comment)
      status = "Thanks — feedback saved to Firestore and Analytics."
      comment = ""
    } catch {
      status = error.localizedDescription
    }
  }
}
