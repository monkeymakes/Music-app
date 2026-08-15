import SwiftUI

/// Demo surface for ML Kit inference plus Firebase customer feedback.
struct MusicVisionDemoView: View {
  @StateObject private var classifier = MusicVisionClassifier()
  @State private var status = "Tap Prepare, then run a prediction and rate it."
  @State private var comment = ""
  @State private var isSubmittingFeedback = false

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      Text("Music Vision")
        .font(.largeTitle.bold())

      Text(status)
        .foregroundStyle(.secondary)

      Button("Prepare ML Kit bridge") {
        classifier.prepare()
        status = classifier.isPrepared
          ? "Bridge ready. Analytics will record prepare + inference."
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

        TextField("Optional feedback comment", text: $comment, axis: .vertical)
          .textFieldStyle(.roundedBorder)
          .lineLimit(2...4)

        HStack {
          Button("Helpful") {
            Task { await sendFeedback(.helpful) }
          }
          .buttonStyle(.bordered)
          .disabled(isSubmittingFeedback)

          Button("Not helpful") {
            Task { await sendFeedback(.notHelpful) }
          }
          .buttonStyle(.bordered)
          .disabled(isSubmittingFeedback)
        }
      }
    }
    .padding()
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
