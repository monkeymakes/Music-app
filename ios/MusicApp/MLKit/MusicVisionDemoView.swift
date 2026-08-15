import SwiftUI

/// Host that rebuilds `MusicVisionClassifier` when the user toggles runtime.
struct MusicVisionDemoView: View {
  @State private var runtime: VisionInferenceRuntime = .coreML

  var body: some View {
    MusicVisionRuntimeDemo(runtime: runtime) { newRuntime in
      runtime = newRuntime
    }
    .id(runtime)
  }
}

/// Accessible demo for a single runtime: large controls, VoiceOver, feedback.
private struct MusicVisionRuntimeDemo: View {
  let runtime: VisionInferenceRuntime
  let onRuntimeChange: (VisionInferenceRuntime) -> Void

  @StateObject private var classifier: MusicVisionClassifier
  @State private var status: String
  @State private var comment = ""
  @State private var isSubmittingFeedback = false
  @Environment(\.dynamicTypeSize) private var dynamicTypeSize

  init(
    runtime: VisionInferenceRuntime,
    onRuntimeChange: @escaping (VisionInferenceRuntime) -> Void
  ) {
    self.runtime = runtime
    self.onRuntimeChange = onRuntimeChange
    _classifier = StateObject(wrappedValue: MusicVisionClassifier(runtime: runtime))
    _status = State(
      initialValue: "Prepare (\(runtime.displayName)), then run a prediction and rate it."
    )
  }

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 20) {
        Text("Music Vision")
          .font(.largeTitle.bold())
          .accessibilityAddTraits(.isHeader)

        Text("On-device TinyML stays fast for assistive tech. Server uses registry version pins.")
          .font(.body)
          .foregroundStyle(.secondary)

        Picker("Runtime", selection: Binding(
          get: { runtime },
          set: { onRuntimeChange($0) }
        )) {
          Text("Core ML").tag(VisionInferenceRuntime.coreML)
          Text("ML Kit").tag(VisionInferenceRuntime.mlKit)
          Text("Server").tag(VisionInferenceRuntime.serverSide)
        }
        .pickerStyle(.segmented)
        .accessibilityLabel("Inference runtime")

        Text(status)
          .font(.body)
          .foregroundStyle(.secondary)
          .accessibilityLabel(status)

        Button(prepareButtonTitle) {
          Task { await prepare() }
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .accessibilityHint(prepareHint)

        if !classifier.lastLabels.isEmpty {
          Text(classifier.accessibilitySummary)
            .font(.title3)
            .accessibilityLabel(classifier.accessibilitySummary)

          if let version = classifier.lastModelVersion {
            Text("Model version \(version)")
              .font(.footnote)
              .foregroundStyle(.secondary)
              .accessibilityLabel("Model version \(version)")
          }

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
            .accessibilityHint("Marks this prediction as correct.")

            Button("Not helpful") {
              Task { await sendFeedback(.notHelpful) }
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .disabled(isSubmittingFeedback)
            .accessibilityHint("Marks this prediction as incorrect.")
          }
        }
      }
      .padding()
      .frame(maxWidth: dynamicTypeSize.isAccessibilitySize ? .infinity : 560, alignment: .leading)
    }
  }

  private var prepareButtonTitle: String {
    switch runtime {
    case .serverSide:
      return "Prepare server (resolve version)"
    case .coreML, .mlKit:
      return "Prepare TinyML (\(runtime.displayName))"
    }
  }

  private var prepareHint: String {
    switch runtime {
    case .serverSide:
      return "Fetches the production model version from the registry, then enables remote inference."
    case .coreML, .mlKit:
      return "Loads the TinyML model for on-device music scene detection."
    }
  }

  private func prepare() async {
    if runtime == .serverSide {
      await classifier.prepareServerSide()
    } else {
      classifier.prepare()
    }
    if classifier.isPrepared {
      let versionNote = classifier.lastModelVersion.map { " v\($0)" } ?? ""
      status = "\(runtime.displayName)\(versionNote) ready. VoiceOver will announce predictions."
    } else {
      status = classifier.lastError ?? "Prepare failed."
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
