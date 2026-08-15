import Foundation
import UIKit

public enum ServerSideInferenceError: Error, LocalizedError {
  case invalidURL
  case invalidResponse
  case httpStatus(Int, String)
  case encodingFailed
  case notPrepared
  case emptyLabels

  public var errorDescription: String? {
    switch self {
    case .invalidURL:
      return "Invalid server-side inference URL."
    case .invalidResponse:
      return "Invalid response from inference server."
    case .httpStatus(let code, let body):
      return "Inference server HTTP \(code): \(body.prefix(200))"
    case .encodingFailed:
      return "Could not encode image as JPEG for upload."
    case .notPrepared:
      return "Call prepare() before classify(); server URL or model pin is missing."
    case .emptyLabels:
      return "Server returned no labels."
    }
  }
}

/// One prediction from the server-side `/v1/infer` API.
public struct ServerSideInferResponse: Decodable, Sendable {
  public var inferenceId: String
  public var model: String
  public var channel: String
  public var version: String
  public var architecture: String
  public var deviceClass: String
  public var runtime: String
  public var engineMode: String
  public var latencyMs: Double
  public var labels: [ServerSideLabel]

  enum CodingKeys: String, CodingKey {
    case model, channel, version, architecture, runtime, labels
    case inferenceId = "inference_id"
    case deviceClass = "device_class"
    case engineMode = "engine_mode"
    case latencyMs = "latency_ms"
  }
}

public struct ServerSideLabel: Decodable, Sendable {
  public var text: String
  public var confidence: Float
  public var index: Int
}

/// HTTP client for versioned server-side Music Vision inference.
public final class ServerSideInferenceClient: @unchecked Sendable {
  public struct Configuration: Sendable {
    public var baseURL: URL
    public var modelName: String
    public var channel: String
    public var pinnedVersion: String?
    public var topK: Int
    public var jpegQuality: CGFloat

    public init(
      baseURL: URL,
      modelName: String = "music_vision_efficientnet",
      channel: String = "production",
      pinnedVersion: String? = nil,
      topK: Int = 5,
      jpegQuality: CGFloat = 0.85
    ) {
      self.baseURL = baseURL
      self.modelName = modelName
      self.channel = channel
      self.pinnedVersion = pinnedVersion
      self.topK = topK
      self.jpegQuality = jpegQuality
    }
  }

  public let configuration: Configuration
  public private(set) var activeVersion: ModelVersionInfo?
  public private(set) var isReady = false

  private let session: URLSession
  private let resolver: ModelVersionResolver

  public init(configuration: Configuration, session: URLSession = .shared) {
    self.configuration = configuration
    self.session = session
    self.resolver = ModelVersionResolver(baseURL: configuration.baseURL, session: session)
  }

  /// Resolves the active registry pin (unless an exact version is already pinned).
  @discardableResult
  public func prepare() async throws -> ModelVersionInfo {
    if let pinned = configuration.pinnedVersion {
      let info = ModelVersionInfo(
        model: configuration.modelName,
        channel: "pin",
        version: pinned,
        architecture: "",
        deviceClass: "",
        format: "tflite",
        artifactPresent: nil,
        notes: "Client-pinned version"
      )
      activeVersion = info
      isReady = true
      return info
    }

    let info = try await resolver.fetchActive(
      model: configuration.modelName,
      channel: configuration.channel
    )
    activeVersion = info
    isReady = true
    return info
  }

  /// Uploads a JPEG and returns server labels for the pinned/channel version.
  public func classify(image: UIImage) async throws -> ServerSideInferResponse {
    guard isReady else { throw ServerSideInferenceError.notPrepared }
    guard let jpeg = image.jpegData(compressionQuality: configuration.jpegQuality) else {
      throw ServerSideInferenceError.encodingFailed
    }

    let url = configuration.baseURL.appendingPathComponent("v1/infer")
    var request = URLRequest(url: url)
    request.httpMethod = "POST"

    let boundary = "Boundary-\(UUID().uuidString)"
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

    var body = Data()
    func appendField(_ name: String, _ value: String) {
      body.append("--\(boundary)\r\n".data(using: .utf8)!)
      body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
      body.append("\(value)\r\n".data(using: .utf8)!)
    }

    appendField("model", configuration.modelName)
    appendField("channel", configuration.channel)
    appendField("top_k", String(configuration.topK))
    if let pinned = configuration.pinnedVersion ?? activeVersion?.version {
      // Prefer explicit pin; otherwise send resolved version for reproducibility.
      if configuration.pinnedVersion != nil {
        appendField("version", pinned)
      }
    }

    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append(
      "Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n".data(using: .utf8)!
    )
    body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
    body.append(jpeg)
    body.append("\r\n".data(using: .utf8)!)
    body.append("--\(boundary)--\r\n".data(using: .utf8)!)
    request.httpBody = body

    let (data, response) = try await session.data(for: request)
    try ModelVersionResolver.throwIfNeeded(response: response, data: data)
    let decoded = try JSONDecoder().decode(ServerSideInferResponse.self, from: data)
    if decoded.labels.isEmpty {
      throw ServerSideInferenceError.emptyLabels
    }
    return decoded
  }
}
