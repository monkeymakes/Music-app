import Foundation
import UIKit

/// Resolves the active model version from the server-side registry API.
public struct ModelVersionInfo: Equatable, Sendable, Codable {
  public var model: String
  public var channel: String
  public var version: String
  public var architecture: String
  public var deviceClass: String
  public var format: String
  public var artifactPresent: Bool?
  public var notes: String?

  enum CodingKeys: String, CodingKey {
    case model, channel, version, architecture, format, notes
    case deviceClass = "device_class"
    case artifactPresent = "artifact_present"
  }
}

public final class ModelVersionResolver: Sendable {
  public let baseURL: URL
  public let session: URLSession

  public init(baseURL: URL, session: URLSession = .shared) {
    self.baseURL = baseURL
    self.session = session
  }

  /// GET /v1/models/{name}/active?channel=...
  public func fetchActive(
    model: String,
    channel: String = "production"
  ) async throws -> ModelVersionInfo {
    var components = URLComponents(
      url: baseURL.appendingPathComponent("v1/models/\(model)/active"),
      resolvingAgainstBaseURL: false
    )
    components?.queryItems = [URLQueryItem(name: "channel", value: channel)]
    guard let url = components?.url else {
      throw ServerSideInferenceError.invalidURL
    }

    let (data, response) = try await session.data(from: url)
    try Self.throwIfNeeded(response: response, data: data)
    return try JSONDecoder().decode(ModelVersionInfo.self, from: data)
  }

  static func throwIfNeeded(response: URLResponse, data: Data) throws {
    guard let http = response as? HTTPURLResponse else {
      throw ServerSideInferenceError.invalidResponse
    }
    guard (200..<300).contains(http.statusCode) else {
      let body = String(data: data, encoding: .utf8) ?? ""
      throw ServerSideInferenceError.httpStatus(http.statusCode, body)
    }
  }
}
