import Foundation

#if canImport(FirebaseStorage)
import FirebaseStorage
#endif

/// Downloads a hosted `.tflite` model into the app documents directory so
/// `MLKitModelBridge` can load it as a `LocalModel` (ML Kit custom-model path).
///
/// Firebase Storage is optional: without it, call `install(from:)` with any
/// already-downloaded file URL (for example after your own Cloud Storage fetch).
public final class RemoteModelDownloader: Sendable {
  public let destinationFileName: String

  public init(destinationFileName: String = "music_vision_remote.tflite") {
    self.destinationFileName = destinationFileName
  }

  public var destinationURL: URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
      .appendingPathComponent(destinationFileName)
  }

  /// Copies an on-disk model into the documents path used by the bridge.
  public func install(from sourceURL: URL) throws -> URL {
    let destination = destinationURL
    if FileManager.default.fileExists(atPath: destination.path) {
      try FileManager.default.removeItem(at: destination)
    }
    try FileManager.default.copyItem(at: sourceURL, to: destination)
    return destination
  }

  /// Returns true when a previously downloaded/cached model is present.
  public var isCached: Bool {
    FileManager.default.fileExists(atPath: destinationURL.path)
  }

#if canImport(FirebaseStorage)
  /// Downloads from a `gs://` Firebase Storage URL into the bridge cache path.
  public func downloadFromFirebase(
    gsURL: String,
    completion: @escaping (Result<URL, Error>) -> Void
  ) {
    let storage = Storage.storage()
    let ref = storage.reference(forURL: gsURL)
    let destination = destinationURL

    ref.write(toFile: destination) { url, error in
      if let error {
        completion(.failure(error))
      } else if let url {
        completion(.success(url))
      } else {
        completion(.failure(
          NSError(
            domain: "RemoteModelDownloader",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "Model download returned no URL."]
          )
        ))
      }
    }
  }
#endif
}
