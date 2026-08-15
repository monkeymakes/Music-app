import Foundation
import FirebaseCore

/// Call once at app launch before using Analytics, Firestore, or Storage.
enum FirebaseBootstrap {
  private static let lock = NSLock()
  private static var didConfigure = false

  /// Configures `FirebaseApp` from `GoogleService-Info.plist` if present.
  /// Safe to call multiple times.
  @discardableResult
  static func configureIfNeeded() -> Bool {
    lock.lock()
    defer { lock.unlock() }

    if didConfigure || FirebaseApp.app() != nil {
      didConfigure = true
      return true
    }

    guard Bundle.main.path(forResource: "GoogleService-Info", ofType: "plist") != nil else {
      return false
    }

    FirebaseApp.configure()
    didConfigure = true
    return true
  }
}
