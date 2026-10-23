import Foundation

/// Locates a file in `Tests/Fixtures` under every way the tests run: the
/// SwiftPM package (`swift test`, or `xcodebuild` on a device, where the
/// fixtures are a resource bundle), the CocoaPods test spec, and a plain
/// checkout as a last resort.
enum FixturePaths {
  private final class Anchor {}

  static func path(_ name: String) -> String {
    let base = (name as NSString).deletingPathExtension
    let ext = (name as NSString).pathExtension
    #if SWIFT_PACKAGE
    if let path = Bundle.module.path(forResource: base, ofType: ext, inDirectory: "Fixtures") {
      return path
    }
    #endif
    let bundle = Bundle(for: Anchor.self)
    if let path = bundle.path(forResource: base, ofType: ext, inDirectory: "Fixtures")
      ?? bundle.path(forResource: base, ofType: ext) {
      return path
    }
    let testDir = (#filePath as NSString).deletingLastPathComponent
    return (testDir as NSString).appendingPathComponent("Fixtures/\(name)")
  }
}
