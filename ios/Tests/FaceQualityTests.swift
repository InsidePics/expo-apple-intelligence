import XCTest
import Vision
@testable import ExpoAppleIntelligence

/// Every detected face carries its capture quality. The landmarks and quality
/// requests each detect faces on their own, so their observations do not share
/// UUIDs: quality has to be measured on the landmark request's faces.
final class FaceQualityTests: XCTestCase {

  private let fixtures = [
    "face.jpg", "face-orientation-1.jpg", "face-orientation-3.jpg",
    "face-orientation-6.jpg", "face-orientation-8.jpg",
  ]

  func testDetectFaces_everyFaceHasQuality() throws {
    #if targetEnvironment(simulator)
    throw XCTSkip("Face detection requires Neural Engine")
    #endif
    for name in fixtures {
      guard let (cgImage, _, _) = VisionHelpers.loadImage(FixturePaths.path(name)) else {
        XCTFail("Could not load \(name)")
        continue
      }
      let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
      let faces = try VisionProcessors.detectFaces(handler)

      XCTAssertGreaterThanOrEqual(faces.count, 1, "\(name): should detect a face")
      for face in faces {
        let quality = face["quality"] as? Double
        XCTAssertNotNil(quality, "\(name): every face should have a quality")
        if let quality {
          XCTAssertGreaterThanOrEqual(quality, 0.0, "\(name): quality >= 0")
          XCTAssertLessThanOrEqual(quality, 1.0, "\(name): quality <= 1")
        }
      }
    }
  }
}
