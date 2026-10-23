import ImageIO
import Vision
import XCTest
@testable import ExpoAppleIntelligence

/// Camera photos store their pixels in sensor orientation plus an EXIF tag
/// (an iPhone portrait is landscape pixels + orientation 6). Everything the
/// module hands back — the decoded pixels, their size, and Vision's
/// normalized boxes and landmarks — must be in the upright frame a viewer
/// shows, or a face box cropped out of an upright thumbnail lands beside the
/// face. The `face-orientation-*` fixtures are `face.jpg` stored rotated with
/// each tag (see `scripts/make-orientation-fixtures.swift`).
final class OrientationTests: XCTestCase {

  private static let tags = [1, 3, 6, 8]
  private static let formats = ["jpg", "heic"]
  private static let uprightSize = (width: 400, height: 600)

  private var variants: [String] {
    Self.tags.flatMap { tag in Self.formats.map { "face-orientation-\(tag).\($0)" } }
  }

  private var isSimulator: Bool {
    #if targetEnvironment(simulator)
    return true
    #else
    return false
    #endif
  }

  // MARK: - Fixtures model the camera

  func testFixtures_storeRotatedPixelsUnderTheirTag() throws {
    for tag in Self.tags {
      for format in Self.formats {
        let name = "face-orientation-\(tag).\(format)"
        let source = try imageSource(name)
        let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        XCTAssertEqual(props?[kCGImagePropertyOrientation] as? Int, tag, name)

        let raw = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil), name)
        let sideways = tag == 6 || tag == 8
        XCTAssertEqual(raw.width, sideways ? Self.uprightSize.height : Self.uprightSize.width, name)
        XCTAssertEqual(raw.height, sideways ? Self.uprightSize.width : Self.uprightSize.height, name)
      }
    }
  }

  // MARK: - Size and pixels

  func testLoadImage_reportsTheUprightSize() throws {
    for name in variants {
      let (image, width, height) = try XCTUnwrap(VisionHelpers.loadImage(FixturePaths.path(name)), name)
      XCTAssertEqual(Int(width), Self.uprightSize.width, name)
      XCTAssertEqual(Int(height), Self.uprightSize.height, name)
      XCTAssertEqual(image.width, Self.uprightSize.width, name)
      XCTAssertEqual(image.height, Self.uprightSize.height, name)
    }
  }

  func testDecodePixels_matchTheUprightPhoto() throws {
    let reference = try decoded("face.jpg")
    for name in variants {
      let pixels = try decoded(name)
      XCTAssertEqual(pixels.count, reference.count, name)
      guard pixels.count == reference.count else { continue }
      // Re-encoding costs a few levels per channel; a rotated or flipped face
      // differs by tens.
      XCTAssertLessThan(meanAbsoluteDifference(pixels, reference), 6, name)
    }
  }

  func testDecodePixels_upsideDownDiffersFromUpright() throws {
    // Guards the tolerance above: an upside-down decode of the same face must
    // fail it, or the pixel test could not tell a rotation from noise.
    let reference = try decoded("face.jpg")
    let source = try imageSource("face-orientation-3.jpg")
    let raw = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
    let pixels = try XCTUnwrap(VisionHelpers.decodePixels(raw))
    XCTAssertGreaterThan(meanAbsoluteDifference([UInt8](pixels), reference), 20)
  }

  // MARK: - Vision geometry

  func testDetectFaces_boxesAndLandmarksAreInTheUprightFrame() throws {
    try XCTSkipIf(isSimulator, "Face detection requires Neural Engine")
    let reference = try face("face.jpg")
    for name in variants {
      let found = try face(name)
      XCTAssertEqual(found.box.midX, reference.box.midX, accuracy: 0.03, name)
      XCTAssertEqual(found.box.midY, reference.box.midY, accuracy: 0.03, name)
      XCTAssertEqual(found.box.width, reference.box.width, accuracy: 0.03, name)
      XCTAssertEqual(found.box.height, reference.box.height, accuracy: 0.03, name)
      for (point, expected) in zip(found.points, reference.points) {
        XCTAssertEqual(point.x, expected.x, accuracy: 0.03, name)
        XCTAssertEqual(point.y, expected.y, accuracy: 0.03, name)
      }
      // Upright, in Vision's bottom-left-origin frame: eyes above the mouth,
      // the subject's left eye on the image's right.
      let (leftEye, rightEye, mouth) = (found.points[0], found.points[1], found.points[2])
      XCTAssertGreaterThan(leftEye.y, mouth.y, name)
      XCTAssertGreaterThan(rightEye.y, mouth.y, name)
      XCTAssertLessThan(leftEye.x, rightEye.x, name)
    }
  }

  // MARK: - Helpers

  /// The fixture as ImageIO stores it, before any orientation is applied.
  private func imageSource(_ name: String) throws -> CGImageSource {
    let url = URL(fileURLWithPath: FixturePaths.path(name))
    return try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil), name)
  }

  private func decoded(_ name: String) throws -> [UInt8] {
    let (image, _, _) = try XCTUnwrap(VisionHelpers.loadImage(FixturePaths.path(name)), name)
    return [UInt8](try XCTUnwrap(VisionHelpers.decodePixels(image), name))
  }

  private func meanAbsoluteDifference(_ a: [UInt8], _ b: [UInt8]) -> Double {
    var total = 0
    for i in 0..<min(a.count, b.count) { total += abs(Int(a[i]) - Int(b[i])) }
    return Double(total) / Double(max(1, min(a.count, b.count)))
  }

  /// The one face in a fixture, read from the module's serialized output the
  /// way the JS side receives it: the box, and the image-normalized centres
  /// of the left eye, right eye and outer lips.
  private func face(_ name: String) throws -> (box: CGRect, points: [CGPoint]) {
    let (image, _, _) = try XCTUnwrap(VisionHelpers.loadImage(FixturePaths.path(name)), name)
    let request = VNDetectFaceLandmarksRequest()
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    let faces = VisionProcessors.processFaces(landmarks: request.results ?? [], quality: [])
    XCTAssertEqual(faces.count, 1, "\(name): faces")
    let first = try XCTUnwrap(faces.first, "\(name): no face")

    let b = try XCTUnwrap(first["boundingBox"] as? [String: Double], name)
    let box = CGRect(x: b["x"]!, y: b["y"]!, width: b["width"]!, height: b["height"]!)
    let landmarks = try XCTUnwrap(first["landmarks"] as? [String: [[String: Double]]], name)
    let points = try ["leftEye", "rightEye", "outerLips"].map { key -> CGPoint in
      let region = try XCTUnwrap(landmarks[key], "\(name): \(key)")
      let x = region.map { $0["x"]! }.reduce(0, +) / Double(region.count)
      let y = region.map { $0["y"]! }.reduce(0, +) / Double(region.count)
      // Landmarks are normalized to the face box; lift them to the image.
      return CGPoint(x: box.minX + x * box.width, y: box.minY + y * box.height)
    }
    return (box, points)
  }
}
