import XCTest
@testable import ExpoAppleIntelligence

final class VisionHelpersTests: XCTestCase {

  // MARK: - rawRect

  func testRawRect_serializesNormalizedRect() {
    let rect = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
    let result = VisionHelpers.rawRect(rect)

    XCTAssertEqual(result["x"]!, 0.25, accuracy: 0.001)
    XCTAssertEqual(result["y"]!, 0.25, accuracy: 0.001)
    XCTAssertEqual(result["width"]!, 0.5, accuracy: 0.001)
    XCTAssertEqual(result["height"]!, 0.5, accuracy: 0.001)
  }

  func testRawRect_fullImageBox() {
    let rect = CGRect(x: 0, y: 0, width: 1, height: 1)
    let result = VisionHelpers.rawRect(rect)

    XCTAssertEqual(result["x"]!, 0.0, accuracy: 0.001)
    XCTAssertEqual(result["y"]!, 0.0, accuracy: 0.001)
    XCTAssertEqual(result["width"]!, 1.0, accuracy: 0.001)
    XCTAssertEqual(result["height"]!, 1.0, accuracy: 0.001)
  }

  func testRawRect_zeroSizeBox() {
    let rect = CGRect(x: 0.5, y: 0.5, width: 0, height: 0)
    let result = VisionHelpers.rawRect(rect)

    XCTAssertEqual(result["x"]!, 0.5, accuracy: 0.001)
    XCTAssertEqual(result["y"]!, 0.5, accuracy: 0.001)
    XCTAssertEqual(result["width"]!, 0.0, accuracy: 0.001)
    XCTAssertEqual(result["height"]!, 0.0, accuracy: 0.001)
  }

  // MARK: - rawPoint

  func testRawPoint_serializesNormalizedPoint() {
    let point = CGPoint(x: 0.5, y: 0.75)
    let result = VisionHelpers.rawPoint(point)

    XCTAssertEqual(result["x"]!, 0.5, accuracy: 0.001)
    XCTAssertEqual(result["y"]!, 0.75, accuracy: 0.001)
  }

  func testRawPoint_originPoint() {
    let point = CGPoint(x: 0, y: 0)
    let result = VisionHelpers.rawPoint(point)

    XCTAssertEqual(result["x"]!, 0.0, accuracy: 0.001)
    XCTAssertEqual(result["y"]!, 0.0, accuracy: 0.001)
  }

  func testRawPoint_topRightCorner() {
    let point = CGPoint(x: 1, y: 1)
    let result = VisionHelpers.rawPoint(point)

    XCTAssertEqual(result["x"]!, 1.0, accuracy: 0.001)
    XCTAssertEqual(result["y"]!, 1.0, accuracy: 0.001)
  }

  // MARK: - loadImage

  func testLoadImage_returnsNilForNonexistentPath() {
    let result = VisionHelpers.loadImage("/nonexistent/path/image.jpg")
    XCTAssertNil(result)
  }

  func testLoadImage_returnsNilForEmptyPath() {
    let result = VisionHelpers.loadImage("")
    XCTAssertNil(result)
  }

  func testLoadImage_handlesFileUrlPrefix() {
    let result = VisionHelpers.loadImage("file:///nonexistent/path/image.jpg")
    XCTAssertNil(result)
  }

  func testLoadImage_loadsValidImage() {
    let path = createTestImage(width: 100, height: 50)
    defer { try? FileManager.default.removeItem(atPath: path) }

    let result = VisionHelpers.loadImage(path)
    XCTAssertNotNil(result)

    let (cgImage, width, height) = result!
    XCTAssertEqual(Int(width), 100)
    XCTAssertEqual(Int(height), 50)
    XCTAssertEqual(cgImage.width, 100)
    XCTAssertEqual(cgImage.height, 50)
  }

  func testLoadImage_loadsWithFileUrlPrefix() {
    let path = createTestImage(width: 64, height: 64)
    defer { try? FileManager.default.removeItem(atPath: path) }

    let fileUrl = "file://" + path
    let result = VisionHelpers.loadImage(fileUrl)
    XCTAssertNotNil(result)
  }

  // MARK: - decodePixels

  func testDecodePixels_returnsCorrectSize() {
    let path = createTestImage(width: 10, height: 10)
    defer { try? FileManager.default.removeItem(atPath: path) }

    let (cgImage, _, _) = VisionHelpers.loadImage(path)!
    let data = VisionHelpers.decodePixels(cgImage)

    XCTAssertNotNil(data)
    // 10x10 image * 4 bytes (RGBA) = 400 bytes
    XCTAssertEqual(data!.count, 10 * 10 * 4)
  }

  func testDecodePixels_redImageHasCorrectPixels() {
    // Decode directly from in-memory CGImage to avoid PNG round-trip
    // color space conversion artifacts
    let cgImage = createTestCGImage(width: 2, height: 2)
    let data = VisionHelpers.decodePixels(cgImage)!
    let bytes = [UInt8](data)

    // First pixel should be red (R=255, G=0, B=0, A=255)
    XCTAssertEqual(bytes[0], 255) // R
    XCTAssertEqual(bytes[1], 0)   // G
    XCTAssertEqual(bytes[2], 0)   // B
    XCTAssertEqual(bytes[3], 255) // A
  }

  func testDecodePixels_1x1Image() {
    let path = createTestImage(width: 1, height: 1)
    defer { try? FileManager.default.removeItem(atPath: path) }

    let (cgImage, _, _) = VisionHelpers.loadImage(path)!
    let data = VisionHelpers.decodePixels(cgImage)

    XCTAssertNotNil(data)
    XCTAssertEqual(data!.count, 4)
  }

  func testDecodePixels_largeImage() {
    let path = createTestImage(width: 1000, height: 500)
    defer { try? FileManager.default.removeItem(atPath: path) }

    let (cgImage, _, _) = VisionHelpers.loadImage(path)!
    let data = VisionHelpers.decodePixels(cgImage)

    XCTAssertNotNil(data)
    XCTAssertEqual(data!.count, 1000 * 500 * 4)
  }

  // MARK: - Helpers

  private func createTestCGImage(width: Int, height: Int) -> CGImage {
    // Must use the same color space as decodePixels (deviceRGB) and define
    // the fill color in that space to avoid cross-space conversion artifacts
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(
      data: nil, width: width, height: height,
      bitsPerComponent: 8, bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )!
    context.setFillColor(CGColor(colorSpace: colorSpace, components: [1, 0, 0, 1])!)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()!
  }

  private func createTestImage(width: Int, height: Int) -> String {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(
      data: nil, width: width, height: height,
      bitsPerComponent: 8, bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )!
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))

    let cgImage = context.makeImage()!
    let path = NSTemporaryDirectory() + "test_\(UUID().uuidString).png"

    #if canImport(UIKit)
    let uiImage = UIImage(cgImage: cgImage)
    try! uiImage.pngData()!.write(to: URL(fileURLWithPath: path))
    #elseif canImport(AppKit)
    let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
    let rep = NSBitmapImageRep(cgImage: cgImage)
    try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: path))
    #endif

    return path
  }
}
