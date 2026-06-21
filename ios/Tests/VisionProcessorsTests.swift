import XCTest
import Vision
@testable import ExpoAppleIntelligence

final class VisionProcessorsTests: XCTestCase {

  // MARK: - Classification

  func testProcessClassification_returnsAllObservations() {
    let path = createTestImage(width: 224, height: 224)
    defer { try? FileManager.default.removeItem(atPath: path) }

    guard let (cgImage, _, _) = VisionHelpers.loadImage(path) else {
      XCTFail("Could not load test image")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNClassifyImageRequest()
    try? handler.perform([req])

    let results = VisionProcessors.processClassification(req.results ?? [])

    // Should return all observations, unfiltered
    for item in results {
      XCTAssertNotNil(item["identifier"] as? String)
      XCTAssertNotNil(item["confidence"] as? Double)
    }
  }

  // MARK: - Text recognition

  func testProcessText_returnsEmptyForNoText() {
    let result = VisionProcessors.processText([])
    XCTAssertTrue(result.isEmpty)
  }

  // MARK: - Barcodes

  func testProcessBarcodes_returnsEmptyForNoObservations() {
    let result = VisionProcessors.processBarcodes([])
    XCTAssertTrue(result.isEmpty)
  }

  // MARK: - Body poses

  func testProcessBodyPoses_returnsEmptyForNoObservations() {
    let result = VisionProcessors.processBodyPoses([])
    XCTAssertTrue(result.isEmpty)
  }

  // MARK: - Hand poses

  func testProcessHandPoses_returnsEmptyForNoObservations() {
    let result = VisionProcessors.processHandPoses([])
    XCTAssertTrue(result.isEmpty)
  }

  // MARK: - Feature print

  func testProcessFeaturePrint_returnsNilForNilInput() {
    let result = VisionProcessors.processFeaturePrint(nil)
    XCTAssertNil(result)
  }

  func testProcessFeaturePrint_returnsValidDataForRealImage() throws {
    #if targetEnvironment(simulator)
    throw XCTSkip("Feature print requires Neural Engine (real device only)")
    #endif
    let path = createTestImage(width: 224, height: 224)
    defer { try? FileManager.default.removeItem(atPath: path) }

    guard let (cgImage, _, _) = VisionHelpers.loadImage(path) else {
      XCTFail("Could not load test image")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNGenerateImageFeaturePrintRequest()
    try? handler.perform([req])

    let obs = req.results?.first as? VNFeaturePrintObservation
    let result = VisionProcessors.processFeaturePrint(obs)

    XCTAssertNotNil(result)
    if let result = result {
      let data = result["data"] as? [Any]
      XCTAssertNotNil(data)
      XCTAssertGreaterThan(data!.count, 0, "Feature print should have elements")

      let elementType = result["elementType"] as? String
      XCTAssertTrue(elementType == "float" || elementType == "double")

      let elementCount = result["elementCount"] as? Double
      XCTAssertEqual(Int(elementCount!), data!.count)
    }
  }

  // MARK: - Saliency

  func testProcessSaliency_returnsEmptyForNoObservations() {
    let result = VisionProcessors.processSaliency([])
    XCTAssertTrue(result.isEmpty)
  }

  // MARK: - Animals

  func testProcessAnimals_returnsEmptyForNoObservations() {
    let result = VisionProcessors.processAnimals([])
    XCTAssertTrue(result.isEmpty)
  }

  // MARK: - Rectangles

  func testProcessRectangles_returnsEmptyForNoObservations() {
    let result = VisionProcessors.processRectangles([])
    XCTAssertTrue(result.isEmpty)
  }

  // MARK: - Horizon

  func testProcessHorizon_returnsNilForNilInput() {
    let result = VisionProcessors.processHorizon(nil)
    XCTAssertNil(result)
  }

  // MARK: - Faces (integration)

  func testProcessFaces_returnsEmptyForNoFaces() {
    let result = VisionProcessors.processFaces(
      landmarks: [], quality: []
    )
    XCTAssertTrue(result.isEmpty)
  }

  // MARK: - Helpers

  private func createTestImage(width: Int, height: Int) -> String {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(
      data: nil, width: width, height: height,
      bitsPerComponent: 8, bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )!
    context.setFillColor(CGColor(red: 0, green: 0, blue: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))

    let cgImage = context.makeImage()!
    let path = NSTemporaryDirectory() + "test_\(UUID().uuidString).png"

    #if canImport(UIKit)
    try! UIImage(cgImage: cgImage).pngData()!.write(to: URL(fileURLWithPath: path))
    #elseif canImport(AppKit)
    try! NSBitmapImageRep(cgImage: cgImage).representation(using: .png, properties: [:])!
      .write(to: URL(fileURLWithPath: path))
    #endif

    return path
  }
}
