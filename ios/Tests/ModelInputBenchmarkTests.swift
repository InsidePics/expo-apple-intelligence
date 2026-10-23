import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import ExpoAppleIntelligence

/// Native cost of one 24 MP photo's model inputs, before and after #791:
/// `decodeImagePixels` (decode, then the whole RGBA as base64 for JS to
/// resize) against `centerCropTensor` + `alignFaces` (decode, then only the
/// tensors). Opt-in, because it writes a ~10 MB JPEG to a temporary
/// directory: `MODEL_INPUT_BENCHMARK=1 swift test --filter ModelInputBenchmarkTests`
/// (`TEST_RUNNER_MODEL_INPUT_BENCHMARK=1` through xcodebuild).
final class ModelInputBenchmarkTests: XCTestCase {

  private static var photo: String!

  override class func setUp() {
    super.setUp()
    guard ProcessInfo.processInfo.environment["MODEL_INPUT_BENCHMARK"] != nil else { return }
    photo = try? makePhoto(width: 4000, height: 6000)
  }

  override class func tearDown() {
    if let photo { try? FileManager.default.removeItem(atPath: photo) }
    super.tearDown()
  }

  override func setUpWithError() throws {
    try XCTSkipIf(Self.photo == nil, "Set MODEL_INPUT_BENCHMARK=1 to run")
  }

  private var metrics: [XCTMetric] { [XCTClockMetric(), XCTMemoryMetric(), XCTCPUMetric()] }

  func testBefore_fullResolutionBase64() {
    var transferred = 0
    measure(metrics: metrics) {
      autoreleasepool {
        let (image, _, _) = VisionHelpers.loadImage(Self.photo)!
        let pixels = VisionHelpers.decodePixels(image)!
        transferred = pixels.base64EncodedString().utf8.count
      }
    }
    print("[benchmark] before: \(transferred) bytes of base64 cross to JS")
  }

  func testAfter_modelInputsOnly() {
    var transferred = 0
    // A face in the middle of the frame, in the five-point layout JS sends.
    let face = [[0.45, 0.45], [0.55, 0.45], [0.5, 0.5], [0.46, 0.56], [0.54, 0.56]]
    let targets = [
      [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
      [41.5493, 92.3655], [70.7299, 92.2041],
    ]
    measure(metrics: metrics) {
      autoreleasepool {
        // Two native calls, as the app makes them: CLIP, then the faces.
        let clipImage = VisionHelpers.decodeRGBA(path: Self.photo)!
        let clip = ModelInputs.bytes(ModelInputs.centerCropTensor(
          clipImage, size: 256,
          mean: ModelInputParityTests.clipMean, std: ModelInputParityTests.clipStd))
        let faceImage = VisionHelpers.decodeRGBA(path: Self.photo)!
        let aligned = ModelInputs.alignedFace(faceImage, landmarks: face, targets: targets, size: 112)!
        transferred = clip.count + ModelInputs.bytes(aligned.tensor).count
      }
    }
    print("[benchmark] after: \(transferred) bytes of Float32 cross to JS")
  }

  /// A 24 MP portrait JPEG: `face.jpg` scaled up, so it decodes like a photo.
  private static func makePhoto(width: Int, height: Int) throws -> String {
    let source = VisionHelpers.loadImage(FixturePaths.path("face.jpg"))!.0
    let ctx = CGContext(
      data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    ctx.interpolationQuality = .high
    ctx.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
    let path = (NSTemporaryDirectory() as NSString)
      .appendingPathComponent("benchmark-\(UUID().uuidString).jpg")
    let dest = CGImageDestinationCreateWithURL(
      URL(fileURLWithPath: path) as CFURL, UTType.jpeg.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(
      dest, ctx.makeImage()!, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
    guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    return path
  }
}
