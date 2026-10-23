import JavaScriptCore
import Vision
import XCTest
@testable import ExpoAppleIntelligence

/// `ModelInputs` against the JS pipeline it replaced, on every fixture.
///
/// `Fixtures/ml-reference.js` is that pipeline, bundled from the InsidePics
/// monorepo by `scripts/ml-reference/build.sh`: the CLIP crop loop and
/// pixel-space face alignment as they ran in JS, plus the monorepo's
/// `alignMath`, `blurMath`, `tensorPrep` and `histogram` modules, all as of
/// monorepo commit 79a602613, the last to run them on device. Both sides
/// read the same decoded pixels, so any difference is the port's, and the
/// bar is bit-identical: an embedding computed from the native input must be
/// the one the JS input produced.
final class ModelInputParityTests: XCTestCase {

  static let photos = [
    "face.jpg", "landscape.jpg", "body.jpg", "hand.jpg", "animal.jpg", "text.jpg",
  ] + [1, 3, 6, 8].flatMap { ["face-orientation-\($0).jpg", "face-orientation-\($0).heic"] }

  static let clipMean = [0.485, 0.456, 0.406]
  static let clipStd = [0.229, 0.224, 0.225]

  private var js: JSContext!
  private var reference: JSValue { js.objectForKeyedSubscript("MLReference") }

  override func setUpWithError() throws {
    js = try XCTUnwrap(JSContext())
    js.exceptionHandler = { _, exception in
      XCTFail("JS exception: \(exception?.toString() ?? "?")")
    }
    // histogram.ts yields between slices with setTimeout; run it inline.
    let setTimeout: @convention(block) (JSValue) -> Void = { $0.call(withArguments: []) }
    js.setObject(setTimeout, forKeyedSubscript: "setTimeout" as NSString)
    let source = try String(contentsOfFile: FixturePaths.path("ml-reference.js"), encoding: .utf8)
    js.evaluateScript(source)
    XCTAssertTrue(reference.isObject, "ml-reference.js did not load")
  }

  // MARK: - CLIP

  func testCenterCropTensor_matchesJsBitForBit() throws {
    for name in Self.photos {
      let image = try decode(name)
      let expected = try floats(reference.invokeMethod(
        "clipTensor", withArguments: [uint8Array(image.pixels), image.width, image.height]))
      let actual = ModelInputs.centerCropTensor(
        image, size: 256, mean: Self.clipMean, std: Self.clipStd)
      assertBitIdentical(actual, expected, name)
    }
  }

  // MARK: - Faces

  func testAlignedFace_matchesJsBitForBit() throws {
    try XCTSkipIf(isSimulator, "Face detection requires Neural Engine")
    var aligned = 0
    for name in Self.photos {
      let image = try decode(name)
      for face in try faces(name) where try assertAlignmentParity(image, face, name) {
        aligned += 1
      }
    }
    // face.jpg and its eight rotated copies at least.
    XCTAssertGreaterThanOrEqual(aligned, 9)
  }

  /// No Vision needed, so this runs on a simulator too: fixed faces that are
  /// tilted, tiny, huge, hanging off each edge, or degenerate.
  func testAlignedFace_syntheticFacesMatchJsBitForBit() throws {
    var aligned = 0
    for name in Self.photos {
      let image = try decode(name)
      for face in Self.syntheticFaces where try assertAlignmentParity(image, face, name) {
        aligned += 1
      }
    }
    // Every photo aligns every face but the collapsed one, which both sides
    // decline or (when the five identical points' mean rounds) both align.
    XCTAssertGreaterThanOrEqual(aligned, Self.photos.count * (Self.syntheticFaces.count - 1))
  }

  /// A face as the module serializes it: a Vision box (bottom-left origin)
  /// and landmark regions normalized to the box.
  private static func syntheticFace(
    box: [Double], tilt: Double = 0, collapsed: Bool = false
  ) -> [String: Any] {
    func pt(_ x: Double, _ y: Double) -> [String: Double] {
      collapsed ? ["x": 0.5, "y": 0.5] : ["x": x, "y": y + tilt * (x - 0.5)]
    }
    return [
      "boundingBox": ["x": box[0], "y": box[1], "width": box[2], "height": box[3]],
      "confidence": 0.99,
      "landmarks": [
        "leftEye": [pt(0.25, 0.68), pt(0.38, 0.7)],
        "rightEye": [pt(0.62, 0.7), pt(0.75, 0.68)],
        "nose": [pt(0.5, 0.55), pt(0.47, 0.45), pt(0.53, 0.45)],
        "outerLips": [0.32, 0.38, 0.45, 0.5, 0.55, 0.62, 0.68].map { pt($0, 0.25) },
      ],
    ]
  }

  private static let syntheticFaces = [
    syntheticFace(box: [0.3, 0.3, 0.4, 0.3]),
    syntheticFace(box: [0.35, 0.4, 0.3, 0.2], tilt: 0.4),
    syntheticFace(box: [0.48, 0.48, 0.03, 0.02]),  // a few pixels
    syntheticFace(box: [-0.2, -0.1, 1.4, 1.2]),  // bigger than the frame
    syntheticFace(box: [-0.25, 0.2, 0.4, 0.3]),  // off the left edge
    syntheticFace(box: [0.8, 0.85, 0.4, 0.3], tilt: -0.3),  // off top right
    syntheticFace(box: [0.3, 0.3, 0.4, 0.3], collapsed: true),  // degenerate
  ]

  /// Native and JS align `face` identically: the same tensor and blur bits,
  /// or both decline. Returns whether a face was aligned.
  private func assertAlignmentParity(
    _ image: RGBAImage, _ face: [String: Any], _ name: String
  ) throws -> Bool {
    let targets = try XCTUnwrap(
      reference.objectForKeyedSubscript("targetLandmarks").toArray() as? [[Double]])
    let jsFace = try XCTUnwrap(JSValue(object: face, in: js))
    let expected = try XCTUnwrap(reference.invokeMethod(
      "alignFace", withArguments: [uint8Array(image.pixels), image.width, image.height, jsFace]))
    // The consumer sends image-normalized landmarks: the pixel-space
    // extraction at a 1×1 size, since multiplying by 1 is exact.
    let points = reference.invokeMethod("extractFiveLandmarks", withArguments: [jsFace, 1, 1])?
      .toArray() as? [[Double]]
    let actual = points.flatMap {
      ModelInputs.alignedFace(image, landmarks: $0, targets: targets, size: 112)
    }
    guard let actual, !expected.isNull else {
      XCTAssertTrue(expected.isNull, "\(name): JS aligned a face native declined")
      XCTAssertNil(actual, "\(name): native aligned a face JS declined")
      return false
    }
    assertBitIdentical(actual.tensor, try floats(expected.objectForKeyedSubscript("tensor")), name)
    let blur = expected.objectForKeyedSubscript("blur").toDouble()
    XCTAssertEqual(actual.blur.bitPattern, blur.bitPattern, "\(name): blur \(actual.blur) vs \(blur)")
    return true
  }

  func testAlignedFace_degenerateLandmarksAreNil() {
    let image = RGBAImage(width: 4, height: 4, pixels: [UInt8](repeating: 9, count: 64))
    let same = [[Double]](repeating: [0.5, 0.5], count: 5)
    XCTAssertNil(ModelInputs.alignedFace(image, landmarks: same, targets: same, size: 112))
  }

  // MARK: - Video frames

  func testBlurScore_matchesJsBitForBit() throws {
    for name in Self.photos {
      let image = try decode(name)
      let expected = reference.invokeMethod(
        "blurScoreFromRgba", withArguments: [uint8Array(image.pixels), image.width, image.height])
      let actual = ModelInputs.blurScore(image.pixels, width: image.width, height: image.height)
      XCTAssertEqual(actual.bitPattern, expected?.toDouble().bitPattern, name)
    }
    XCTAssertEqual(ModelInputs.blurScore([UInt8](repeating: 0, count: 16), width: 2, height: 2), 0)
  }

  func testLuminanceHistogram_matchesJsBitForBit() throws {
    for name in Self.photos {
      let image = try decode(name)
      js.setObject(uint8Array(image.pixels), forKeyedSubscript: "pixels" as NSString)
      js.evaluateScript("""
        globalThis.histogram = undefined;
        MLReference.luminanceHistogram(pixels, \(image.width * image.height))
          .then((h) => { globalThis.histogram = h; });
        """)
      // The setTimeout shim runs inline, so the promise has settled once the
      // script's microtasks drained.
      let settled = js.objectForKeyedSubscript("histogram")
      XCTAssertFalse(settled?.isUndefined ?? true, "\(name): histogram promise did not settle")
      let expected = try floats(settled)
      assertBitIdentical(ModelInputs.luminanceHistogram(image, bins: 64), expected, name)
    }
  }

  // MARK: - Transfer

  func testBytes_areLittleEndianFloat32() {
    let data = ModelInputs.bytes([1, -0.5])
    XCTAssertEqual([UInt8](data), [0, 0, 0x80, 0x3F, 0, 0, 0, 0xBF])
  }

  // MARK: - Helpers

  private var isSimulator: Bool {
    #if targetEnvironment(simulator)
    return true
    #else
    return false
    #endif
  }

  private func decode(_ name: String) throws -> RGBAImage {
    try XCTUnwrap(VisionHelpers.decodeRGBA(path: FixturePaths.path(name)), name)
  }

  /// Faces as the module serializes them to JS.
  private func faces(_ name: String) throws -> [[String: Any]] {
    let (image, _, _) = try XCTUnwrap(VisionHelpers.loadImage(FixturePaths.path(name)), name)
    let request = VNDetectFaceLandmarksRequest()
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    return VisionProcessors.processFaces(landmarks: request.results ?? [], quality: [])
  }

  private func uint8Array(_ bytes: [UInt8]) -> JSValue {
    let count = bytes.count
    let copy = UnsafeMutableRawPointer.allocate(byteCount: count, alignment: 16)
    bytes.withUnsafeBytes { copy.copyMemory(from: $0.baseAddress!, byteCount: count) }
    let object = JSObjectMakeTypedArrayWithBytesNoCopy(
      js.jsGlobalContextRef, kJSTypedArrayTypeUint8Array, copy, count,
      { bytes, _ in bytes?.deallocate() }, nil, nil)
    return JSValue(jsValueRef: object, in: js)
  }

  private func floats(_ value: JSValue?) throws -> [Float] {
    let ref = try XCTUnwrap(value?.jsValueRef)
    let ctx = js.jsGlobalContextRef
    let object = try XCTUnwrap(JSValueToObject(ctx, ref, nil))
    XCTAssertEqual(JSValueGetTypedArrayType(ctx, ref, nil), kJSTypedArrayTypeFloat32Array)
    let count = JSObjectGetTypedArrayLength(ctx, object, nil)
    let base = try XCTUnwrap(JSObjectGetTypedArrayBytesPtr(ctx, object, nil))
    let offset = JSObjectGetTypedArrayByteOffset(ctx, object, nil)
    let start = (base + offset).assumingMemoryBound(to: Float.self)
    return Array(UnsafeBufferPointer(start: start, count: count))
  }

  private func assertBitIdentical(
    _ actual: [Float], _ expected: [Float], _ name: String,
    file: StaticString = #filePath, line: UInt = #line
  ) {
    XCTAssertEqual(actual.count, expected.count, "\(name): length", file: file, line: line)
    guard actual.count == expected.count else { return }
    var mismatches = 0
    var maxDiff: Float = 0
    for i in 0..<actual.count where actual[i].bitPattern != expected[i].bitPattern {
      mismatches += 1
      maxDiff = max(maxDiff, abs(actual[i] - expected[i]))
    }
    XCTAssertEqual(
      mismatches, 0, "\(name): \(mismatches) of \(actual.count) values differ, max \(maxDiff)",
      file: file, line: line)
  }
}
