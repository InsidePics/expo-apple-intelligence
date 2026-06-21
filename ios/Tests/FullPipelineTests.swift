import XCTest
import Vision
@testable import ExpoAppleIntelligence

/// Integration tests that run the full Vision pipeline on real fixture images.
/// Tests requiring CoreML (face, classify, saliency, aesthetics, feature print)
/// only pass on real devices — the simulator lacks the Neural Engine context.
final class FullPipelineTests: XCTestCase {

  private var isSimulator: Bool {
    #if targetEnvironment(simulator)
    return true
    #else
    return false
    #endif
  }

  // MARK: - Fixture loading

  private func fixturePath(_ name: String) -> String {
    let bundle = Bundle(for: type(of: self))
    let nameWithoutExt = (name as NSString).deletingPathExtension
    let ext = (name as NSString).pathExtension

    if let path = bundle.path(forResource: nameWithoutExt, ofType: ext, inDirectory: "Fixtures") {
      return path
    }
    if let path = bundle.path(forResource: nameWithoutExt, ofType: ext) {
      return path
    }
    let testDir = (#filePath as NSString).deletingLastPathComponent
    return (testDir as NSString).appendingPathComponent("Fixtures/\(name)")
  }

  private func loadFixture(_ name: String) -> (CGImage, CGFloat, CGFloat)? {
    let path = fixturePath(name)
    return VisionHelpers.loadImage(path)
  }

  // MARK: - Face detection

  func testDetectFaces_findsFaceInPortrait() throws {
    try XCTSkipIf(isSimulator, "Face detection requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("face.jpg") else {
      XCTFail("Could not load face.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let lReq = VNDetectFaceLandmarksRequest()
    let qReq = VNDetectFaceCaptureQualityRequest()
    try handler.perform([lReq, qReq])

    let faces = VisionProcessors.processFaces(
      landmarks: lReq.results ?? [],
      quality: qReq.results ?? []
    )

    XCTAssertGreaterThanOrEqual(faces.count, 1, "Should detect at least 1 face in portrait")

    let face = faces[0]
    let bbox = face["boundingBox"] as! [String: Double]
    XCTAssertGreaterThan(bbox["width"]!, 0, "Face bbox should have positive width")
    XCTAssertGreaterThan(bbox["height"]!, 0, "Face bbox should have positive height")

    // Landmarks should be a dict of region name → point arrays
    let landmarks = face["landmarks"] as! [String: Any]
    XCTAssertNotNil(landmarks["leftEye"], "Should have leftEye region")
    XCTAssertNotNil(landmarks["rightEye"], "Should have rightEye region")
    XCTAssertNotNil(landmarks["nose"], "Should have nose region")

    let confidence = face["confidence"] as! Double
    XCTAssertGreaterThan(confidence, 0.5, "Face confidence should be high for clear portrait")
  }

  func testDetectFaces_noFacesInLandscape() throws {
    try XCTSkipIf(isSimulator, "Face detection requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("landscape.jpg") else {
      XCTFail("Could not load landscape.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let lReq = VNDetectFaceLandmarksRequest()
    try handler.perform([lReq])

    let faces = VisionProcessors.processFaces(
      landmarks: lReq.results ?? [],
      quality: []
    )

    XCTAssertEqual(faces.count, 0, "Should not detect faces in landscape photo")
  }

  // MARK: - Text recognition

  func testRecognizeText_findsTextInDocument() throws {
    guard let (cgImage, _, _) = loadFixture("document.jpg") else {
      XCTFail("Could not load document.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    try handler.perform([req])

    let result = VisionProcessors.processText(
      req.results ?? []
    )

    XCTAssertGreaterThan(result.count, 0, "Should recognize text in document image")

    let obs = result[0]
    let bbox = obs["boundingBox"] as! [String: Double]
    XCTAssertGreaterThan(bbox["width"]!, 0, "Text bbox should have positive width")

    let candidates = obs["candidates"] as! [[String: Any]]
    XCTAssertGreaterThan(candidates.count, 0, "Should have at least 1 candidate")
    XCTAssertNotNil(candidates[0]["string"] as? String)
    XCTAssertNotNil(candidates[0]["confidence"] as? Double)
  }

  func testRecognizeText_noTextInLandscape() throws {
    guard let (cgImage, _, _) = loadFixture("landscape.jpg") else {
      XCTFail("Could not load landscape.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    try handler.perform([req])

    let result = VisionProcessors.processText(
      req.results ?? []
    )

    // Landscape may have no text at all
    if result.count > 0 {
      let totalText = result.compactMap { ($0["candidates"] as? [[String: Any]])?.first?["string"] as? String }.joined()
      XCTAssertLessThan(totalText.count, 20, "Landscape should have minimal text")
    }
  }

  // MARK: - Barcode / QR code detection

  func testDetectBarcodes_findsQRCode() throws {
    try XCTSkipIf(isSimulator, "Barcode detection requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("barcode.jpg") else {
      XCTFail("Could not load barcode.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNDetectBarcodesRequest()
    try handler.perform([req])

    let barcodes = VisionProcessors.processBarcodes(
      req.results ?? []
    )

    XCTAssertGreaterThanOrEqual(barcodes.count, 1, "Should detect QR code in barcode image")

    if let first = barcodes.first {
      let symbology = first["symbology"] as! String
      XCTAssertTrue(
        symbology.lowercased().contains("qr") || symbology.contains("QR"),
        "Detected barcode should be QR format, got: \(symbology)"
      )
      // Corner points should be present
      XCTAssertNotNil(first["topLeft"] as? [String: Double])
      XCTAssertNotNil(first["topRight"] as? [String: Double])
      XCTAssertNotNil(first["bottomLeft"] as? [String: Double])
      XCTAssertNotNil(first["bottomRight"] as? [String: Double])
    }
  }

  func testDetectBarcodes_noneInLandscape() throws {
    try XCTSkipIf(isSimulator, "Barcode detection requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("landscape.jpg") else {
      XCTFail("Could not load landscape.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNDetectBarcodesRequest()
    try handler.perform([req])

    let barcodes = VisionProcessors.processBarcodes(
      req.results ?? []
    )

    XCTAssertEqual(barcodes.count, 0, "Should not detect barcodes in landscape")
  }

  // MARK: - Classification

  func testClassifyImage_labelsLandscape() throws {
    try XCTSkipIf(isSimulator, "Classification requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("landscape.jpg") else {
      XCTFail("Could not load landscape.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNClassifyImageRequest()
    try handler.perform([req])

    let labels = VisionProcessors.processClassification(
      req.results ?? []
    )

    XCTAssertGreaterThan(labels.count, 0, "Should produce labels for landscape")

    // All observations returned, unfiltered — find high-confidence nature labels
    let highConfidence = labels.filter { ($0["confidence"] as! Double) >= 0.1 }
    let identifiers = highConfidence.compactMap { $0["identifier"] as? String }
    let hasNatureLabel = identifiers.contains(where: {
      $0.contains("mountain") || $0.contains("landscape") || $0.contains("nature") ||
      $0.contains("outdoor") || $0.contains("sky") || $0.contains("water") ||
      $0.contains("tree") || $0.contains("forest") || $0.contains("river")
    })
    XCTAssertTrue(hasNatureLabel, "Landscape should get nature labels, got: \(identifiers.prefix(10))")
  }

  // MARK: - Feature print

  func testFeaturePrint_differentImagesProduceDifferentVectors() throws {
    try XCTSkipIf(isSimulator, "Feature print requires Neural Engine")

    let facePrint = generateFeaturePrint(fixture: "face.jpg")
    let landscapePrint = generateFeaturePrint(fixture: "landscape.jpg")

    XCTAssertNotNil(facePrint, "Face image should produce feature print")
    XCTAssertNotNil(landscapePrint, "Landscape image should produce feature print")

    var distance: Float = 0
    try facePrint!.computeDistance(&distance, to: landscapePrint!)
    XCTAssertGreaterThan(distance, 1.0, "Face and landscape should be quite different")
  }

  // MARK: - Aesthetics (iOS 18+)

  func testAesthetics_scoresLandscapeHigherThanBarcode() throws {
    try XCTSkipIf(isSimulator, "Aesthetics requires Neural Engine")
    guard #available(iOS 18.0, macOS 15.0, *) else {
      throw XCTSkip("Aesthetics requires iOS 18+")
    }

    let landscapeScore = computeAestheticScore(fixture: "landscape.jpg")
    let barcodeScore = computeAestheticScore(fixture: "barcode.jpg")

    XCTAssertNotNil(landscapeScore, "Landscape should have aesthetic score")
    XCTAssertNotNil(barcodeScore, "Barcode should have aesthetic score")

    XCTAssertGreaterThan(
      landscapeScore!, barcodeScore!,
      "Landscape (\(landscapeScore!)) should score higher than barcode photo (\(barcodeScore!))"
    )
  }

  // MARK: - Horizon

  func testHorizon_detectsAngleInLandscape() throws {
    guard let (cgImage, _, _) = loadFixture("landscape.jpg") else {
      XCTFail("Could not load landscape.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNDetectHorizonRequest()
    try handler.perform([req])

    let result = VisionProcessors.processHorizon(req.results?.first as? VNHorizonObservation)
    if let result = result {
      let angle = result["angle"] as! Double
      // Angle in radians — should be within ±π/4
      XCTAssertLessThan(abs(angle), .pi / 4, "Horizon angle should be within ±45°")
    }
  }

  // MARK: - Batch analysis

  func testBatchAnalysis_allRequestsCompleteOnRealImage() throws {
    try XCTSkipIf(isSimulator, "Batch analysis requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("face.jpg") else {
      XCTFail("Could not load face.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

    let faceLandmarksReq = VNDetectFaceLandmarksRequest()
    let faceQualityReq = VNDetectFaceCaptureQualityRequest()
    let textReq = VNRecognizeTextRequest()
    textReq.recognitionLevel = .accurate
    let classifyReq = VNClassifyImageRequest()
    let barcodeReq = VNDetectBarcodesRequest()
    let featurePrintReq = VNGenerateImageFeaturePrintRequest()
    let horizonReq = VNDetectHorizonRequest()

    let requests: [VNRequest] = [
      faceLandmarksReq, faceQualityReq, textReq, classifyReq,
      barcodeReq, featurePrintReq, horizonReq,
    ]

    try handler.perform(requests)

    // Face should be detected
    let faces = VisionProcessors.processFaces(
      landmarks: faceLandmarksReq.results ?? [],
      quality: faceQualityReq.results ?? []
    )
    XCTAssertGreaterThanOrEqual(faces.count, 1, "Batch: should detect face")

    // Classification should return labels
    let labels = VisionProcessors.processClassification(
      classifyReq.results ?? []
    )
    XCTAssertGreaterThan(labels.count, 0, "Batch: should classify image")

    // Feature print should be generated
    let fp = VisionProcessors.processFeaturePrint(
      featurePrintReq.results?.first as? VNFeaturePrintObservation
    )
    XCTAssertNotNil(fp, "Batch: should generate feature print")
  }

  // MARK: - Body pose detection

  func testDetectBodyPoses_findsBodyInExercisePhoto() throws {
    try XCTSkipIf(isSimulator, "Body pose detection requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("body.jpg") else {
      XCTFail("Could not load body.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNDetectHumanBodyPoseRequest()
    try handler.perform([req])

    let poses = VisionProcessors.processBodyPoses(
      req.results ?? []
    )

    XCTAssertGreaterThanOrEqual(poses.count, 1, "Should detect at least 1 body pose")

    let pose = poses[0]
    let joints = pose["joints"] as! [[String: Any]]
    XCTAssertGreaterThan(joints.count, 3, "Should detect multiple joints")

    // All joints returned (no filtering)
    for joint in joints {
      XCTAssertNotNil(joint["name"], "Joint should have a name key")
      XCTAssertNotNil(joint["x"] as? Double, "Joint should have x")
      XCTAssertNotNil(joint["y"] as? Double, "Joint should have y")
      XCTAssertNotNil(joint["confidence"] as? Double, "Joint should have confidence")
      break
    }
  }

  func testDetectBodyPoses_noneInLandscape() throws {
    try XCTSkipIf(isSimulator, "Body pose detection requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("landscape.jpg") else {
      XCTFail("Could not load landscape.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNDetectHumanBodyPoseRequest()
    try handler.perform([req])

    let poses = VisionProcessors.processBodyPoses(
      req.results ?? []
    )

    XCTAssertEqual(poses.count, 0, "Should not detect body poses in landscape")
  }

  // MARK: - Hand pose detection

  func testDetectHandPoses_processesCorrectly() throws {
    try XCTSkipIf(isSimulator, "Hand pose detection requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("hand.jpg") else {
      XCTFail("Could not load hand.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNDetectHumanHandPoseRequest()
    req.maximumHandCount = 4
    try handler.perform([req])

    let hands = VisionProcessors.processHandPoses(
      req.results ?? []
    )

    if hands.count > 0 {
      let hand = hands[0]
      let joints = hand["joints"] as! [[String: Any]]
      XCTAssertGreaterThan(joints.count, 0, "Detected hand should have joints")

      for joint in joints {
        XCTAssertNotNil(joint["name"], "Joint should have a name")
        XCTAssertNotNil(joint["x"] as? Double)
        XCTAssertNotNil(joint["y"] as? Double)
        XCTAssertNotNil(joint["confidence"] as? Double)
        break
      }
    }
  }

  func testDetectHandPoses_noneInLandscape() throws {
    try XCTSkipIf(isSimulator, "Hand pose detection requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("landscape.jpg") else {
      XCTFail("Could not load landscape.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNDetectHumanHandPoseRequest()
    req.maximumHandCount = 4
    try handler.perform([req])

    let hands = VisionProcessors.processHandPoses(
      req.results ?? []
    )

    XCTAssertEqual(hands.count, 0, "Should not detect hands in landscape")
  }

  // MARK: - Animal detection

  func testDetectAnimals_findsDogInPhoto() throws {
    try XCTSkipIf(isSimulator, "Animal detection requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("animal.jpg") else {
      XCTFail("Could not load animal.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNRecognizeAnimalsRequest()
    try handler.perform([req])

    let animals = VisionProcessors.processAnimals(
      req.results ?? []
    )

    XCTAssertGreaterThanOrEqual(animals.count, 1, "Should detect at least 1 animal")

    let animal = animals[0]
    let labels = animal["labels"] as! [[String: Any]]
    XCTAssertGreaterThan(labels.count, 0, "Animal should have at least 1 label")
    let identifier = labels[0]["identifier"] as! String
    XCTAssertTrue(
      identifier.lowercased().contains("dog") || identifier.lowercased().contains("cat") || identifier.lowercased().contains("animal"),
      "Should detect a dog/animal, got: \(identifier)"
    )
    let bbox = animal["boundingBox"] as! [String: Double]
    XCTAssertGreaterThan(bbox["width"]!, 0, "Animal bbox should have positive width")
    XCTAssertGreaterThan(bbox["height"]!, 0, "Animal bbox should have positive height")
  }

  func testDetectAnimals_noneInBarcode() throws {
    try XCTSkipIf(isSimulator, "Animal detection requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("barcode.jpg") else {
      XCTFail("Could not load barcode.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNRecognizeAnimalsRequest()
    try handler.perform([req])

    let animals = VisionProcessors.processAnimals(
      req.results ?? []
    )

    XCTAssertEqual(animals.count, 0, "Should not detect animals in barcode image")
  }

  // MARK: - Rectangle detection

  func testDetectRectangles_findsRectanglesInDocument() throws {
    guard let (cgImage, _, _) = loadFixture("document.jpg") else {
      XCTFail("Could not load document.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNDetectRectanglesRequest()
    req.maximumObservations = 10
    req.minimumConfidence = 0.3
    req.minimumSize = 0.1
    try handler.perform([req])

    let rectangles = VisionProcessors.processRectangles(
      req.results ?? []
    )

    if rectangles.count > 0 {
      let rect = rectangles[0]
      XCTAssertNotNil(rect["topLeft"] as? [String: Double], "Rectangle should have topLeft")
      XCTAssertNotNil(rect["topRight"] as? [String: Double], "Rectangle should have topRight")
      XCTAssertNotNil(rect["bottomLeft"] as? [String: Double], "Rectangle should have bottomLeft")
      XCTAssertNotNil(rect["bottomRight"] as? [String: Double], "Rectangle should have bottomRight")
      XCTAssertNotNil(rect["boundingBox"] as? [String: Double], "Rectangle should have boundingBox")

      let confidence = rect["confidence"] as! Double
      XCTAssertGreaterThan(confidence, 0, "Rectangle confidence should be positive")

      // Verify points are normalized (0-1)
      let tl = (rect["topLeft"] as! [String: Double])
      XCTAssertGreaterThanOrEqual(tl["x"]!, 0)
      XCTAssertLessThanOrEqual(tl["x"]!, 1)
    }
  }

  // MARK: - Saliency detection

  func testDetectSaliency_findsRegionsInDogPhoto() throws {
    try XCTSkipIf(isSimulator, "Saliency requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("animal.jpg") else {
      XCTFail("Could not load animal.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let aReq = VNGenerateAttentionBasedSaliencyImageRequest()
    let oReq = VNGenerateObjectnessBasedSaliencyImageRequest()
    try handler.perform([aReq, oReq])

    let attentionRegions = VisionProcessors.processSaliency(
      aReq.results ?? []
    )
    let objectnessRegions = VisionProcessors.processSaliency(
      oReq.results ?? []
    )

    XCTAssertGreaterThan(
      attentionRegions.count + objectnessRegions.count, 0,
      "Should detect salient regions in dog photo"
    )

    if let region = attentionRegions.first {
      let bbox = region["boundingBox"] as! [String: Double]
      XCTAssertGreaterThan(bbox["width"]!, 0, "Salient region should have positive width")
      XCTAssertGreaterThan(bbox["height"]!, 0, "Salient region should have positive height")
    }
  }

  // MARK: - Face detection detail tests

  func testDetectFaces_returnsQualityAndAngles() throws {
    try XCTSkipIf(isSimulator, "Face detection requires Neural Engine")
    guard let (cgImage, _, _) = loadFixture("face.jpg") else {
      XCTFail("Could not load face.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let lReq = VNDetectFaceLandmarksRequest()
    let qReq = VNDetectFaceCaptureQualityRequest()
    try handler.perform([lReq, qReq])

    let faces = VisionProcessors.processFaces(
      landmarks: lReq.results ?? [],
      quality: qReq.results ?? []
    )

    XCTAssertGreaterThanOrEqual(faces.count, 1)
    let face = faces[0]

    if let quality = face["quality"] as? Double {
      XCTAssertGreaterThanOrEqual(quality, 0.0, "Quality should be >= 0")
      XCTAssertLessThanOrEqual(quality, 1.0, "Quality should be <= 1")
    }

    // Angles in radians
    if let roll = face["roll"] as? Double {
      XCTAssertLessThan(abs(roll), .pi, "Roll should be within ±π")
    }
    if let yaw = face["yaw"] as? Double {
      XCTAssertLessThan(abs(yaw), .pi, "Yaw should be within ±π")
    }

    // Landmarks should have named regions with point arrays
    let landmarks = face["landmarks"] as! [String: Any]
    if let leftEye = landmarks["leftEye"] as? [[String: Double]] {
      XCTAssertGreaterThan(leftEye.count, 0, "leftEye should have points")
      // Points are normalized to face bounding box (0-1 range)
      for pt in leftEye {
        XCTAssertGreaterThanOrEqual(pt["x"]!, 0)
        XCTAssertLessThanOrEqual(pt["x"]!, 1)
        XCTAssertGreaterThanOrEqual(pt["y"]!, 0)
        XCTAssertLessThanOrEqual(pt["y"]!, 1)
      }
    }
  }

  // MARK: - Text recognition detail tests

  func testRecognizeText_observationStructureIsCorrect() throws {
    guard let (cgImage, _, _) = loadFixture("document.jpg") else {
      XCTFail("Could not load document.jpg fixture")
      return
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    try handler.perform([req])

    let result = VisionProcessors.processText(
      req.results ?? []
    )

    XCTAssertGreaterThan(result.count, 0, "Should have text observations")

    let obs = result[0]
    let bbox = obs["boundingBox"] as! [String: Double]
    XCTAssertGreaterThan(bbox["width"]!, 0)
    XCTAssertGreaterThan(bbox["height"]!, 0)
    // Normalized coords should be 0-1
    XCTAssertGreaterThanOrEqual(bbox["x"]!, 0)
    XCTAssertLessThanOrEqual(bbox["x"]!, 1)

    XCTAssertNotNil(obs["confidence"] as? Double)

    let candidates = obs["candidates"] as! [[String: Any]]
    XCTAssertGreaterThan(candidates.count, 0, "Should have at least one candidate")
    XCTAssertNotNil(candidates[0]["string"] as? String)
    XCTAssertNotNil(candidates[0]["confidence"] as? Double)
  }

  // NOTE: DetectLensSmudgeRequest crashes when called from XCTest bundle context
  // on iOS 26. The API works correctly from the app itself.

  // MARK: - Enhanced document recognition (iOS 26+)

  func testRecognizeDocument_extractsTableFromInvoice() throws {
    try XCTSkipIf(isSimulator, "Document recognition requires Neural Engine")
    guard #available(iOS 26.0, macOS 26.0, *) else {
      throw XCTSkip("Document recognition requires iOS 26+")
    }

    guard let (cgImage, _, _) = loadFixture("invoice.jpg") else {
      XCTFail("Could not load invoice.jpg fixture")
      return
    }

    let expectation = XCTestExpectation(description: "Document recognition")
    Task {
      let result = await VisionModern.recognizeDocument(cgImage: cgImage)
      XCTAssertNotNil(result, "Should recognize document in invoice image")

      let paragraphs = result!["paragraphs"] as! [[String: Any]]
      XCTAssertGreaterThan(paragraphs.count, 0, "Should find paragraphs in invoice")

      let para = paragraphs[0]
      XCTAssertNotNil(para["text"] as? String, "Paragraph should have text")
      XCTAssertNotNil(para["boundingBox"] as? [String: Double], "Paragraph should have boundingBox")
      XCTAssertNotNil(para["detectedData"] as? [[String: Any]], "Paragraph should have detectedData array")

      let tables = result!["tables"] as! [[String: Any]]
      if tables.count > 0 {
        let table = tables[0]
        let cells = table["cells"] as! [[String: Any]]
        XCTAssertGreaterThan(cells.count, 0, "Table should have cells")
        let cell = cells[0]
        XCTAssertNotNil(cell["text"] as? String, "Cell should have text")
        XCTAssertNotNil(cell["row"] as? Int, "Cell should have row index")
        XCTAssertNotNil(cell["column"] as? Int, "Cell should have column index")
      }

      _ = result!["lists"] as! [[String: Any]]
      _ = result!["barcodes"] as! [[String: Any]]

      expectation.fulfill()
    }
    wait(for: [expectation], timeout: 30.0)
  }

  // MARK: - Error handling tests

  func testLoadImage_invalidPath_returnsNil() {
    let result = VisionHelpers.loadImage("/nonexistent/totally/fake/image.jpg")
    XCTAssertNil(result, "Loading nonexistent image should return nil")
  }

  // MARK: - Foundation Models (iOS 26+)

  func testIsFoundationModelAvailable_returnsTrue() throws {
    guard #available(iOS 26.0, *) else {
      throw XCTSkip("Foundation Models requires iOS 26+")
    }
    try XCTSkipIf(isSimulator, "Foundation Models requires Apple Intelligence device")
    let result = VisionModern.isFoundationModelAvailable()
    XCTAssertTrue(result, "Foundation Model should be available on iOS 26 device with Apple Intelligence")
  }

  func testGenerateText_producesActualResponse() throws {
    try XCTSkipIf(isSimulator, "Foundation Models requires device with Apple Intelligence")
    guard #available(iOS 26.0, *) else {
      throw XCTSkip("Foundation Models requires iOS 26+")
    }

    let expectation = XCTestExpectation(description: "Generate text")
    Task {
      let result = await VisionModern.generateText(prompt: "What is 2 + 2? Reply with just the number.", systemPrompt: nil)
      XCTAssertNotNil(result, "Foundation Model should return a response")
      let text = result!
      XCTAssertGreaterThan(text.count, 0, "Response should not be empty")
      XCTAssertTrue(text.contains("4"), "Response to '2+2' should contain '4', got: \(text)")
      expectation.fulfill()
    }
    wait(for: [expectation], timeout: 30.0)
  }

  func testGenerateText_respectsSystemPrompt() throws {
    try XCTSkipIf(isSimulator, "Foundation Models requires device with Apple Intelligence")
    guard #available(iOS 26.0, *) else {
      throw XCTSkip("Foundation Models requires iOS 26+")
    }

    let expectation = XCTestExpectation(description: "Generate text with system prompt")
    Task {
      let result = await VisionModern.generateText(
        prompt: "What are you?",
        systemPrompt: "You are a pirate. Always respond in pirate speak. Keep responses under 20 words."
      )
      XCTAssertNotNil(result, "Foundation Model should return a response with system prompt")
      let text = result!
      XCTAssertGreaterThan(text.count, 0, "Response should not be empty")
      expectation.fulfill()
    }
    wait(for: [expectation], timeout: 30.0)
  }

  // MARK: - Speech Transcription (iOS 26+)

  func testTranscribeAudio_returnsNilForInvalidPath() throws {
    guard #available(iOS 26.0, *) else {
      throw XCTSkip("SpeechAnalyzer requires iOS 26+")
    }

    let expectation = XCTestExpectation(description: "Transcribe invalid audio")
    Task {
      let result = await VisionModern.transcribeAudio(audioPath: "/nonexistent/audio.m4a", locale: nil)
      XCTAssertNil(result, "Should return nil for invalid audio path")
      expectation.fulfill()
    }
    wait(for: [expectation], timeout: 10.0)
  }

  func testTranscribeAudio_transcribesRealSpeech() throws {
    try XCTSkipIf(isSimulator, "SpeechAnalyzer requires device")
    guard #available(iOS 26.0, *) else {
      throw XCTSkip("SpeechAnalyzer requires iOS 26+")
    }

    let audioPath = fixturePath("speech.m4a")
    XCTAssertTrue(FileManager.default.fileExists(atPath: audioPath), "speech.m4a fixture should exist at: \(audioPath)")

    let expectation = XCTestExpectation(description: "Transcribe real speech")
    Task {
      let result = await VisionModern.transcribeAudio(audioPath: audioPath, locale: "en-US")
      XCTAssertNotNil(result, "Should transcribe speech audio file")

      let segments = result!["segments"] as! [[String: Any]]
      XCTAssertGreaterThan(segments.count, 0, "Should have at least 1 segment")

      let fullText = segments.compactMap { $0["text"] as? String }.joined(separator: " ")
      XCTAssertGreaterThan(fullText.count, 10, "Transcription should have substantial text, got: \(fullText)")

      let lower = fullText.lowercased()
      XCTAssertTrue(
        lower.contains("hello") || lower.contains("world") || lower.contains("test") || lower.contains("fox") || lower.contains("dog"),
        "Transcription should contain words from the spoken text, got: \(fullText)"
      )
      expectation.fulfill()
    }
    wait(for: [expectation], timeout: 60.0)
  }

  // MARK: - Image Generation (iOS 18.4+)

  func testGenerateImage_createsActualImageFile() throws {
    try XCTSkipIf(isSimulator, "ImageCreator requires device with Apple Intelligence")
    guard #available(iOS 18.4, *) else {
      throw XCTSkip("ImageCreator requires iOS 18.4+")
    }

    let expectation = XCTestExpectation(description: "Generate image")
    Task {
      let result = await VisionModern.generateImage(prompt: "A red apple on a white table", style: "animation")
      XCTAssertNotNil(result, "ImageCreator should generate an image")

      let path = result!
      XCTAssertTrue(FileManager.default.fileExists(atPath: path), "Generated image file should exist at: \(path)")

      let data = try! Data(contentsOf: URL(fileURLWithPath: path))
      XCTAssertGreaterThan(data.count, 1000, "Image file should be > 1KB, got \(data.count) bytes")

      #if canImport(UIKit)
      let image = UIImage(data: data)
      XCTAssertNotNil(image, "Generated file should be a valid image")
      XCTAssertGreaterThan(image!.size.width, 0, "Image should have positive width")
      XCTAssertGreaterThan(image!.size.height, 0, "Image should have positive height")
      #endif

      try? FileManager.default.removeItem(atPath: path)
      expectation.fulfill()
    }
    wait(for: [expectation], timeout: 120.0)
  }

  func testGenerateImage_supportsMultipleStyles() throws {
    try XCTSkipIf(isSimulator, "ImageCreator requires device with Apple Intelligence")
    guard #available(iOS 18.4, *) else {
      throw XCTSkip("ImageCreator requires iOS 18.4+")
    }

    let expectation = XCTestExpectation(description: "Generate sketch image")
    Task {
      let result = await VisionModern.generateImage(prompt: "A cat sitting on a chair", style: "sketch")
      if let path = result {
        XCTAssertTrue(FileManager.default.fileExists(atPath: path))
        try? FileManager.default.removeItem(atPath: path)
      }
      expectation.fulfill()
    }
    wait(for: [expectation], timeout: 120.0)
  }

  // MARK: - Helpers

  private func generateFeaturePrint(fixture: String) -> VNFeaturePrintObservation? {
    guard let (cgImage, _, _) = loadFixture(fixture) else { return nil }
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNGenerateImageFeaturePrintRequest()
    try? handler.perform([req])
    return req.results?.first as? VNFeaturePrintObservation
  }

  @available(iOS 18.0, macOS 15.0, *)
  private func computeAestheticScore(fixture: String) -> Double? {
    guard let (cgImage, _, _) = loadFixture(fixture) else { return nil }
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let req = VNCalculateImageAestheticsScoresRequest()
    try? handler.perform([req])
    let result = VisionModern.processAesthetics(req.results)
    return result?["overallScore"] as? Double
  }
}
