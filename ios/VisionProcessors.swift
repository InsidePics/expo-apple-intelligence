import Vision

/// Raw passthrough processing for VNRequest results.
/// Returns Apple API values directly — no filtering, no coordinate conversion.
/// All coordinates are normalized (0-1, bottom-left origin) as Apple returns them.
/// Face landmark points are normalized to the face bounding box.
enum VisionProcessors {

  // MARK: - Faces

  static func processFaces(
    landmarks landmarkObs: [VNFaceObservation],
    quality qualityObs: [VNFaceObservation]
  ) -> [[String: Any]] {
    return landmarkObs.map { face in
      var dict: [String: Any] = [
        "boundingBox": VisionHelpers.rawRect(face.boundingBox),
        "confidence": Double(face.confidence),
      ]

      if let roll = face.roll { dict["roll"] = roll.doubleValue }
      if let yaw = face.yaw { dict["yaw"] = yaw.doubleValue }
      if let pitch = face.pitch { dict["pitch"] = pitch.doubleValue }

      if let lm = face.landmarks {
        var landmarks: [String: Any] = [:]
        if let r = lm.faceContour { landmarks["faceContour"] = rawLandmarkPoints(r) }
        if let r = lm.leftEye { landmarks["leftEye"] = rawLandmarkPoints(r) }
        if let r = lm.rightEye { landmarks["rightEye"] = rawLandmarkPoints(r) }
        if let r = lm.leftEyebrow { landmarks["leftEyebrow"] = rawLandmarkPoints(r) }
        if let r = lm.rightEyebrow { landmarks["rightEyebrow"] = rawLandmarkPoints(r) }
        if let r = lm.nose { landmarks["nose"] = rawLandmarkPoints(r) }
        if let r = lm.noseCrest { landmarks["noseCrest"] = rawLandmarkPoints(r) }
        if let r = lm.medianLine { landmarks["medianLine"] = rawLandmarkPoints(r) }
        if let r = lm.outerLips { landmarks["outerLips"] = rawLandmarkPoints(r) }
        if let r = lm.innerLips { landmarks["innerLips"] = rawLandmarkPoints(r) }
        if let r = lm.leftPupil { landmarks["leftPupil"] = rawLandmarkPoints(r) }
        if let r = lm.rightPupil { landmarks["rightPupil"] = rawLandmarkPoints(r) }
        dict["landmarks"] = landmarks
      }

      let qualityMatch = qualityObs.first { $0.uuid == face.uuid }
      if let q = qualityMatch?.faceCaptureQuality {
        dict["quality"] = Double(q)
      }

      return dict
    }
  }

  private static func rawLandmarkPoints(_ region: VNFaceLandmarkRegion2D) -> [[String: Double]] {
    (0..<region.pointCount).map { i in
      let pt = region.normalizedPoints[i]
      return ["x": Double(pt.x), "y": Double(pt.y)]
    }
  }

  // MARK: - Text

  static func processText(_ observations: [VNRecognizedTextObservation]) -> [[String: Any]] {
    return observations.compactMap { obs in
      let candidates: [[String: Any]] = obs.topCandidates(10).map { candidate in
        ["string": candidate.string, "confidence": Double(candidate.confidence)]
      }
      if candidates.isEmpty { return nil }
      return [
        "boundingBox": VisionHelpers.rawRect(obs.boundingBox),
        "confidence": Double(obs.confidence),
        "candidates": candidates,
      ] as [String: Any]
    }
  }

  // MARK: - Classification

  static func processClassification(_ observations: [VNClassificationObservation]) -> [[String: Any]] {
    return observations.map {
      ["identifier": $0.identifier, "confidence": Double($0.confidence)]
    }
  }

  // MARK: - Barcodes

  static func processBarcodes(_ observations: [VNBarcodeObservation]) -> [[String: Any]] {
    return observations.map { obs in
      var dict: [String: Any] = [
        "boundingBox": VisionHelpers.rawRect(obs.boundingBox),
        "confidence": Double(obs.confidence),
        "symbology": obs.symbology.rawValue,
        "topLeft": VisionHelpers.rawPoint(obs.topLeft),
        "topRight": VisionHelpers.rawPoint(obs.topRight),
        "bottomLeft": VisionHelpers.rawPoint(obs.bottomLeft),
        "bottomRight": VisionHelpers.rawPoint(obs.bottomRight),
      ]
      if let payload = obs.payloadStringValue {
        dict["payloadStringValue"] = payload
      }
      return dict
    }
  }

  // MARK: - Body pose

  static func processBodyPoses(_ observations: [VNHumanBodyPoseObservation]) -> [[String: Any]] {
    return observations.compactMap { obs in
      guard let points = try? obs.recognizedPoints(.all) else { return nil }
      let joints: [[String: Any]] = points.map { (key, point) in
        ["name": key.rawValue, "x": Double(point.location.x), "y": Double(point.location.y), "confidence": Double(point.confidence)]
      }
      return ["joints": joints]
    }
  }

  // MARK: - Hand pose

  static func processHandPoses(_ observations: [VNHumanHandPoseObservation]) -> [[String: Any]] {
    return observations.compactMap { obs in
      guard let points = try? obs.recognizedPoints(.all) else { return nil }
      let joints: [[String: Any]] = points.map { (key, point) in
        ["name": key.rawValue, "x": Double(point.location.x), "y": Double(point.location.y), "confidence": Double(point.confidence)]
      }
      return ["joints": joints]
    }
  }

  // MARK: - Feature print

  static func processFeaturePrint(_ obs: VNFeaturePrintObservation?) -> [String: Any]? {
    guard let obs = obs else { return nil }
    var data: [Double] = []
    let count = obs.elementCount
    if obs.elementType == .float {
      var floats = [Float](repeating: 0, count: count)
      _ = floats.withUnsafeMutableBytes { ptr in
        obs.data.copyBytes(to: ptr)
      }
      data = floats.map { Double($0) }
    } else {
      var doubles = [Double](repeating: 0, count: count)
      _ = doubles.withUnsafeMutableBytes { ptr in
        obs.data.copyBytes(to: ptr)
      }
      data = doubles
    }
    return ["data": data as [Any], "elementType": obs.elementType == .float ? "float" : "double", "elementCount": Double(count)]
  }

  // MARK: - Saliency

  static func processSaliency(_ observations: [VNSaliencyImageObservation]) -> [[String: Any]] {
    var regions: [[String: Any]] = []
    for obs in observations {
      guard let salientObjects = obs.salientObjects else { continue }
      for obj in salientObjects {
        regions.append(["boundingBox": VisionHelpers.rawRect(obj.boundingBox), "confidence": Double(obj.confidence)])
      }
    }
    return regions
  }

  // MARK: - Animals

  static func processAnimals(_ observations: [VNRecognizedObjectObservation]) -> [[String: Any]] {
    return observations.map { obs in
      let labels: [[String: Any]] = obs.labels.map {
        ["identifier": $0.identifier, "confidence": Double($0.confidence)]
      }
      return [
        "boundingBox": VisionHelpers.rawRect(obs.boundingBox),
        "confidence": Double(obs.confidence),
        "labels": labels,
      ]
    }
  }

  // MARK: - Rectangles

  static func processRectangles(_ observations: [VNRectangleObservation]) -> [[String: Any]] {
    return observations.map { obs in
      [
        "boundingBox": VisionHelpers.rawRect(obs.boundingBox),
        "topLeft": VisionHelpers.rawPoint(obs.topLeft),
        "topRight": VisionHelpers.rawPoint(obs.topRight),
        "bottomLeft": VisionHelpers.rawPoint(obs.bottomLeft),
        "bottomRight": VisionHelpers.rawPoint(obs.bottomRight),
        "confidence": Double(obs.confidence),
      ]
    }
  }

  // MARK: - Horizon

  static func processHorizon(_ obs: VNHorizonObservation?) -> [String: Any]? {
    guard let obs = obs else { return nil }
    return ["angle": Double(obs.angle)]
  }
}
