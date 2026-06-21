import ExpoModulesCore
import Vision

#if canImport(UIKit)
import UIKit
#endif

public class ExpoAppleIntelligenceModule: Module {
  public func definition() -> ModuleDefinition {
    Name("ExpoAppleIntelligence")

    // MARK: - Batch analysis

    AsyncFunction("analyzeImage") { (imagePath: String, promise: Promise) in
      guard let (cgImage, width, height) = VisionHelpers.loadImage(imagePath) else {
        promise.reject("ERR_LOAD_IMAGE", "Could not load image at \(imagePath)")
        return
      }

      DispatchQueue.global(qos: .userInitiated).async {
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

        let faceLandmarksReq = VNDetectFaceLandmarksRequest()
        let faceQualityReq = VNDetectFaceCaptureQualityRequest()
        let textReq = VNRecognizeTextRequest()
        textReq.recognitionLevel = .accurate
        let classifyReq = VNClassifyImageRequest()
        let barcodeReq = VNDetectBarcodesRequest()
        let bodyPoseReq = VNDetectHumanBodyPoseRequest()
        let handPoseReq = VNDetectHumanHandPoseRequest()
        handPoseReq.maximumHandCount = 4
        let featurePrintReq = VNGenerateImageFeaturePrintRequest()
        let attentionReq = VNGenerateAttentionBasedSaliencyImageRequest()
        let objectnessReq = VNGenerateObjectnessBasedSaliencyImageRequest()
        let animalReq = VNRecognizeAnimalsRequest()
        let rectangleReq = VNDetectRectanglesRequest()
        rectangleReq.maximumObservations = 10
        let horizonReq = VNDetectHorizonRequest()

        var requests: [VNRequest] = [
          faceLandmarksReq, faceQualityReq, textReq, classifyReq, barcodeReq,
          bodyPoseReq, handPoseReq, featurePrintReq, attentionReq, objectnessReq,
          animalReq, rectangleReq, horizonReq,
        ]

        var aestheticsReq: VNRequest? = nil
        if #available(iOS 18.0, macOS 15.0, *) {
          let req = VNCalculateImageAestheticsScoresRequest()
          aestheticsReq = req
          requests.append(req)
        }

        do { try handler.perform(requests) } catch {
          promise.reject("ERR_ANALYSIS", error.localizedDescription)
          return
        }

        let faces = VisionProcessors.processFaces(landmarks: faceLandmarksReq.results ?? [], quality: faceQualityReq.results ?? [])
        let text = VisionProcessors.processText(textReq.results ?? [])
        let labels = VisionProcessors.processClassification(classifyReq.results ?? [])
        let barcodes = VisionProcessors.processBarcodes(barcodeReq.results ?? [])
        let bodyPoses = VisionProcessors.processBodyPoses(bodyPoseReq.results ?? [])
        let handPoses = VisionProcessors.processHandPoses(handPoseReq.results ?? [])
        let featurePrint = VisionProcessors.processFeaturePrint(featurePrintReq.results?.first)
        let attentionRegions = VisionProcessors.processSaliency(attentionReq.results ?? [])
        let objectnessRegions = VisionProcessors.processSaliency(objectnessReq.results ?? [])
        let animals = VisionProcessors.processAnimals(animalReq.results ?? [])
        let rectangles = VisionProcessors.processRectangles(rectangleReq.results ?? [])
        let horizon = VisionProcessors.processHorizon(horizonReq.results?.first)

        var aesthetics: [String: Any]? = nil
        if #available(iOS 18.0, macOS 15.0, *) {
          aesthetics = VisionModern.processAesthetics(aestheticsReq?.results)
        }

        var result: [String: Any] = [
          "faces": faces, "text": text, "labels": labels, "barcodes": barcodes,
          "bodyPoses": bodyPoses, "handPoses": handPoses, "rectangles": rectangles,
          "animals": animals,
          "featurePrint": featurePrint ?? [:] as [String: Any],
          "aesthetics": aesthetics ?? [:] as [String: Any],
          "saliency": ["attentionRegions": attentionRegions, "objectnessRegions": objectnessRegions],
          "horizon": horizon ?? [:] as [String: Any],
          "lensSmudge": [:] as [String: Any],
          "document": [:] as [String: Any],
          "imageWidth": Double(width), "imageHeight": Double(height),
        ]
        #if os(iOS)
        result["platform"] = "ios"
        #else
        result["platform"] = "macos"
        #endif

        #if os(iOS)
        if #available(iOS 26.0, *) {
          Task {
            let lensSmudge = await VisionModern.detectLensSmudge(cgImage: cgImage)
            let document = await VisionModern.recognizeDocument(cgImage: cgImage)
            var final26 = result
            if let ls = lensSmudge { final26["lensSmudge"] = ls }
            if let doc = document { final26["document"] = doc }
            promise.resolve(final26)
          }
        } else {
          promise.resolve(result)
        }
        #elseif os(macOS)
        if #available(macOS 26.0, *) {
          Task {
            let lensSmudge = await VisionModern.detectLensSmudge(cgImage: cgImage)
            let document = await VisionModern.recognizeDocument(cgImage: cgImage)
            var final26 = result
            if let ls = lensSmudge { final26["lensSmudge"] = ls }
            if let doc = document { final26["document"] = doc }
            promise.resolve(final26)
          }
        } else {
          promise.resolve(result)
        }
        #endif
      }
    }

    // MARK: - Individual functions

    AsyncFunction("detectFaces") { (imagePath: String, promise: Promise) in
      Self.runVision(imagePath, promise: promise) { handler in
        let lReq = VNDetectFaceLandmarksRequest()
        let qReq = VNDetectFaceCaptureQualityRequest()
        try handler.perform([lReq, qReq])
        return VisionProcessors.processFaces(landmarks: lReq.results ?? [], quality: qReq.results ?? [])
      }
    }

    AsyncFunction("recognizeText") { (imagePath: String, promise: Promise) in
      Self.runVision(imagePath, promise: promise) { handler in
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        try handler.perform([req])
        return VisionProcessors.processText(req.results ?? [])
      }
    }

    AsyncFunction("classifyImage") { (imagePath: String, promise: Promise) in
      Self.runVision(imagePath, promise: promise) { handler in
        let req = VNClassifyImageRequest()
        try handler.perform([req])
        return VisionProcessors.processClassification(req.results ?? [])
      }
    }

    AsyncFunction("detectBarcodes") { (imagePath: String, promise: Promise) in
      Self.runVision(imagePath, promise: promise) { handler in
        let req = VNDetectBarcodesRequest()
        try handler.perform([req])
        return VisionProcessors.processBarcodes(req.results ?? [])
      }
    }

    AsyncFunction("detectBodyPoses") { (imagePath: String, promise: Promise) in
      Self.runVision(imagePath, promise: promise) { handler in
        let req = VNDetectHumanBodyPoseRequest()
        try handler.perform([req])
        return VisionProcessors.processBodyPoses(req.results ?? [])
      }
    }

    AsyncFunction("detectHandPoses") { (imagePath: String, promise: Promise) in
      Self.runVision(imagePath, promise: promise) { handler in
        let req = VNDetectHumanHandPoseRequest()
        req.maximumHandCount = 4
        try handler.perform([req])
        return VisionProcessors.processHandPoses(req.results ?? [])
      }
    }

    AsyncFunction("generateFeaturePrint") { (imagePath: String, promise: Promise) in
      Self.runVision(imagePath, promise: promise) { handler in
        let req = VNGenerateImageFeaturePrintRequest()
        try handler.perform([req])
        return VisionProcessors.processFeaturePrint(req.results?.first) as Any
      }
    }

    AsyncFunction("calculateAesthetics") { (imagePath: String, promise: Promise) in
      guard #available(iOS 18.0, macOS 15.0, *) else {
        promise.resolve(nil)
        return
      }
      Self.runVision(imagePath, promise: promise) { handler in
        let req = VNCalculateImageAestheticsScoresRequest()
        try handler.perform([req])
        return VisionModern.processAesthetics(req.results) as Any
      }
    }

    AsyncFunction("detectSaliency") { (imagePath: String, promise: Promise) in
      Self.runVision(imagePath, promise: promise) { handler in
        let aReq = VNGenerateAttentionBasedSaliencyImageRequest()
        let oReq = VNGenerateObjectnessBasedSaliencyImageRequest()
        try handler.perform([aReq, oReq])
        return [
          "attentionRegions": VisionProcessors.processSaliency(aReq.results ?? []),
          "objectnessRegions": VisionProcessors.processSaliency(oReq.results ?? []),
        ]
      }
    }

    AsyncFunction("detectAnimals") { (imagePath: String, promise: Promise) in
      Self.runVision(imagePath, promise: promise) { handler in
        let req = VNRecognizeAnimalsRequest()
        try handler.perform([req])
        return VisionProcessors.processAnimals(req.results ?? [])
      }
    }

    AsyncFunction("detectRectangles") { (imagePath: String, promise: Promise) in
      Self.runVision(imagePath, promise: promise) { handler in
        let req = VNDetectRectanglesRequest()
        req.maximumObservations = 10
        try handler.perform([req])
        return VisionProcessors.processRectangles(req.results ?? [])
      }
    }

    AsyncFunction("detectHorizon") { (imagePath: String, promise: Promise) in
      Self.runVision(imagePath, promise: promise) { handler in
        let req = VNDetectHorizonRequest()
        try handler.perform([req])
        return VisionProcessors.processHorizon(req.results?.first) as Any
      }
    }

    AsyncFunction("detectLensSmudge") { (imagePath: String, promise: Promise) in
      if #available(iOS 26.0, macOS 26.0, *) {
        guard let (cgImage, _, _) = VisionHelpers.loadImage(imagePath) else {
          promise.reject("ERR_LOAD_IMAGE", "Could not load image at \(imagePath)")
          return
        }
        Task {
          let result = await VisionModern.detectLensSmudge(cgImage: cgImage)
          promise.resolve(result)
        }
      } else {
        promise.resolve(nil)
      }
    }

    AsyncFunction("recognizeDocument") { (imagePath: String, promise: Promise) in
      if #available(iOS 26.0, macOS 26.0, *) {
        guard let (cgImage, _, _) = VisionHelpers.loadImage(imagePath) else {
          promise.reject("ERR_LOAD_IMAGE", "Could not load image at \(imagePath)")
          return
        }
        Task {
          let result = await VisionModern.recognizeDocument(cgImage: cgImage)
          promise.resolve(result)
        }
      } else {
        promise.resolve(nil)
      }
    }

    // MARK: - Image decoding

    AsyncFunction("decodeImagePixels") { (imagePath: String, promise: Promise) in
      guard let (cgImage, _, _) = VisionHelpers.loadImage(imagePath) else {
        promise.reject("ERR_LOAD_IMAGE", "Could not load image at \(imagePath)")
        return
      }
      DispatchQueue.global(qos: .userInitiated).async {
        guard let data = VisionHelpers.decodePixels(cgImage) else {
          promise.reject("ERR_DECODE", "Failed to decode pixels")
          return
        }
        promise.resolve([
          "pixels": data.base64EncodedString(),
          "width": Double(cgImage.width),
          "height": Double(cgImage.height),
        ])
      }
    }

    // MARK: - Foundation Models (iOS 26+)

    Function("isFoundationModelAvailable") { () -> Bool in
      if #available(iOS 26.0, *) {
        return VisionModern.isFoundationModelAvailable()
      }
      return false
    }

    AsyncFunction("generateText") { (prompt: String, systemPrompt: String?, promise: Promise) in
      if #available(iOS 26.0, *) {
        Task {
          let result = await VisionModern.generateText(prompt: prompt, systemPrompt: systemPrompt)
          promise.resolve(result)
        }
      } else {
        promise.resolve(nil)
      }
    }

    // MARK: - Speech Transcription (iOS 26+)

    AsyncFunction("transcribeAudio") { (audioPath: String, locale: String?, promise: Promise) in
      if #available(iOS 26.0, *) {
        Task {
          let result = await VisionModern.transcribeAudio(audioPath: audioPath, locale: locale)
          promise.resolve(result)
        }
      } else {
        promise.resolve(nil)
      }
    }

    // MARK: - Image Generation (iOS 18.4+)

    AsyncFunction("generateImage") { (prompt: String, style: String?, promise: Promise) in
      if #available(iOS 18.4, *) {
        Task {
          let result = await VisionModern.generateImage(prompt: prompt, style: style)
          promise.resolve(result)
        }
      } else {
        promise.resolve(nil)
      }
    }
  }

  // MARK: - Helper to reduce boilerplate

  private static func runVision(
    _ imagePath: String,
    promise: Promise,
    work: @escaping (VNImageRequestHandler) throws -> Any
  ) {
    guard let (cgImage, _, _) = VisionHelpers.loadImage(imagePath) else {
      promise.reject("ERR_LOAD_IMAGE", "Could not load image at \(imagePath)")
      return
    }
    DispatchQueue.global(qos: .userInitiated).async {
      let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
      do {
        let result = try work(handler)
        promise.resolve(result)
      } catch {
        promise.reject("ERR_VISION", error.localizedDescription)
      }
    }
  }
}
