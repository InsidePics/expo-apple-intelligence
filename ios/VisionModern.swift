import Vision
import CoreImage
import AVFAudio

#if canImport(UIKit)
import UIKit
#endif

#if canImport(FoundationModels)
import FoundationModels
#endif

#if canImport(Speech)
import Speech
#endif

#if canImport(ImagePlayground)
import ImagePlayground
#endif

/// iOS 18+/macOS 15+ and iOS 26+/macOS 26+ Vision APIs (Swift-native, async/await).
enum VisionModern {

  // MARK: - Aesthetics (iOS 18+ / macOS 15+)

  @available(iOS 18.0, macOS 15.0, *)
  static func processAesthetics(_ results: [Any]?) -> [String: Any]? {
    guard let obs = results?.first as? VNImageAestheticsScoresObservation else { return nil }
    return ["overallScore": Double(obs.overallScore), "isUtility": obs.isUtility]
  }

  // MARK: - Lens Smudge (iOS 26+ / macOS 26+)

  @available(iOS 26.0, macOS 26.0, *)
  static func detectLensSmudge(cgImage: CGImage) async -> [String: Any]? {
    let ciImage = CIImage(cgImage: cgImage)
    let request = DetectLensSmudgeRequest()
    guard let obs = try? await request.perform(on: ciImage) else { return nil }
    return ["confidence": obs.confidence]
  }

  // MARK: - Document Recognition (iOS 26+ / macOS 26+)

  @available(iOS 26.0, macOS 26.0, *)
  static func recognizeDocument(cgImage: CGImage) async -> [String: Any]? {
    let ciImage = CIImage(cgImage: cgImage)
    let request = RecognizeDocumentsRequest()
    guard let docObs = try? await request.perform(on: ciImage).first else { return nil }
    let doc = docObs.document
    return processDocument(doc)
  }

  @available(iOS 26.0, macOS 26.0, *)
  private static func processDocument(_ doc: DocumentObservation.Container) -> [String: Any] {
    let paragraphs: [[String: Any]] = doc.paragraphs.map { para in
      var detectedData: [[String: Any]] = []
      for item in para.detectedData {
        detectedData.append([
          "type": detectedDataType(item),
          "value": "\(item.match)",
          "boundingBox": VisionHelpers.rawRect(item.boundingRegion.boundingBox.cgRect),
        ])
      }
      return [
        "text": para.transcript,
        "boundingBox": VisionHelpers.rawRect(para.boundingRegion.boundingBox.cgRect),
        "detectedData": detectedData,
      ]
    }

    let tables: [[String: Any]] = doc.tables.enumerated().map { (_, table) in
      var cells: [[String: Any]] = []
      for (rowIdx, row) in table.rows.enumerated() {
        for (colIdx, cell) in row.enumerated() {
          let text = cell.content.paragraphs.map { $0.transcript }.joined(separator: "\n")
          cells.append(["text": text, "row": rowIdx, "column": colIdx])
        }
      }
      return [
        "cells": cells,
        "boundingBox": VisionHelpers.rawRect(table.boundingRegion.boundingBox.cgRect),
      ]
    }

    let lists: [[String: Any]] = doc.lists.map { list in
      let items: [[String: Any]] = list.items.map { item in
        [
          "text": item.itemString,
          "marker": item.markerString,
        ]
      }
      return [
        "items": items,
        "boundingBox": VisionHelpers.rawRect(list.boundingRegion.boundingBox.cgRect),
      ]
    }

    let barcodes: [[String: Any]] = doc.barcodes.map { barcode in
      [
        "value": barcode.payloadString ?? "",
        "symbology": "\(barcode.symbology)",
        "boundingBox": VisionHelpers.rawRect(barcode.boundingRegion.boundingBox.cgRect),
      ]
    }

    return ["paragraphs": paragraphs, "tables": tables, "lists": lists, "barcodes": barcodes]
  }

  @available(iOS 26.0, macOS 26.0, *)
  private static func detectedDataType(_ data: DocumentObservation.Container.DataDetectorMatch) -> String {
    switch data.match.details {
    case .emailAddress: return "email"
    case .phoneNumber: return "phoneNumber"
    case .link: return "url"
    case .calendarEvent: return "calendarEvent"
    case .postalAddress: return "postalAddress"
    case .moneyAmount: return "moneyAmount"
    case .measurement: return "measurement"
    default: return "unknown"
    }
  }

  // MARK: - Foundation Models (iOS 26+)

  @available(iOS 26.0, *)
  static func isFoundationModelAvailable() -> Bool {
    #if canImport(FoundationModels)
    return SystemLanguageModel.default.availability == .available
    #else
    return false
    #endif
  }

  @available(iOS 26.0, *)
  static func generateText(prompt: String, systemPrompt: String?) async -> String? {
    #if canImport(FoundationModels)
    guard SystemLanguageModel.default.availability == .available else { return nil }
    do {
      let session: LanguageModelSession
      if let systemPrompt = systemPrompt {
        session = LanguageModelSession(instructions: systemPrompt)
      } else {
        session = LanguageModelSession()
      }
      let response = try await session.respond(to: prompt)
      return response.content
    } catch {
      return nil
    }
    #else
    return nil
    #endif
  }

  // MARK: - Speech Transcription (iOS 26+)

  @available(iOS 26.0, *)
  static func transcribeAudio(audioPath: String, locale: String?) async -> [String: Any]? {
    #if canImport(Speech)
    let url: URL
    if audioPath.hasPrefix("file://") {
      guard let parsed = URL(string: audioPath) else { return nil }
      url = parsed
    } else {
      url = URL(fileURLWithPath: audioPath)
    }

    guard let audioFile = try? AVAudioFile(forReading: url) else { return nil }

    let loc = locale != nil ? Locale(identifier: locale!) : .current

    do {
      let transcriber = SpeechTranscriber(locale: loc, preset: .transcription)
      _ = try await SpeechAnalyzer(
        inputAudioFile: audioFile,
        modules: [transcriber],
        finishAfterFile: true
      )

      var segments: [[String: Any]] = []
      for try await result in transcriber.results {
        let text = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
          segments.append(["text": text])
        }
      }

      if segments.isEmpty { return nil }
      return ["segments": segments]
    } catch {
      return nil
    }
    #else
    return nil
    #endif
  }

  // MARK: - Image Generation (iOS 18.4+)

  @available(iOS 18.4, *)
  static func generateImage(prompt: String, style: String?) async -> String? {
    #if canImport(ImagePlayground)
    do {
      let creator = try await ImageCreator()

      let imageStyle: ImagePlaygroundStyle
      switch style {
      case "illustration": imageStyle = .illustration
      case "sketch": imageStyle = .sketch
      default: imageStyle = .animation
      }

      var resultImage: CGImage? = nil
      for try await image in creator.images(for: [.text(prompt)], style: imageStyle, limit: 1) {
        resultImage = image.cgImage
        break
      }

      guard let cgImage = resultImage else { return nil }

      let fileName = UUID().uuidString + ".jpg"
      let filePath = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

      #if canImport(UIKit)
      guard let data = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.9) else { return nil }
      try data.write(to: filePath)
      #else
      return nil
      #endif

      return filePath.path
    } catch {
      return nil
    }
    #else
    return nil
    #endif
  }
}
