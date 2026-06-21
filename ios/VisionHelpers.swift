import Vision

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// Shared helpers for image loading and coordinate serialization.
enum VisionHelpers {
  static func loadImage(_ path: String) -> (CGImage, CGFloat, CGFloat)? {
    let url: URL
    if path.hasPrefix("file://") {
      guard let parsed = URL(string: path) else { return nil }
      url = parsed
    } else {
      url = URL(fileURLWithPath: path)
    }
    guard let data = try? Data(contentsOf: url) else { return nil }

    #if canImport(UIKit)
    guard let uiImage = UIImage(data: data),
          let cgImage = uiImage.cgImage else { return nil }
    #elseif canImport(AppKit)
    guard let nsImage = NSImage(data: data),
          let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
    #endif

    return (cgImage, CGFloat(cgImage.width), CGFloat(cgImage.height))
  }

  /// Serialize CGRect as-is (normalized 0-1, bottom-left origin).
  static func rawRect(_ rect: CGRect) -> [String: Double] {
    ["x": Double(rect.origin.x), "y": Double(rect.origin.y),
     "width": Double(rect.size.width), "height": Double(rect.size.height)]
  }

  /// Serialize CGPoint as-is (normalized 0-1, bottom-left origin).
  static func rawPoint(_ point: CGPoint) -> [String: Double] {
    ["x": Double(point.x), "y": Double(point.y)]
  }

  /// Decode a CGImage to raw RGBA pixel data.
  static func decodePixels(_ cgImage: CGImage) -> Data? {
    let w = cgImage.width
    let h = cgImage.height
    let bytesPerRow = w * 4
    var pixels = [UInt8](repeating: 0, count: w * h * 4)

    guard let ctx = CGContext(
      data: &pixels,
      width: w, height: h,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
    return Data(pixels)
  }
}
