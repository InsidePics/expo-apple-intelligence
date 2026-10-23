import ImageIO
import Vision

/// Shared helpers for image loading and coordinate serialization.
enum VisionHelpers {
  /// Loads an image upright: the EXIF orientation is applied to the pixels,
  /// so the returned image, its size, every Vision box and landmark computed
  /// on it, and `decodePixels` all share the frame a viewer shows. Camera
  /// photos store sensor-orientation pixels plus a tag (an iPhone portrait is
  /// landscape + orientation 6); ignoring the tag puts every box on the
  /// photo's side.
  static func loadImage(_ path: String) -> (CGImage, CGFloat, CGFloat)? {
    let url: URL
    if path.hasPrefix("file://") {
      guard let parsed = URL(string: path) else { return nil }
      url = parsed
    } else {
      url = URL(fileURLWithPath: path)
    }
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cgImage = uprightImage(source) else { return nil }
    return (cgImage, CGFloat(cgImage.width), CGFloat(cgImage.height))
  }

  private static func uprightImage(_ source: CGImageSource) -> CGImage? {
    let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
    let orientation = props?[kCGImagePropertyOrientation] as? UInt32 ?? 1
    if orientation == CGImagePropertyOrientation.up.rawValue {
      return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
    // The thumbnail API is ImageIO's way to apply the orientation; with no
    // max pixel size it is a full-resolution decode.
    let options: [CFString: Any] = [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: true,
    ]
    return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
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
