// Writes the EXIF-orientation test fixtures from the committed upright
// `ios/Tests/Fixtures/face.jpg`:
//
//   swift scripts/make-orientation-fixtures.swift   (from the repo root)
//
// For each tag in 1, 3, 6 and 8 it stores the pixels the way a camera would
// (rotated by the inverse of the tag) and writes the tag, as both JPEG and
// HEIC. A viewer that honours the tag shows exactly face.jpg; a loader that
// drops it sees the face on its side or upside down, which is how iPhone
// camera photos (stored landscape + orientation 6) reach Vision.
import CoreGraphics
import Foundation
import ImageIO

// Run from the repository root.
let fixtures = URL(fileURLWithPath: "ios/Tests/Fixtures")

func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data((message + "\n").utf8))
  exit(1)
}

/// The stored (camera) pixels for an upright image shown under `tag`.
/// Each tag names the rotation a viewer applies to the stored pixels, so the
/// stored pixels are the upright image turned the other way.
func storedPixels(_ upright: CGImage, tag: Int) -> CGImage {
  let w = upright.width
  let h = upright.height
  let swaps = tag == 6 || tag == 8
  let ctx = CGContext(
    data: nil,
    width: swaps ? h : w,
    height: swaps ? w : h,
    bitsPerComponent: 8,
    bytesPerRow: 0,
    space: CGColorSpace(name: CGColorSpace.sRGB)!,
    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
  )!
  // CoreGraphics is y-up, so a positive angle turns the picture
  // counter-clockwise as it is displayed.
  switch tag {
  case 3:  // viewer turns 180°
    ctx.translateBy(x: CGFloat(w), y: CGFloat(h))
    ctx.rotate(by: .pi)
  case 6:  // viewer turns 90° clockwise → store 90° counter-clockwise
    ctx.translateBy(x: CGFloat(h), y: 0)
    ctx.rotate(by: .pi / 2)
  case 8:  // viewer turns 90° counter-clockwise → store 90° clockwise
    ctx.translateBy(x: 0, y: CGFloat(w))
    ctx.rotate(by: -.pi / 2)
  default:
    break
  }
  ctx.draw(upright, in: CGRect(x: 0, y: 0, width: w, height: h))
  return ctx.makeImage()!
}

func write(_ image: CGImage, tag: Int, type: String, to url: URL) {
  guard
    let dest = CGImageDestinationCreateWithURL(
      url as CFURL, type as CFString, 1, nil)
  else { fail("cannot write \(type) at \(url.path)") }
  let props: [CFString: Any] = [
    kCGImagePropertyOrientation: tag,
    kCGImagePropertyTIFFDictionary: [kCGImagePropertyTIFFOrientation: tag],
    kCGImageDestinationLossyCompressionQuality: 0.9,
  ]
  CGImageDestinationAddImage(dest, image, props as CFDictionary)
  guard CGImageDestinationFinalize(dest) else { fail("finalize \(url.path)") }
}

guard
  let source = CGImageSourceCreateWithURL(
    fixtures.appendingPathComponent("face.jpg") as CFURL, nil),
  let upright = CGImageSourceCreateImageAtIndex(source, 0, nil)
else { fail("cannot read face.jpg in \(fixtures.path)") }

for tag in [1, 3, 6, 8] {
  let stored = storedPixels(upright, tag: tag)
  for (ext, type) in [("jpg", "public.jpeg"), ("heic", "public.heic")] {
    let url = fixtures.appendingPathComponent("face-orientation-\(tag).\(ext)")
    write(stored, tag: tag, type: type, to: url)
    print("wrote \(url.lastPathComponent) (\(stored.width)x\(stored.height))")
  }
}
