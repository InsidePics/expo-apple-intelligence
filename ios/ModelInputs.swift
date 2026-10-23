import Foundation

/// Model-ready inputs built from a decoded photo, so JS receives a small
/// tensor instead of the full-resolution pixels.
///
/// Every function here is a port of the InsidePics JS pipeline it replaces,
/// operation for operation: Double arithmetic where JS used numbers, Float
/// where JS stored into a `Float32Array`, and the same order of additions.
/// Given the same RGBA it produces bit-identical results;
/// `ModelInputParityTests` runs the JS originals in JavaScriptCore on every
/// fixture and compares. Change the arithmetic here and an embedding stops
/// matching one the JS pipeline (or a fixture generated from it) produced.
enum ModelInputs {

  // MARK: - Center-crop tensor (CLIP)

  /// Scale the shorter side to `size` by nearest neighbour, center-crop
  /// `size`×`size`, and normalize each channel as `(v / 255 - mean) / std`.
  /// Returns `[1, 3, size, size]` Float32, planar R, G, B.
  ///
  /// Port of `computeClipEmbedding`'s preprocessing (`object/clip.ts` as of
  /// monorepo 79a602613; see `Tests/Fixtures/ml-reference.js`).
  static func centerCropTensor(
    _ image: RGBAImage, size: Int, mean: [Double], std: [Double]
  ) -> [Float] {
    let width = image.width
    let height = image.height
    let side = Double(size)
    let scale = side / Double(min(width, height))
    let scaledW = jsRound(Double(width) * scale)
    let scaledH = jsRound(Double(height) * scale)
    let cropX = ((scaledW - side) / 2).rounded(.down)
    let cropY = ((scaledH - side) / 2).rounded(.down)

    let plane = size * size
    var out = [Float](repeating: 0, count: 3 * plane)
    image.pixels.withUnsafeBufferPointer { px in
      for y in 0..<size {
        let srcY = sourceIndex(Double(y) + cropY, scale: scale, limit: height)
        for x in 0..<size {
          let srcX = sourceIndex(Double(x) + cropX, scale: scale, limit: width)
          let srcIdx = (srcY * width + srcX) * 4
          let offset = y * size + x
          for ch in 0..<3 {
            let value = Double(px[srcIdx + ch]) / 255
            out[ch * plane + offset] = Float((value - mean[ch]) / std[ch])
          }
        }
      }
    }
    return out
  }

  /// `Math.min(Math.floor(p / scale), limit - 1)`, never negative.
  private static func sourceIndex(_ p: Double, scale: Double, limit: Int) -> Int {
    let index = (p / scale).rounded(.down)
    return max(0, min(Int(index), limit - 1))
  }

  // MARK: - Face alignment (farside)

  /// One face aligned onto `targets` and cut to `size`×`size`: the
  /// `[1, 3, size, size]` Float32 tensor (`v / 127.5 - 1`, planar R, G, B)
  /// and the crop's blur score. Nil when the landmarks are degenerate.
  ///
  /// `landmarks` are image-normalized with a top-left origin; they are
  /// lifted to pixels exactly as `face/align.ts` did, then run through
  /// `estimateSimilarityTransform`, `warpAffine`, `blurScoreFromRgba` and
  /// `rgbaToNchwFloat`.
  static func alignedFace(
    _ image: RGBAImage, landmarks: [[Double]], targets: [[Double]], size: Int
  ) -> (tensor: [Float], blur: Double)? {
    let points = landmarks.map {
      [$0[0] * Double(image.width), $0[1] * Double(image.height)]
    }
    guard let transform = similarityTransform(from: points, to: targets) else {
      return nil
    }
    let crop = warpAffine(image, transform, outW: size, outH: size)
    return (signedUnitTensor(crop, size: size), blurScore(crop, width: size, height: size))
  }

  /// Least-squares similarity transform (rotation, uniform scale,
  /// translation) from `src` to `dst`, as `[a, b, tx, c, d, ty]`.
  /// Port of `estimateSimilarityTransform` (`face/alignMath.ts`).
  static func similarityTransform(from src: [[Double]], to dst: [[Double]]) -> [Double]? {
    let n = min(src.count, dst.count)
    if n < 2 { return nil }

    var srcCx = 0.0, srcCy = 0.0, dstCx = 0.0, dstCy = 0.0
    for i in 0..<n {
      srcCx += src[i][0]
      srcCy += src[i][1]
      dstCx += dst[i][0]
      dstCy += dst[i][1]
    }
    let count = Double(n)
    srcCx /= count
    srcCy /= count
    dstCx /= count
    dstCy /= count

    var num1 = 0.0, num2 = 0.0, denom = 0.0
    for i in 0..<n {
      let sx = src[i][0] - srcCx
      let sy = src[i][1] - srcCy
      let dx = dst[i][0] - dstCx
      let dy = dst[i][1] - dstCy
      num1 += dx * sx + dy * sy
      num2 += dy * sx - dx * sy
      denom += sx * sx + sy * sy
    }
    if denom == 0 { return nil }

    let a = num1 / denom
    let b = num2 / denom
    let tx = dstCx - a * srcCx + b * srcCy
    let ty = dstCy - a * srcCy - b * srcCx
    return [a, -b, tx, b, a, ty]
  }

  /// Bilinear backward warp of RGBA through the affine `m`; samples whose
  /// 2×2 neighbourhood leaves the image stay 0. Port of `warpAffine`.
  static func warpAffine(_ image: RGBAImage, _ m: [Double], outW: Int, outH: Int) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: outW * outH * 4)
    let (a, b, tx, c, d, ty) = (m[0], m[1], m[2], m[3], m[4], m[5])
    let det = a * d - b * c
    if abs(det) < 1e-10 { return out }
    let ia = d / det
    let ib = -b / det
    let ic = -c / det
    let id = a / det
    let srcW = Double(image.width)
    let srcH = Double(image.height)

    image.pixels.withUnsafeBufferPointer { px in
      for oy in 0..<outH {
        for ox in 0..<outW {
          let dx = Double(ox) - tx
          let dy = Double(oy) - ty
          let srcX = ia * dx + ib * dy
          let srcY = ic * dx + id * dy
          let x0 = srcX.rounded(.down)
          let y0 = srcY.rounded(.down)
          // Bounds in Double first: a NaN or huge coordinate must not reach Int().
          guard x0 >= 0, x0 + 1 < srcW, y0 >= 0, y0 + 1 < srcH else { continue }
          let fx = srcX - x0
          let fy = srcY - y0
          let row0 = Int(y0) * image.width
          let row1 = row0 + image.width
          let col0 = Int(x0)
          let outIdx = (oy * outW + ox) * 4
          for ch in 0..<4 {
            let v00 = Double(px[(row0 + col0) * 4 + ch])
            let v01 = Double(px[(row0 + col0 + 1) * 4 + ch])
            let v10 = Double(px[(row1 + col0) * 4 + ch])
            let v11 = Double(px[(row1 + col0 + 1) * 4 + ch])
            let w00: Double = v00 * (1 - fx) * (1 - fy)
            let w01: Double = v01 * fx * (1 - fy)
            let w10: Double = v10 * (1 - fx) * fy
            let w11: Double = v11 * fx * fy
            let value: Double = w00 + w01 + w10 + w11
            out[outIdx + ch] = UInt8(jsRound(value))
          }
        }
      }
    }
    return out
  }

  /// RGBA to `[1, 3, size, size]` Float32 in `[-1, 1]`, alpha dropped.
  /// Port of `rgbaToNchwFloat` (`face/tensorPrep.ts`).
  static func signedUnitTensor(_ rgba: [UInt8], size: Int) -> [Float] {
    let plane = size * size
    var out = [Float](repeating: 0, count: 3 * plane)
    for p in 0..<plane {
      let src = p * 4
      out[p] = Float(Double(rgba[src]) / 127.5 - 1)
      out[plane + p] = Float(Double(rgba[src + 1]) / 127.5 - 1)
      out[2 * plane + p] = Float(Double(rgba[src + 2]) / 127.5 - 1)
    }
    return out
  }

  // MARK: - Sharpness and histogram

  /// Variance of the 4-neighbour Laplacian over Rec.601 grey, which is
  /// stored as Float32 as `rgbaToGrayscale` did. Higher is sharper.
  /// Port of `blurScoreFromRgba` (`blurMath.ts`).
  static func blurScore(_ rgba: [UInt8], width: Int, height: Int) -> Double {
    if width < 3 || height < 3 { return 0 }
    let n = width * height
    var gray = [Float](repeating: 0, count: n)
    rgba.withUnsafeBufferPointer { px in
      for i in 0..<n {
        let idx = i * 4
        gray[i] = Float(luma(px[idx], px[idx + 1], px[idx + 2]))
      }
    }

    var sum = 0.0, sumSq = 0.0
    var count = 0
    gray.withUnsafeBufferPointer { g in
      for y in 1..<(height - 1) {
        for x in 1..<(width - 1) {
          let up = Double(g[(y - 1) * width + x])
          let down = Double(g[(y + 1) * width + x])
          let left = Double(g[y * width + (x - 1)])
          let right = Double(g[y * width + (x + 1)])
          let center: Double = 4 * Double(g[y * width + x])
          let lap: Double = up + down + left + right - center
          sum += lap
          sumSq += lap * lap
          count += 1
        }
      }
    }
    let mean = sum / Double(count)
    return sumSq / Double(count) - mean * mean
  }

  /// Normalized luminance histogram with Float32 bins, counted in pixel
  /// order. Port of `luminanceHistogram` (`video/histogram.ts`).
  static func luminanceHistogram(_ image: RGBAImage, bins: Int) -> [Float] {
    var hist = [Float](repeating: 0, count: bins)
    let total = image.width * image.height
    let binCount = Double(bins)
    image.pixels.withUnsafeBufferPointer { px in
      for i in 0..<total {
        let idx = i * 4
        let lum = luma(px[idx], px[idx + 1], px[idx + 2])
        let bin = min(Int((lum / 256 * binCount).rounded(.down)), bins - 1)
        hist[bin] += 1
      }
    }
    for i in 0..<bins {
      hist[i] = Float(Double(hist[i]) / Double(total))
    }
    return hist
  }

  // MARK: - Transfer

  /// Float32 values as their little-endian bytes, which JS views as a
  /// `Float32Array` without copying.
  static func bytes(_ floats: [Float]) -> Data {
    floats.withUnsafeBytes { Data($0) }
  }

  /// Rec.601 luma, `0.299 R + 0.587 G + 0.114 B`, summed left to right.
  private static func luma(_ r: UInt8, _ g: UInt8, _ b: UInt8) -> Double {
    let red: Double = 0.299 * Double(r)
    let green: Double = 0.587 * Double(g)
    let blue: Double = 0.114 * Double(b)
    return red + green + blue
  }

  /// `Math.round` for the non-negative values these ports round: nearest,
  /// ties up (for a non-negative value, up is away from zero).
  private static func jsRound(_ value: Double) -> Double {
    value.rounded(.toNearestOrAwayFromZero)
  }
}
