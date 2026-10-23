// JS reference for ModelInputParityTests: the InsidePics ML pipeline as it
// computed model inputs in JS before they moved native (monorepo #791).
// `clipTensor` and `extractFiveLandmarks` are copied verbatim from monorepo
// commit 79a602613 (object/clip.ts, face/align.ts), where they were inline
// or private; everything else is imported from that commit unchanged.
import { estimateSimilarityTransform, warpAffine } from '@monorepo/ui/src/providers/ml/face/alignMath';
import { blurScoreFromRgba } from '@monorepo/ui/src/providers/ml/blurMath';
import { rgbaToNchwFloat } from '@monorepo/ui/src/providers/ml/face/tensorPrep';
import { luminanceHistogram } from '@monorepo/ui/src/providers/ml/video/histogram';

const IMAGE_SIZE = 256;
const CROP_SIZE = 112;
const TARGET_LANDMARKS: Point2D[] = [
  [38.2946, 51.6963], // left eye
  [73.5318, 51.5014], // right eye
  [56.0252, 71.7366], // nose
  [41.5493, 92.3655], // left mouth
  [70.7299, 92.2041], // right mouth
];

function clipTensor(pixels, width, height) {
  // Scale shorter side to 256, then center-crop 256x256
  const scale = IMAGE_SIZE / Math.min(width, height);
  const scaledW = Math.round(width * scale);
  const scaledH = Math.round(height * scale);

  const cropX = Math.floor((scaledW - IMAGE_SIZE) / 2);
  const cropY = Math.floor((scaledH - IMAGE_SIZE) / 2);

  // Prepare input tensor [1, 3, 256, 256] float32
  // MobileCLIP normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  const tensorData = new Float32Array(1 * 3 * IMAGE_SIZE * IMAGE_SIZE);

  for (let y = 0; y < IMAGE_SIZE; y++) {
    for (let x = 0; x < IMAGE_SIZE; x++) {
      // Map back to original image coordinates
      const srcX = Math.min(Math.floor((x + cropX) / scale), width - 1);
      const srcY = Math.min(Math.floor((y + cropY) / scale), height - 1);
      const srcIdx = (srcY * width + srcX) * 4;

      const pixelOffset = y * IMAGE_SIZE + x;
      for (let ch = 0; ch < 3; ch++) {
        const val = pixels[srcIdx + ch] / 255;
        tensorData[ch * IMAGE_SIZE * IMAGE_SIZE + pixelOffset] =
          (val - mean[ch]) / std[ch];
      }
    }
  }

  return tensorData;
}

function extractFiveLandmarks(
  face: FaceObservation,
  imageWidth: number,
  imageHeight: number
): Point2D[] | null {
  const lm = face.landmarks;
  if (!lm) return null;

  const bbox = face.boundingBox;

  // Center of a landmark region in pixel coords
  function regionCenter(region: NormalizedPoint[] | undefined): Point2D | null {
    if (!region || region.length === 0) return null;
    let sx = 0,
      sy = 0;
    for (const pt of region) {
      sx += pt.x;
      sy += pt.y;
    }
    const ax = sx / region.length;
    const ay = sy / region.length;
    // face-bbox-relative → image-normalized → pixel
    const imgX = (bbox.x + ax * bbox.width) * imageWidth;
    const imgY = (1.0 - (bbox.y + ay * bbox.height)) * imageHeight;
    return [imgX, imgY];
  }

  // Specific point from a region in pixel coords
  function regionPoint(
    region: NormalizedPoint[] | undefined,
    idx: number
  ): Point2D | null {
    if (!region || idx >= region.length) return null;
    const pt = region[idx];
    const imgX = (bbox.x + pt.x * bbox.width) * imageWidth;
    const imgY = (1.0 - (bbox.y + pt.y * bbox.height)) * imageHeight;
    return [imgX, imgY];
  }

  const leftEye = regionCenter(lm.leftEye);
  const rightEye = regionCenter(lm.rightEye);
  const nose = regionCenter(lm.nose);
  const leftMouth = regionPoint(lm.outerLips, 0);
  const rightMouth = regionPoint(lm.outerLips, 6);

  if (!leftEye || !rightEye || !nose || !leftMouth || !rightMouth) return null;
  return [leftEye, rightEye, nose, leftMouth, rightMouth];
}


function alignFace(pixels, width, height, detection) {
  const landmarks = extractFiveLandmarks(detection, width, height);
  if (!landmarks) return null;
  const transform = estimateSimilarityTransform(landmarks, TARGET_LANDMARKS);
  if (!transform) return null;
  const cropped = warpAffine(pixels, width, height, transform, CROP_SIZE, CROP_SIZE);
  const blur = blurScoreFromRgba(cropped, CROP_SIZE, CROP_SIZE);
  return { tensor: rgbaToNchwFloat(cropped, CROP_SIZE), blur };
}

globalThis.MLReference = {
  clipTensor,
  alignFace,
  extractFiveLandmarks,
  targetLandmarks: TARGET_LANDMARKS,
  blurScoreFromRgba,
  luminanceHistogram,
};
