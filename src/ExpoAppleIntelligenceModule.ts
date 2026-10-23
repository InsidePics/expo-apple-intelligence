import { NativeModule, requireNativeModule } from 'expo';

import type {
  ExpoAppleIntelligenceModuleEvents,
  ImageAnalysisResult,
  FaceObservation,
  TextObservation,
  ClassificationObservation,
  BarcodeObservation,
  BodyPoseObservation,
  HandPoseObservation,
  AnimalObservation,
  RectangleObservation,
  FeaturePrintResult,
  AestheticsResult,
  SaliencyResult,
  HorizonResult,
  LensSmudgeResult,
  DocumentRecognitionResult,
  TranscriptionResult,
  Float32TensorBytes,
  AlignedFaceTensor,
} from './ExpoAppleIntelligence.types';

declare class ExpoAppleIntelligenceModule extends NativeModule<ExpoAppleIntelligenceModuleEvents> {
  /** Run all available analyses in a single pass (most efficient) */
  analyzeImage(imagePath: string): Promise<ImageAnalysisResult>;

  /** Face detection with landmarks, quality, angles */
  detectFaces(imagePath: string): Promise<FaceObservation[]>;

  /** OCR text recognition — array of observations with candidates */
  recognizeText(imagePath: string): Promise<TextObservation[]>;

  /** Image classification — all observations, unfiltered */
  classifyImage(imagePath: string): Promise<ClassificationObservation[]>;

  /** Barcode and QR code detection */
  detectBarcodes(imagePath: string): Promise<BarcodeObservation[]>;

  /** Human body pose detection */
  detectBodyPoses(imagePath: string): Promise<BodyPoseObservation[]>;

  /** Hand pose detection */
  detectHandPoses(imagePath: string): Promise<HandPoseObservation[]>;

  /** Image similarity vector */
  generateFeaturePrint(imagePath: string): Promise<FeaturePrintResult | null>;

  /** Aesthetic quality score (iOS 18+/macOS 15+ only, returns null otherwise) */
  calculateAesthetics(imagePath: string): Promise<AestheticsResult | null>;

  /** Salient regions */
  detectSaliency(imagePath: string): Promise<SaliencyResult | null>;

  /** Animal detection with all labels */
  detectAnimals(imagePath: string): Promise<AnimalObservation[]>;

  /** Rectangle detection with corner points */
  detectRectangles(imagePath: string): Promise<RectangleObservation[]>;

  /** Horizon angle in radians */
  detectHorizon(imagePath: string): Promise<HorizonResult | null>;

  /** Lens smudge detection (iOS 26+/macOS 26+ only, returns null otherwise) */
  detectLensSmudge(imagePath: string): Promise<LensSmudgeResult | null>;

  /** Structured document recognition (iOS 26+/macOS 26+ only, returns null otherwise) */
  recognizeDocument(
    imagePath: string
  ): Promise<DocumentRecognitionResult | null>;

  /** Check if on-device Foundation Model is available (iOS 26+) */
  isFoundationModelAvailable(): boolean;

  /** Generate text using on-device Foundation Model (iOS 26+ only) */
  generateText(prompt: string, systemPrompt?: string): Promise<string | null>;

  /** Transcribe audio file to text (iOS 26+ only) */
  transcribeAudio(
    audioPath: string,
    locale?: string
  ): Promise<TranscriptionResult | null>;

  /** Generate image from text prompt using ImagePlayground (iOS 18.4+ only) */
  generateImage(
    prompt: string,
    style?: 'animation' | 'illustration' | 'sketch'
  ): Promise<string | null>;

  /**
   * Decode any image (JPEG, PNG, HEIC, etc.) to raw RGBA pixels. Full
   * resolution, as base64: tens of megabytes for a camera photo. To feed a
   * model, prefer `centerCropTensor` / `alignFaces`, which return only the
   * model's input.
   */
  decodeImagePixels(imagePath: string): Promise<{
    pixels: string; // base64-encoded RGBA bytes
    width: number;
    height: number;
  }>;

  /**
   * Model input for a square image model (e.g. CLIP): the upright image's
   * shorter side scaled to `size` (nearest neighbour), center-cropped to
   * `size`×`size`, each channel normalized as `(v / 255 - mean) / std`.
   * The image is decoded natively; only the tensor crosses to JS.
   */
  centerCropTensor(
    imagePath: string,
    size: number,
    mean: [number, number, number],
    std: [number, number, number],
  ): Promise<Float32TensorBytes>;

  /**
   * Model input for a face encoder: each face's five landmarks (left eye,
   * right eye, nose, left and right mouth corner; image-normalized, top-left
   * origin) aligned onto `targets` (pixels of the `size`×`size` crop) by a
   * least-squares similarity transform and bilinear warp. Each tensor is
   * planar RGB scaled to `[-1, 1]` (`v / 127.5 - 1`), with the crop's blur
   * score (Laplacian variance). One decode serves every face; a face whose
   * landmarks are degenerate resolves to null.
   */
  alignFaces(
    imagePath: string,
    landmarks: [number, number][][],
    targets: [number, number][],
    size: number,
  ): Promise<(AlignedFaceTensor | null)[]>;

  /** Sharpness (Laplacian variance of Rec.601 grey) of the whole image. */
  imageSharpness(imagePath: string): Promise<number>;

  /** Luminance histogram with `bins` bins, normalized to sum to 1. */
  luminanceHistogram(imagePath: string, bins: number): Promise<number[]>;
}

export default requireNativeModule<ExpoAppleIntelligenceModule>(
  'ExpoAppleIntelligence'
);
