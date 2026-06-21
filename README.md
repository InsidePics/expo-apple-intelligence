# @insidepics/expo-apple-intelligence

[![npm version](https://img.shields.io/npm/v/@insidepics/expo-apple-intelligence.svg)](https://www.npmjs.com/package/@insidepics/expo-apple-intelligence)
[![npm downloads](https://img.shields.io/npm/dm/@insidepics/expo-apple-intelligence.svg)](https://www.npmjs.com/package/@insidepics/expo-apple-intelligence)
[![CI](https://github.com/InsidePics/expo-apple-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/InsidePics/expo-apple-intelligence/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/npm/l/@insidepics/expo-apple-intelligence.svg)](./LICENSE)

An Expo native module exposing Apple's on-device intelligence APIs — Foundation Models, Vision, Speech, and Image Playground — to React Native / Expo.

All processing runs locally on the device. Nothing is sent to the network. Features that depend on newer OS releases are availability-gated at runtime: when a capability is unavailable (older OS, unsupported hardware, or model not yet downloaded), the corresponding call resolves to `null` rather than throwing, so you can degrade gracefully.

## Features

Grouped by the Apple framework that backs each capability, with the minimum OS version derived from the `@available` gates in the native source.

### Vision — image analysis (iOS 15.1+ / macOS 13.0+)

- **Comprehensive analysis** — run every available Vision request in a single pass.
- **Face detection** — bounding boxes, landmarks, roll/yaw/pitch angles, and capture quality.
- **Text recognition (OCR)** — accurate-level recognition with up to 10 ranked candidates per observation.
- **Image classification** — full unfiltered label set with confidences.
- **Barcode & QR detection** — symbology, payload string, and corner points.
- **Human body pose** and **hand pose** detection — named joints with normalized coordinates and confidences.
- **Image feature print** — a similarity vector for nearest-neighbour / dedup use cases.
- **Saliency** — attention-based and objectness-based salient regions.
- **Animal detection** — bounding boxes with per-animal labels.
- **Rectangle detection** — bounding box plus four corner points.
- **Horizon detection** — horizon angle in radians.
- **Raw pixel decode** — decode any JPEG/PNG/HEIC to base64 RGBA bytes.

### Vision — modern requests

- **Image aesthetics score** — overall aesthetic score and a utility-image flag. _Requires iOS 18.0+ / macOS 15.0+._
- **Lens smudge detection** — confidence that the lens was smudged/dirty. _Requires iOS 26.0+ / macOS 26.0+._
- **Structured document recognition** — paragraphs, tables, lists, barcodes, and detected data (emails, phone numbers, URLs, addresses, etc.). _Requires iOS 26.0+ / macOS 26.0+._

### Foundation Models — on-device LLM (iOS 26.0+)

- **Availability check** and **text generation** against Apple's on-device `SystemLanguageModel`, with an optional system prompt.

### Speech — transcription (iOS 26.0+)

- **Audio transcription** of a local audio file via `SpeechAnalyzer` / `SpeechTranscriber`, with an optional BCP-47 locale.

### Image Playground — image generation (iOS 18.4+)

- **Text-to-image generation** via `ImageCreator`, in `animation`, `illustration`, or `sketch` styles; returns a path to the generated JPEG.

## Requirements

- **iOS deployment target: 15.1** (macOS 13.0). This is the floor enforced by the podspec; the module compiles and links against this baseline.
- **Availability-gated features.** The 15.1 floor only covers the classic Vision requests. Newer capabilities are gated at runtime:
  - Aesthetics → iOS 18.0 / macOS 15.0
  - Image generation (Image Playground) → iOS 18.4
  - Foundation Models, Speech transcription, lens-smudge, document recognition → iOS 26.0 / macOS 26.0
  - On older OS versions these calls resolve to `null` instead of throwing.
- **Custom dev client / prebuild required.** This is a native module — it does **not** run in Expo Go. You must use a [development build](https://docs.expo.dev/develop/development-builds/introduction/) (or a bare/prebuild workflow) and rebuild the native app after installing.
- **Platforms:** `apple` (`ios` + `macos`), per `expo-module.config.json`.

> Note: many features (Foundation Models, Image Playground, document recognition) also require Apple Intelligence-capable hardware and that the relevant on-device models have been downloaded. Always check the return value for `null`.

## Installation

```sh
npx expo install @insidepics/expo-apple-intelligence
```

Then regenerate the native project and rebuild the dev client:

```sh
npx expo prebuild
# build & run the dev client
npx expo run:ios
```

If you manage the iOS project directly, run `npx pod-install` after installing.

## Usage

```ts
import ExpoAppleIntelligence from '@insidepics/expo-apple-intelligence';
import type {
  ImageAnalysisResult,
  TextObservation,
} from '@insidepics/expo-apple-intelligence';

// 1. Run every available analysis on an image in a single pass.
const result: ImageAnalysisResult = await ExpoAppleIntelligence.analyzeImage(
  '/path/to/photo.jpg'
);
console.log(`${result.faces.length} faces, ${result.labels.length} labels`);
console.log('image size', result.imageWidth, result.imageHeight);

// 2. Or call a single Vision request directly.
const lines: TextObservation[] = await ExpoAppleIntelligence.recognizeText(
  '/path/to/receipt.jpg'
);
for (const line of lines) {
  console.log(line.candidates[0]?.string, '@', line.candidates[0]?.confidence);
}

// 3. On-device LLM (iOS 26+). Returns null when unavailable.
if (ExpoAppleIntelligence.isFoundationModelAvailable()) {
  const reply = await ExpoAppleIntelligence.generateText(
    'Summarize this caption in five words.',
    'You are a concise assistant.' // optional system prompt
  );
  console.log(reply);
}

// 4. Text-to-image (iOS 18.4+). Returns a file path, or null when unavailable.
const imagePath = await ExpoAppleIntelligence.generateImage(
  'a fox reading a book',
  'illustration'
);
```

Image and audio arguments accept either a plain filesystem path or a `file://` URL.

## API reference

The default export is the native module. Every method takes an image (or audio) path string. Vision coordinates are normalized (0–1) with a bottom-left origin, exactly as Apple returns them; face landmark points are normalized to the face bounding box.

| Function | Signature | Description | Min OS / Framework |
| --- | --- | --- | --- |
| `analyzeImage` | `(imagePath: string) => Promise<ImageAnalysisResult>` | Runs all available Vision analyses in one pass (most efficient). Includes iOS 26 lens-smudge & document results when available. | iOS 15.1 (Vision); richer fields gated higher |
| `detectFaces` | `(imagePath: string) => Promise<FaceObservation[]>` | Face detection with landmarks, roll/yaw/pitch, and capture quality. | iOS 15.1 / Vision |
| `recognizeText` | `(imagePath: string) => Promise<TextObservation[]>` | Accurate-level OCR; up to 10 candidates per observation. | iOS 15.1 / Vision |
| `classifyImage` | `(imagePath: string) => Promise<ClassificationObservation[]>` | Full unfiltered image classification. | iOS 15.1 / Vision |
| `detectBarcodes` | `(imagePath: string) => Promise<BarcodeObservation[]>` | Barcode & QR detection with symbology, payload, corner points. | iOS 15.1 / Vision |
| `detectBodyPoses` | `(imagePath: string) => Promise<BodyPoseObservation[]>` | Human body pose detection (named joints). | iOS 15.1 / Vision |
| `detectHandPoses` | `(imagePath: string) => Promise<HandPoseObservation[]>` | Hand pose detection (up to 4 hands). | iOS 15.1 / Vision |
| `generateFeaturePrint` | `(imagePath: string) => Promise<FeaturePrintResult \| null>` | Image similarity feature-print vector. | iOS 15.1 / Vision |
| `detectSaliency` | `(imagePath: string) => Promise<SaliencyResult \| null>` | Attention-based and objectness-based salient regions. | iOS 15.1 / Vision |
| `detectAnimals` | `(imagePath: string) => Promise<AnimalObservation[]>` | Animal detection with all labels. | iOS 15.1 / Vision |
| `detectRectangles` | `(imagePath: string) => Promise<RectangleObservation[]>` | Rectangle detection with corner points (max 10). | iOS 15.1 / Vision |
| `detectHorizon` | `(imagePath: string) => Promise<HorizonResult \| null>` | Horizon angle in radians. | iOS 15.1 / Vision |
| `decodeImagePixels` | `(imagePath: string) => Promise<{ pixels: string; width: number; height: number }>` | Decode any image to base64-encoded RGBA bytes. | iOS 15.1 |
| `calculateAesthetics` | `(imagePath: string) => Promise<AestheticsResult \| null>` | Aesthetic quality score + utility-image flag. Returns `null` on older OS. | iOS 18.0 / macOS 15.0 — Vision |
| `generateImage` | `(prompt: string, style?: 'animation' \| 'illustration' \| 'sketch') => Promise<string \| null>` | Text-to-image; returns a path to the generated JPEG, or `null`. | iOS 18.4 / ImagePlayground |
| `detectLensSmudge` | `(imagePath: string) => Promise<LensSmudgeResult \| null>` | Lens smudge confidence. Returns `null` on older OS. | iOS 26.0 / macOS 26.0 — Vision |
| `recognizeDocument` | `(imagePath: string) => Promise<DocumentRecognitionResult \| null>` | Structured document recognition (paragraphs, tables, lists, barcodes, detected data). | iOS 26.0 / macOS 26.0 — Vision |
| `isFoundationModelAvailable` | `() => boolean` | Synchronous check for on-device Foundation Model availability. | iOS 26.0 / FoundationModels |
| `generateText` | `(prompt: string, systemPrompt?: string) => Promise<string \| null>` | On-device LLM text generation with an optional system prompt. | iOS 26.0 / FoundationModels |
| `transcribeAudio` | `(audioPath: string, locale?: string) => Promise<TranscriptionResult \| null>` | Transcribe a local audio file; optional BCP-47 locale. | iOS 26.0 / Speech (SpeechAnalyzer) |

All result types (`FaceObservation`, `TextObservation`, `ImageAnalysisResult`, `DocumentRecognitionResult`, etc.) are exported from the package entry point.

## Capability / availability matrix

| Capability | Min iOS | Min macOS | Backing framework |
| --- | --- | --- | --- |
| Faces, text, classification, barcodes, body/hand pose, feature print, saliency, animals, rectangles, horizon, pixel decode | 15.1 | 13.0 | Vision |
| Image aesthetics score | 18.0 | 15.0 | Vision |
| Text-to-image generation | 18.4 | — | ImagePlayground |
| Foundation Models availability + text generation | 26.0 | — | FoundationModels |
| Speech transcription | 26.0 | — | Speech (SpeechAnalyzer / SpeechTranscriber) |
| Lens smudge detection | 26.0 | 26.0 | Vision |
| Structured document recognition | 26.0 | 26.0 | Vision |

## Contributing

```sh
git clone https://github.com/InsidePics/expo-apple-intelligence.git
cd expo-apple-intelligence

pnpm install   # install dependencies
pnpm build     # compile the TypeScript module

# run the example app (open the iOS project)
pnpm open:ios
```

The native Swift sources live in `ios/` (`ExpoAppleIntelligenceModule.swift`, `VisionHelpers.swift`, `VisionModern.swift`, `VisionProcessors.swift`); the TypeScript surface lives in `src/`. Keep the `@available` gates in the Swift code and the documented minimum-OS values in this README in sync when adding capabilities.

## License

MIT © 2026 INSP LLC. See [LICENSE](./LICENSE).
