// --- Geometry primitives (normalized 0-1, bottom-left origin) ---

export type NormalizedRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type NormalizedPoint = {
  x: number;
  y: number;
};

// --- Face detection ---

export type FaceLandmarkRegion = NormalizedPoint[];

export type FaceLandmarks = {
  faceContour?: FaceLandmarkRegion;
  leftEye?: FaceLandmarkRegion;
  rightEye?: FaceLandmarkRegion;
  leftEyebrow?: FaceLandmarkRegion;
  rightEyebrow?: FaceLandmarkRegion;
  nose?: FaceLandmarkRegion;
  noseCrest?: FaceLandmarkRegion;
  medianLine?: FaceLandmarkRegion;
  outerLips?: FaceLandmarkRegion;
  innerLips?: FaceLandmarkRegion;
  leftPupil?: FaceLandmarkRegion;
  rightPupil?: FaceLandmarkRegion;
};

export type FaceObservation = {
  boundingBox: NormalizedRect;
  confidence: number;
  /** Roll angle in radians */
  roll?: number;
  /** Yaw angle in radians */
  yaw?: number;
  /** Pitch angle in radians */
  pitch?: number;
  /** Face capture quality 0-1 */
  quality?: number;
  /** Landmark regions — points are normalized to the face bounding box */
  landmarks?: FaceLandmarks;
};

// --- OCR / Text recognition ---

export type TextCandidate = {
  string: string;
  confidence: number;
};

export type TextObservation = {
  boundingBox: NormalizedRect;
  confidence: number;
  /** Up to 10 candidates, ordered by confidence */
  candidates: TextCandidate[];
};

// --- Image classification ---

export type ClassificationObservation = {
  identifier: string;
  confidence: number;
};

// --- Barcode detection ---

export type BarcodeObservation = {
  boundingBox: NormalizedRect;
  confidence: number;
  symbology: string;
  payloadStringValue?: string;
  topLeft: NormalizedPoint;
  topRight: NormalizedPoint;
  bottomLeft: NormalizedPoint;
  bottomRight: NormalizedPoint;
};

// --- Body pose detection ---

export type PoseJoint = {
  name: string;
  x: number;
  y: number;
  confidence: number;
};

export type BodyPoseObservation = {
  joints: PoseJoint[];
};

// --- Hand pose detection ---

export type HandPoseObservation = {
  joints: PoseJoint[];
};

// --- Image feature print ---

export type FeaturePrintResult = {
  data: number[];
  elementType: string;
  elementCount: number;
};

// --- Aesthetics score (iOS 18+) ---

export type AestheticsResult = {
  overallScore: number;
  isUtility: boolean;
};

// --- Saliency ---

export type SaliencyRegion = {
  boundingBox: NormalizedRect;
  confidence: number;
};

export type SaliencyResult = {
  attentionRegions: SaliencyRegion[];
  objectnessRegions: SaliencyRegion[];
};

// --- Animal detection ---

export type AnimalObservation = {
  boundingBox: NormalizedRect;
  confidence: number;
  labels: ClassificationObservation[];
};

// --- Rectangle detection ---

export type RectangleObservation = {
  boundingBox: NormalizedRect;
  topLeft: NormalizedPoint;
  topRight: NormalizedPoint;
  bottomLeft: NormalizedPoint;
  bottomRight: NormalizedPoint;
  confidence: number;
};

// --- Horizon detection ---

export type HorizonResult = {
  /** Angle in radians */
  angle: number;
};

// --- Lens smudge detection (iOS 26+) ---

export type LensSmudgeResult = {
  confidence: number;
};

// --- Document recognition (iOS 26+) ---

export type DocumentParagraph = {
  text: string;
  boundingBox: NormalizedRect;
  detectedData: DetectedDataItem[];
};

export type DocumentTableCell = {
  text: string;
  row: number;
  column: number;
};

export type DocumentTable = {
  cells: DocumentTableCell[];
  boundingBox: NormalizedRect;
};

export type DetectedDataItem = {
  type:
    | 'phoneNumber'
    | 'postalAddress'
    | 'calendarEvent'
    | 'moneyAmount'
    | 'measurement'
    | 'url'
    | 'email'
    | 'unknown';
  value: string;
  boundingBox: NormalizedRect;
};

export type DocumentListItem = {
  text: string;
  marker: string;
};

export type DocumentList = {
  items: DocumentListItem[];
  boundingBox: NormalizedRect;
};

export type DocumentBarcode = {
  value: string;
  symbology: string;
  boundingBox: NormalizedRect;
};

export type DocumentRecognitionResult = {
  paragraphs: DocumentParagraph[];
  tables: DocumentTable[];
  lists: DocumentList[];
  barcodes: DocumentBarcode[];
};

// --- Comprehensive analysis result ---

export type ImageAnalysisResult = {
  faces: FaceObservation[];
  text: TextObservation[];
  labels: ClassificationObservation[];
  barcodes: BarcodeObservation[];
  bodyPoses: BodyPoseObservation[];
  handPoses: HandPoseObservation[];
  rectangles: RectangleObservation[];
  animals: AnimalObservation[];
  featurePrint: FeaturePrintResult | null;
  aesthetics: AestheticsResult | null;
  saliency: SaliencyResult;
  horizon: HorizonResult | null;
  lensSmudge: LensSmudgeResult | null;
  document: DocumentRecognitionResult | null;
  imageWidth: number;
  imageHeight: number;
  platform: 'ios' | 'macos';
};

// --- Speech transcription (iOS 26+) ---

export type TranscriptionSegment = {
  text: string;
};

export type TranscriptionResult = {
  segments: TranscriptionSegment[];
};

export type ExpoAppleIntelligenceModuleEvents = Record<string, never>;
