export interface Point2D { x: number; y: number; }
export interface Point3D extends Point2D { z?: number }

export interface FaceBox {
  minX: number; minY: number; maxX: number; maxY: number; width: number; height: number;
}

export interface EyeMetrics {
  points: Point3D[];
  center: Point2D;
  openness: number; // 0..1 relative to face height
}

export interface MouthMetrics {
  points: Point3D[];
  openRatio: number; // 0..1 relative to face height
}

export interface HeadPose {
  roll: number; // radians (in-plane rotation)
  yaw: number;  // -1..1 approx (negative = looking left)
  pitch: number; // -1..1 approx (negative = down)
}

export interface FaceMetricsResult {
  box: FaceBox;
  eyes: { left: EyeMetrics; right: EyeMetrics };
  mouth: MouthMetrics;
  head: HeadPose;
  jawline: { path: Point2D[] };
}

function boundingBox(pts: Point3D[]): FaceBox {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
}

function clamp01(v: number) { return Math.max(0, Math.min(1, v)); }

function percentile(sorted: number[], p: number) {
  if (!sorted.length) return 0;
  const idx = clamp01(p) * (sorted.length - 1);
  const l = Math.floor(idx), r = Math.ceil(idx);
  if (l === r) return sorted[l];
  const t = idx - l;
  return sorted[l] * (1 - t) + sorted[r] * t;
}

function mean(points: Point2D[]) {
  if (!points.length) return { x: 0, y: 0 };
  let sx = 0, sy = 0;
  for (const p of points) { sx += p.x; sy += p.y; }
  return { x: sx / points.length, y: sy / points.length };
}

function subset(pts: Point3D[], box: FaceBox, rx0: number, rx1: number, ry0: number, ry1: number) {
  const x0 = box.minX + box.width * rx0;
  const x1 = box.minX + box.width * rx1;
  const y0 = box.minY + box.height * ry0;
  const y1 = box.minY + box.height * ry1;
  return pts.filter(p => p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1);
}

function opennessRatio(points: Point3D[], faceH: number) {
  if (points.length < 4) return 0;
  const ys = points.map(p => p.y).sort((a,b) => a - b);
  const top = percentile(ys, 0.15);
  const bot = percentile(ys, 0.85);
  return clamp01((bot - top) / Math.max(1e-6, faceH));
}

function mouthOpenRatio(points: Point3D[], faceH: number) {
  if (!points.length) return 0;
  const ys = points.map(p => p.y).sort((a,b) => a - b);
  const top = percentile(ys, 0.2);
  const bot = percentile(ys, 0.9);
  return clamp01((bot - top) / Math.max(1e-6, faceH));
}

function estimateHeadPose(box: FaceBox, leftEye: Point2D, rightEye: Point2D): HeadPose {
  // Roll from eye line
  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;
  const roll = Math.atan2(dy, dx);

  // Yaw from relative eye x positions vs box center
  const cx = (box.minX + box.maxX) / 2;
  const leftDist = Math.abs(leftEye.x - box.minX);
  const rightDist = Math.abs(box.maxX - rightEye.x);
  const yaw = clamp01((rightDist - leftDist) / Math.max(1e-6, box.width)) as number;

  // Pitch from average eye height relative to box center
  const cy = (box.minY + box.maxY) / 2;
  const eyesY = (leftEye.y + rightEye.y) / 2;
  const pitch = clamp01((cy - eyesY) / Math.max(1e-6, box.height)) as number;

  // Map yaw/pitch to -1..1 ranges centered
  return {
    roll,
    yaw: Math.max(-1, Math.min(1, yaw * 2)),
    pitch: Math.max(-1, Math.min(1, pitch * 2)),
  };
}

function lowerJawPath(pts: Point3D[], box: FaceBox): Point2D[] {
  // Take lower third points and compute a simple x-sorted contour
  const lower = pts.filter(p => p.y >= box.minY + box.height * 0.66);
  const sorted = lower.sort((a,b) => a.x - b.x);
  // Thin by taking every Nth to avoid heavy drawing
  const step = Math.max(1, Math.floor(sorted.length / 40));
  const path: Point2D[] = [];
  for (let i = 0; i < sorted.length; i += step) path.push({ x: sorted[i].x, y: sorted[i].y });
  return path;
}

export function getFaceMetrics(landmarks: Point3D[]): FaceMetricsResult {
  if (!landmarks || landmarks.length === 0) {
    const emptyBox: FaceBox = { minX:0, minY:0, maxX:0, maxY:0, width:0, height:0 };
    return {
      box: emptyBox,
      eyes: { left: { points: [], center: { x:0, y:0 }, openness: 0 }, right: { points: [], center: { x:0, y:0 }, openness: 0 } },
      mouth: { points: [], openRatio: 0 },
      head: { roll: 0, yaw: 0, pitch: 0 },
      jawline: { path: [] },
    };
  }

  const box = boundingBox(landmarks);

  // Derive ROIs
  const leftEyePts = subset(landmarks, box, 0.12, 0.46, 0.25, 0.55);
  const rightEyePts = subset(landmarks, box, 0.54, 0.88, 0.25, 0.55);
  const mouthPts = subset(landmarks, box, 0.25, 0.75, 0.60, 1.00);

  const leftCenter = mean(leftEyePts);
  const rightCenter = mean(rightEyePts);

  const leftOpen = opennessRatio(leftEyePts, box.height);
  const rightOpen = opennessRatio(rightEyePts, box.height);
  const mar = mouthOpenRatio(mouthPts, box.height);

  const head = estimateHeadPose(box, leftCenter, rightCenter);

  const jaw = lowerJawPath(landmarks, box);

  return {
    box,
    eyes: {
      left: { points: leftEyePts, center: leftCenter, openness: leftOpen },
      right: { points: rightEyePts, center: rightCenter, openness: rightOpen },
    },
    mouth: { points: mouthPts, openRatio: mar },
    head,
    jawline: { path: jaw },
  };
}
