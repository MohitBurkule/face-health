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
  center: Point2D;
  openRatio: number; // MAR-based ratio (vertical/horizontal)
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
  // Prefer MediaPipe face oval indices when available
  const hasMesh = pts.length >= 400;
  if (hasMesh) {
    // Face oval indices (MediaPipe 468 mesh)
    const OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
    const ovalPts = OVAL.map(i => pts[i]).filter(Boolean) as Point3D[];
    if (ovalPts.length) {
      const ys = ovalPts.map(p => p.y).sort((a,b) => a - b);
      const thr = percentile(ys, 0.55); // keep the lowermost half of oval
      const jaw = OVAL
        .map(i => pts[i])
        .filter((p): p is Point3D => !!p && p.y >= thr)
        .map(p => ({ x: p.x, y: p.y }));
      return jaw;
    }
  }
  // Fallback: take the globally lowest points and sort by x
  const candidates = pts
    .slice()
    .sort((a,b) => b.y - a.y) // lowest (largest y) first
    .slice(0, Math.max(12, Math.floor(pts.length * 0.06))); // ~lowest 6%
  const sorted = candidates.sort((a,b) => a.x - b.x);
  return sorted.map(p => ({ x: p.x, y: p.y }));
}

export function getFaceMetrics(landmarks: Point3D[]): FaceMetricsResult {
  if (!landmarks || landmarks.length === 0) {
    const emptyBox: FaceBox = { minX:0, minY:0, maxX:0, maxY:0, width:0, height:0 };
    return {
      box: emptyBox,
      eyes: { left: { points: [], center: { x:0, y:0 }, openness: 0 }, right: { points: [], center: { x:0, y:0 }, openness: 0 } },
      mouth: { points: [], center: { x:0, y:0 }, openRatio: 0 },
      head: { roll: 0, yaw: 0, pitch: 0 },
      jawline: { path: [] },
    };
  }

  const box = boundingBox(landmarks);

  // Prefer MediaPipe index-based EAR/MAR when landmarks length is sufficient
  const hasMesh = landmarks.length >= 400; // 468/478 mesh
  function dist(i: number, j: number) {
    const a = landmarks[i] || { x: 0, y: 0 } as Point3D;
    const b = landmarks[j] || { x: 0, y: 0 } as Point3D;
    const dx = (a.x - b.x);
    const dy = (a.y - b.y);
    return Math.hypot(dx, dy);
  }

  let leftEyePts: Point3D[] = [];
  let rightEyePts: Point3D[] = [];
  let mouthPts: Point3D[] = [];
  let leftOpen = 0, rightOpen = 0, mar = 0;

  if (hasMesh) {
    // MediaPipe landmark indices (468 mesh)
    const L_H = [33, 133];
    const L_V1 = [159, 145];
    const L_V2 = [158, 153];
    const R_H = [263, 362];
    const R_V1 = [386, 374];
    const R_V2 = [385, 380];

    const M_H = [61, 291];
    const M_V1 = [13, 14];

    leftEyePts = [...L_H, ...L_V1, ...L_V2].map(i => landmarks[i]).filter(Boolean) as Point3D[];
    rightEyePts = [...R_H, ...R_V1, ...R_V2].map(i => landmarks[i]).filter(Boolean) as Point3D[];
    mouthPts = [...M_H, ...M_V1].map(i => landmarks[i]).filter(Boolean) as Point3D[];

    const lEar = (dist(L_V1[0], L_V1[1]) + dist(L_V2[0], L_V2[1])) / (2 * Math.max(1e-6, dist(L_H[0], L_H[1])));
    const rEar = (dist(R_V1[0], R_V1[1]) + dist(R_V2[0], R_V2[1])) / (2 * Math.max(1e-6, dist(R_H[0], R_H[1])));
    leftOpen = lEar; rightOpen = rEar;

    mar = dist(M_V1[0], M_V1[1]) / Math.max(1e-6, dist(M_H[0], M_H[1]));
  } else {
    // Fallback ROI-based metrics
    leftEyePts = subset(landmarks, box, 0.12, 0.46, 0.25, 0.55);
    rightEyePts = subset(landmarks, box, 0.54, 0.88, 0.25, 0.55);
    mouthPts = subset(landmarks, box, 0.25, 0.75, 0.60, 1.00);
    leftOpen = opennessRatio(leftEyePts, box.height);
    rightOpen = opennessRatio(rightEyePts, box.height);
    mar = mouthOpenRatio(mouthPts, box.height);
  }

  const leftCenter = mean(leftEyePts);
  const rightCenter = mean(rightEyePts);
  const mouthCenter = mean(mouthPts);

  const head = estimateHeadPose(box, leftCenter, rightCenter);

  const jaw = lowerJawPath(landmarks, box);

  return {
    box,
    eyes: {
      left: { points: leftEyePts, center: leftCenter, openness: leftOpen },
      right: { points: rightEyePts, center: rightCenter, openness: rightOpen },
    },
    mouth: { points: mouthPts, center: mouthCenter, openRatio: mar },
    head,
    jawline: { path: jaw },
  };
}
