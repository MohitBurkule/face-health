export interface Point3 { x: number; y: number; z?: number }

export interface FacialFatResult {
  fullnessIndex: number; // 0..1
  category: "low" | "medium" | "high";
  details: {
    widthHeightRatio: number;
    cheekPlumpness: number;
    jawTaper: number;
  };
}

function distance(a: Point3, b: Point3) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

// Robust helpers without relying on exact FaceMesh indices
function boundingBox(pts: Point3[]) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
}

export function estimateFacialAdiposity(landmarks: Point3[]): FacialFatResult {
  if (!landmarks.length) {
    return {
      fullnessIndex: 0,
      category: "low",
      details: { widthHeightRatio: 0, cheekPlumpness: 0, jawTaper: 0 },
    };
  }

  const box = boundingBox(landmarks);
  const widthHeightRatio = box.width / Math.max(1e-6, box.height);

  // Cheek plumpness proxy: average lateral deviation of mid-face vs centerline
  const midYTop = box.minY + box.height * 0.35;
  const midYBot = box.minY + box.height * 0.65;
  const slice = landmarks.filter((p) => p.y >= midYTop && p.y <= midYBot);
  const cx = (box.minX + box.maxX) / 2;
  let lateral = 0;
  for (const p of slice) lateral += Math.abs(p.x - cx);
  const cheekPlumpness = slice.length ? (lateral / slice.length) / Math.max(1e-6, box.width / 2) : 0;

  // Jaw taper: ratio of width in lower third vs upper third (smaller -> sharp jaw, larger -> rounder)
  const upperSlice = landmarks.filter((p) => p.y <= box.minY + box.height * 0.33);
  const lowerSlice = landmarks.filter((p) => p.y >= box.minY + box.height * 0.66);
  const upperBox = upperSlice.length ? boundingBox(upperSlice) : box;
  const lowerBox = lowerSlice.length ? boundingBox(lowerSlice) : box;
  const jawTaper = lowerBox.width / Math.max(1e-6, upperBox.width);

  // Normalize heuristics to 0..1 ranges based on plausible faces
  const rWidth = Math.max(0, Math.min(1, (widthHeightRatio - 0.7) / (1.3 - 0.7))); // typical 0.7..1.3
  const rCheek = Math.max(0, Math.min(1, (cheekPlumpness - 0.18) / (0.38 - 0.18))); // tuned empirically
  const rJaw = Math.max(0, Math.min(1, (jawTaper - 0.8) / (1.2 - 0.8)));

  const fullnessIndex = Math.max(0, Math.min(1, 0.45 * rWidth + 0.35 * rCheek + 0.2 * rJaw));
  const category = fullnessIndex < 0.33 ? "low" : fullnessIndex < 0.66 ? "medium" : "high";

  return {
    fullnessIndex,
    category,
    details: { widthHeightRatio, cheekPlumpness, jawTaper },
  };
}
