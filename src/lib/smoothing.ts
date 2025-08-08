export interface P2 { x: number; y: number }

// Chaikin's corner-cutting algorithm to smooth polylines
export function smoothChaikin(points: P2[] = [], iterations = 2): P2[] {
  if (!points || points.length < 3) return points ? points.slice() : [];
  let pts = points.slice();
  for (let it = 0; it < iterations; it++) {
    const out: P2[] = [pts[0]]; // keep endpoints
    for (let i = 0; i < pts.length - 1; i++) {
      const p = pts[i];
      const q = pts[i + 1];
      // Q and R points
      out.push({ x: 0.75 * p.x + 0.25 * q.x, y: 0.75 * p.y + 0.25 * q.y });
      out.push({ x: 0.25 * p.x + 0.75 * q.x, y: 0.25 * p.y + 0.75 * q.y });
    }
    out.push(pts[pts.length - 1]);
    pts = out;
  }
  return pts;
}
