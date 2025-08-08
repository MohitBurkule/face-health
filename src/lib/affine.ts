export interface P2 { x: number; y: number }

// Solve 2D similarity transform (a,b,c,d,tx,ty) mapping src->dst for 3 points
// x' = a x + b y + tx
// y' = c x + d y + ty
export function estimateSimilarity2D(src: P2[], dst: P2[]): [number, number, number, number, number, number] {
  if (src.length < 3 || dst.length < 3) return [1, 0, 0, 1, 0, 0];
  const [s0, s1, s2] = src;
  const [d0, d1, d2] = dst;

  // Build 6x6 system
  // [ x y 0 0 1 0 ] [a] = [X]
  // [ 0 0 x y 0 1 ] [b]   [Y]
  const A = [
    [s0.x, s0.y, 0, 0, 1, 0],
    [0, 0, s0.x, s0.y, 0, 1],
    [s1.x, s1.y, 0, 0, 1, 0],
    [0, 0, s1.x, s1.y, 0, 1],
    [s2.x, s2.y, 0, 0, 1, 0],
    [0, 0, s2.x, s2.y, 0, 1],
  ];
  const b = [d0.x, d0.y, d1.x, d1.y, d2.x, d2.y];

  // Solve via Gaussian elimination (small system)
  const M = A.map((row, i) => [...row, b[i]]);
  const n = 6;
  for (let col = 0; col < n; col++) {
    // pivot
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    if (Math.abs(M[pivot][col]) < 1e-8) continue;
    if (pivot !== col) [M[col], M[pivot]] = [M[pivot], M[col]];

    // normalize
    const div = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= div;

    // eliminate
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }

  const a = M[0][n];
  const b1 = M[1][n];
  const c = M[2][n];
  const d = M[3][n];
  const tx = M[4][n];
  const ty = M[5][n];
  return [a, b1, c, d, tx, ty];
}
