import FFT from "fft.js";

export interface RespirationResult {
  bpm: number | null; // breaths per minute
  confidence: number; // 0..1
  sampleRate: number | null;
  windowSeconds: number;
}

export interface HRVResult {
  rmssd: number | null; // ms
  sdnn: number | null;  // ms
  ibiMs: number | null; // mean inter-beat interval
  beats: number;        // number of detected peaks
}

function mean(arr: number[]) { return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0; }
function std(arr: number[]) {
  if (!arr.length) return 0;
  const m = mean(arr);
  const v = mean(arr.map(x => (x - m) * (x - m)));
  return Math.sqrt(v);
}

function detrend(data: number[], span = 25) {
  const out = new Float32Array(data.length);
  const half = Math.max(1, Math.floor(span / 2));
  for (let i = 0; i < data.length; i++) {
    const s = Math.max(0, i - half);
    const e = Math.min(data.length - 1, i + half);
    let acc = 0, n = 0;
    for (let j = s; j <= e; j++) { acc += data[j]; n++; }
    out[i] = data[i] - (n ? acc / n : 0);
  }
  return out;
}

function nearestPow2(n: number) { return 1 << (32 - Math.clz32(Math.max(2, n) - 1)); }

export function estimateRespirationRate(samples: { t: number; g: number }[], opts?: { minBpm?: number; maxBpm?: number }): RespirationResult {
  const minBpm = opts?.minBpm ?? 6;   // 0.1 Hz
  const maxBpm = opts?.maxBpm ?? 30;  // 0.5 Hz
  if (samples.length < 64) return { bpm: null, confidence: 0, sampleRate: null, windowSeconds: 0 };

  // sample rate
  const dts: number[] = [];
  for (let i = 1; i < samples.length; i++) dts.push((samples[i].t - samples[i-1].t) / 1000);
  const dt = mean(dts);
  if (!isFinite(dt) || dt <= 0) return { bpm: null, confidence: 0, sampleRate: null, windowSeconds: 0 };
  const fs = 1 / dt;

  // signal
  const sig = samples.map(s => s.g);
  const m = mean(sig);
  for (let i = 0; i < sig.length; i++) sig[i] -= m;
  const detr = detrend(sig, Math.round(fs));

  // FFT
  const N0 = detr.length;
  const N = nearestPow2(N0);
  const pad = new Float32Array(N);
  for (let i = 0; i < N; i++) pad[i] = i < N0 ? detr[i] : 0;
  const fft = new FFT(N);
  const out = fft.createComplexArray();
  fft.realTransform(out, pad);
  fft.completeSpectrum(out);

  const mags = new Float32Array(N/2);
  for (let i = 0; i < N/2; i++) {
    const re = out[2*i], im = out[2*i+1];
    mags[i] = Math.hypot(re, im);
  }

  const binToHz = fs / N;
  const minBin = Math.max(1, Math.floor((minBpm/60) / binToHz));
  const maxBin = Math.min(N/2 - 1, Math.ceil((maxBpm/60) / binToHz));

  let peakBin = -1, peakVal = 0;
  for (let i = minBin; i <= maxBin; i++) {
    const v = mags[i];
    if (v > peakVal) { peakVal = v; peakBin = i; }
  }
  if (peakBin < 0) return { bpm: null, confidence: 0, sampleRate: fs, windowSeconds: samples.length / fs };

  const bandEnergy = mags.slice(minBin, maxBin+1).reduce((a,b)=>a+b,0);
  const confidence = Math.max(0, Math.min(1, bandEnergy ? peakVal / (bandEnergy / (maxBin - minBin + 1)) : 0));
  const hz = peakBin * binToHz;
  return { bpm: hz * 60, confidence, sampleRate: fs, windowSeconds: samples.length / fs };
}

export function estimateHRV(samples: { t: number; g: number }[]): HRVResult {
  if (samples.length < 128) return { rmssd: null, sdnn: null, ibiMs: null, beats: 0 };

  // Prepare signal (bandpass-ish via detrend + smooth)
  const times = samples.map(s => s.t / 1000);
  const sig = samples.map(s => s.g);
  const m = mean(sig);
  for (let i = 0; i < sig.length; i++) sig[i] -= m;
  const detr = detrend(sig, 15);
  // Simple moving average to smooth noise
  const smoothed = new Float32Array(detr.length);
  const k = 3;
  for (let i = 0; i < detr.length; i++) {
    let acc = 0, n = 0;
    for (let j = -k; j <= k; j++) {
      const idx = i + j;
      if (idx >= 0 && idx < detr.length) { acc += detr[idx]; n++; }
    }
    smoothed[i] = n ? acc / n : detr[i];
  }

  // Peak detection
  const mu = mean(Array.from(smoothed));
  const sd = std(Array.from(smoothed));
  const thr = mu + sd * 0.3; // adaptive
  const peakIdxs: number[] = [];
  for (let i = 1; i < smoothed.length - 1; i++) {
    if (smoothed[i] > thr && smoothed[i] >= smoothed[i-1] && smoothed[i] > smoothed[i+1]) {
      peakIdxs.push(i);
    }
  }
  if (peakIdxs.length < 3) return { rmssd: null, sdnn: null, ibiMs: null, beats: peakIdxs.length };

  const ibis: number[] = [];
  for (let i = 1; i < peakIdxs.length; i++) {
    const t0 = times[peakIdxs[i-1]];
    const t1 = times[peakIdxs[i]];
    ibis.push((t1 - t0) * 1000); // ms
  }
  if (!ibis.length) return { rmssd: null, sdnn: null, ibiMs: null, beats: peakIdxs.length };

  const diffs: number[] = [];
  for (let i = 1; i < ibis.length; i++) diffs.push(ibis[i] - ibis[i-1]);

  const rmssd = diffs.length ? Math.sqrt(mean(diffs.map(d => d*d))) : null;
  const sdnn = ibis.length ? std(ibis) : null;
  const ibiMs = ibis.length ? mean(ibis) : null;

  return { rmssd, sdnn, ibiMs, beats: peakIdxs.length };
}
