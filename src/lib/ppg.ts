import FFT from "fft.js";

export interface HRResult {
  bpm: number | null;
  confidence: number; // 0..1
  sampleRate: number | null;
  windowSeconds: number;
}

// Hann window
function hann(N: number) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
  return w;
}

function mean(arr: number[]) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function detrend(data: number[], span = 15) {
  // simple moving average detrend
  const out = new Float32Array(data.length);
  const half = Math.max(1, Math.floor(span / 2));
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - half);
    const end = Math.min(data.length - 1, i + half);
    const m = mean(data.slice(start, end + 1));
    out[i] = data[i] - m;
  }
  return out;
}

function nearestPow2(n: number) {
  return 1 << (32 - Math.clz32(n - 1));
}

export function computeHeartRate(
  samples: { t: number; g: number }[],
  opts?: { minBpm?: number; maxBpm?: number }
): HRResult {
  const minBpm = opts?.minBpm ?? 42; // 0.7 Hz
  const maxBpm = opts?.maxBpm ?? 180; // 3.0 Hz

  if (samples.length < 64) return { bpm: null, confidence: 0, sampleRate: null, windowSeconds: 0 };

  // estimate sample rate from timestamps (milliseconds)
  const dts: number[] = [];
  for (let i = 1; i < samples.length; i++) dts.push((samples[i].t - samples[i - 1].t) / 1000);
  const avgDt = mean(dts);
  if (!isFinite(avgDt) || avgDt <= 0) return { bpm: null, confidence: 0, sampleRate: null, windowSeconds: 0 };
  const fs = 1 / avgDt;

  // build signal
  const signal = samples.map((s) => s.g);

  // normalize
  const mu = mean(signal);
  for (let i = 0; i < signal.length; i++) signal[i] -= mu;

  // detrend + window
  const detr = detrend(signal, Math.round(0.5 * fs));
  const N0 = detr.length;
  const N = nearestPow2(N0);
  const padded = new Float32Array(N);
  const win = hann(N);
  const scale = 1 / N; // normalize
  for (let i = 0; i < N; i++) {
    const v = i < N0 ? detr[i] : 0;
    padded[i] = v * win[i] * scale;
  }

  const f = new FFT(N);
  const out = f.createComplexArray();
  f.realTransform(out, padded);
  f.completeSpectrum(out);

  // compute magnitude spectrum for positive freqs
  const mags = new Float32Array(N / 2);
  for (let i = 0; i < N / 2; i++) {
    const re = out[2 * i];
    const im = out[2 * i + 1];
    mags[i] = Math.hypot(re, im);
  }

  // frequency per bin
  const binToHz = fs / N;
  const minBin = Math.max(1, Math.floor((minBpm / 60) / binToHz));
  const maxBin = Math.min(N / 2 - 1, Math.ceil((maxBpm / 60) / binToHz));

  let peakBin = -1;
  let peakVal = 0;
  for (let i = minBin; i <= maxBin; i++) {
    const v = mags[i];
    if (v > peakVal) {
      peakVal = v;
      peakBin = i;
    }
  }

  if (peakBin < 0) return { bpm: null, confidence: 0, sampleRate: fs, windowSeconds: samples.length / fs };

  const peakHz = peakBin * binToHz;
  const bpm = peakHz * 60;

  // crude confidence: peak prominence vs band energy
  const bandEnergy = mags.slice(minBin, maxBin + 1).reduce((a, b) => a + b, 0);
  const confidence = Math.max(0, Math.min(1, bandEnergy ? peakVal / (bandEnergy / (maxBin - minBin + 1)) : 0));

  return { bpm, confidence, sampleRate: fs, windowSeconds: samples.length / fs };
}
