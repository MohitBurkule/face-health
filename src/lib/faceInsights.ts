import type { FaceMetricsResult } from "./faceMetrics";

export interface FaceInsightsSnapshot {
  earLeft: number;
  earRight: number;
  earAvg: number;
  blink: boolean;
  blinkRatePerMin: number;
  perclos: number; // 0..1 over last window
  mar: number;
  yawnProbability: number; // 0..1
}

interface TrackerState {
  closed: boolean;
  lastStateChange: number;
  blinkTimes: number[]; // timestamps (ms)
  closedSpans: { start: number; end: number }[];
  windowMs: number;
  baselineEar: number | null;
}

export function createInsightsTracker(opts?: { windowMs?: number }) {
  const state: TrackerState = {
    closed: false,
    lastStateChange: 0,
    blinkTimes: [],
    closedSpans: [],
    windowMs: opts?.windowMs ?? 60000, // 60s
    baselineEar: null,
  };

  function clamp01(v: number) { return Math.max(0, Math.min(1, v)); }

  function cleanup(now: number) {
    const cutoff = now - state.windowMs;
    state.blinkTimes = state.blinkTimes.filter(t => t >= cutoff);
    state.closedSpans = state.closedSpans.filter(s => s.end >= cutoff || s.start >= cutoff);
  }

  function perclos(now: number) {
    const cutoff = now - state.windowMs;
    let closedMs = 0;
    // include ongoing span if closed
    const spans = state.closedSpans.slice();
    if (state.closed && state.lastStateChange > cutoff) spans.push({ start: state.lastStateChange, end: now });
    for (const s of spans) {
      const start = Math.max(s.start, cutoff);
      const end = Math.max(start, s.end);
      closedMs += Math.max(0, end - start);
    }
    return clamp01(closedMs / state.windowMs);
  }

  function snapshot(now: number, earL: number, earR: number, mar: number, blinkEvent: boolean): FaceInsightsSnapshot {
    const earAvg = (earL + earR) / 2;
    const rate = state.blinkTimes.length * (60000 / state.windowMs);
    // yawn probability based on MAR
    const yawnProb = clamp01((mar - 0.25) / 0.25); // 0 at 0.25, 1 at 0.5
    return {
      earLeft: earL,
      earRight: earR,
      earAvg,
      blink: blinkEvent,
      blinkRatePerMin: rate,
      perclos: perclos(now),
      mar,
      yawnProbability: yawnProb,
    };
  }

  return {
    update(metrics: FaceMetricsResult, nowMs: number) {
      const now = nowMs;
      const earL = metrics.eyes.left.openness; // already normalized 0..1 to face height
      const earR = metrics.eyes.right.openness;
      const mar = metrics.mouth.openRatio; // 0..1 to face height

      // adaptive threshold from baseline
      const earAvg = (earL + earR) / 2;
      if (state.baselineEar == null) state.baselineEar = earAvg;
      // slowly update baseline toward current
      state.baselineEar = 0.98 * state.baselineEar + 0.02 * earAvg;
      const thresh = Math.max(0.08, Math.min(0.6, (state.baselineEar ?? 0.25) * 0.55));

      let blinkEvent = false;
      // state transitions
      if (!state.closed && earAvg < thresh) {
        state.closed = true;
        state.lastStateChange = now;
      } else if (state.closed && earAvg >= thresh) {
        const start = state.lastStateChange;
        const duration = now - start;
        // consider blink if short closure (80-500 ms)
        if (duration >= 80 && duration <= 500) {
          state.blinkTimes.push(now);
        }
        state.closedSpans.push({ start, end: now });
        state.closed = false;
        state.lastStateChange = now;
        blinkEvent = duration >= 80 && duration <= 500;
      }

      cleanup(now);
      return snapshot(now, earL, earR, mar, blinkEvent);
    },
  };
}
