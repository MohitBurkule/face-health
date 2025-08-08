import React, { useEffect, useMemo, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { estimateFacialAdiposity } from "@/lib/facialFat";
import { computeHeartRate } from "@/lib/ppg";
import { getFaceMetrics } from "@/lib/faceMetrics";
import { smoothChaikin, savitzkyGolay } from "@/lib/smoothing";
import { estimateRespirationRate, estimateHRV } from "@/lib/ppgExtras";
import { createInsightsTracker } from "@/lib/faceInsights";
import { estimateSimilarity2D } from "@/lib/affine";

// MediaPipe Tasks Vision types come from the package at runtime; we keep TS light here
// to avoid depending on their types directly.
let FaceLandmarker: any;
let FilesetResolver: any;

const TASKS_VERSION = "0.10.8";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}/wasm`;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

interface Sample { t: number; g: number }

export default function FaceAnalyzer() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const frontRef = useRef<HTMLCanvasElement | null>(null);
  const hrRef = useRef<HTMLCanvasElement | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [bpm, setBpm] = useState<number | null>(null);
  const [hrConfidence, setHrConfidence] = useState(0);
  const [fullness, setFullness] = useState(0);
  const [fatCategory, setFatCategory] = useState<"low" | "medium" | "high">("low");
  const [signalQuality, setSignalQuality] = useState(0);

  const [blinkRate, setBlinkRate] = useState<number | null>(null);
  const [perclos, setPerclos] = useState(0);
  const [yawnProb, setYawnProb] = useState(0);
  const [respRate, setRespRate] = useState<number | null>(null);
  const [rmssd, setRmssd] = useState<number | null>(null);

  const samplesRef = useRef<Sample[]>([]);
  const smoothCtrlRef = useRef<{lx:number;ly:number;rx:number;ry:number;mx:number;my:number}|null>(null);
  const smoothBoxRef = useRef<{x:number;y:number;w:number;h:number}|null>(null);
  const prevPointsRef = useRef<{x:number;y:number}[]|null>(null);
  const prevAnchorsRef = useRef<{L:{x:number;y:number};R:{x:number;y:number};M:{x:number;y:number}}|null>(null);
  const [motionSeries, setMotionSeries] = useState<{t:number;v:number}[]>([]);
  const [motionNow, setMotionNow] = useState(0);
  const [jawRecording, setJawRecording] = useState(false);
  const jawDatasetRef = useRef<any[]>([]);
  const lastJawSaveRef = useRef(0);
  const landmarkerRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const hiddenCanvas = useMemo(() => document.createElement("canvas"), []);
  const hiddenCtx = useMemo(() => hiddenCanvas.getContext("2d", { willReadFrequently: true }), [hiddenCanvas]);
  const insightsRef = useRef(createInsightsTracker());

  // Signature interaction: reactive gradient position
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const x = (e.clientX / window.innerWidth) * 100;
      const y = (e.clientY / window.innerHeight) * 100;
      document.documentElement.style.setProperty("--mx", `${x}%`);
      document.documentElement.style.setProperty("--my", `${y}%`);
    };
    window.addEventListener("pointermove", onMove);
    return () => window.removeEventListener("pointermove", onMove);
  }, []);

  // Initialize camera + model
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (!active) return;
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play();

        // Load model
        const visionMod = await import("@mediapipe/tasks-vision");
        FilesetResolver = visionMod.FilesetResolver;
        FaceLandmarker = visionMod.FaceLandmarker;
        const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
        landmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL },
      numFaces: 1,
      runningMode: "VIDEO",
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: true,
    });

        setStreaming(true);
        setInitializing(false);
        startLoop();
      } catch (e) {
        console.error("Camera/Model init failed", e);
        setInitializing(false);
      }
    })();

    return () => {
      active = false;
      stopLoop();
      const v = videoRef.current;
      if (v?.srcObject) {
        (v.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
        v.srcObject = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startLoop = () => {
    const tick = () => {
      processFrame();
      rafRef.current = requestAnimationFrame(tick);
    };
    if (!rafRef.current) rafRef.current = requestAnimationFrame(tick);
  };
  const stopLoop = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  };

  function processFrame() {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    const ctx = canvas?.getContext("2d");
    const lm = landmarkerRef.current;
    if (!video || !canvas || !ctx || !lm || video.readyState < 2) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ts = performance.now();
    const res = lm.detectForVideo(video, ts);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (res?.faceLandmarks?.length) {
      const points = res.faceLandmarks[0] as { x: number; y: number; z?: number }[];

      // Draw minimal overlay
      ctx.lineWidth = 2;
      ctx.strokeStyle = "hsl(var(--ring))";
      ctx.fillStyle = "hsl(var(--primary) / 0.2)" as any;

      // Bounds (used for ROI and stabilization)
      const box = getBounds(points, canvas.width, canvas.height);
      // no visual face bounding box

      // Draw all landmark points
      ctx.fillStyle = "hsl(var(--primary))" as any;
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, 1.3, 0, Math.PI * 2);
        ctx.fill();
      }

      // Compute contours and metrics
      const m = getFaceMetrics(points as any);
      // Update insights tracker (blinks, PERCLOS, yawn)
      const snap = insightsRef.current.update(m, ts);
      setBlinkRate(snap.blinkRatePerMin ?? null);
      setPerclos(snap.perclos);
      setYawnProb(snap.yawnProbability);

      // Draw jawline path
      if (m.jawline.path.length) {
        const chaikin = smoothChaikin(m.jawline.path as any, 2);
        const smoothed = savitzkyGolay(chaikin as any, 7);
        ctx.strokeStyle = "hsl(var(--muted-foreground))";
        ctx.beginPath();
        smoothed.forEach((p, idx) => {
          const px = p.x * canvas.width;
          const py = p.y * canvas.height;
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      // Highlight eye points
      ctx.fillStyle = "hsl(var(--accent))" as any;
      for (const ep of [...m.eyes.left.points, ...m.eyes.right.points]) {
        ctx.beginPath();
        ctx.arc(ep.x * canvas.width, ep.y * canvas.height, 1.6, 0, Math.PI * 2);
        ctx.fill();
      }

      // Facial motion metric (relative landmark movement)
      {
        const currPx = points.map((p) => ({ x: p.x * canvas.width, y: p.y * canvas.height }));
        const anchorsCurr = {
          L: { x: m.eyes.left.center.x * canvas.width, y: m.eyes.left.center.y * canvas.height },
          R: { x: m.eyes.right.center.x * canvas.width, y: m.eyes.right.center.y * canvas.height },
          M: { x: m.mouth.center.x * canvas.width, y: m.mouth.center.y * canvas.height },
        };
        const prevPts = prevPointsRef.current;
        const prevAnch = prevAnchorsRef.current;
        if (prevPts && prevAnch && prevPts.length === currPx.length) {
          const Tm = estimateSimilarity2D([prevAnch.L, prevAnch.R, prevAnch.M], [anchorsCurr.L, anchorsCurr.R, anchorsCurr.M]);
          const apply = (x: number, y: number) => ({
            x: Tm[0] * x + Tm[2] * y + Tm[4],
            y: Tm[1] * x + Tm[3] * y + Tm[5],
          });
          let sum = 0;
          for (let i = 0; i < currPx.length; i++) {
            const p0 = apply(prevPts[i].x, prevPts[i].y);
            const dx = currPx[i].x - p0.x;
            const dy = currPx[i].y - p0.y;
            sum += dx * dx + dy * dy;
          }
          const rms = Math.sqrt(sum / currPx.length) / Math.max(1, box.h);
          setMotionNow(rms);
          setMotionSeries((s) => {
            const ns = [...s, { t: ts, v: rms }];
            while (ns.length > 300) ns.shift();
            return ns;
          });
        }
        prevPointsRef.current = currPx;
        prevAnchorsRef.current = anchorsCurr;
      }
      // Frontalized view (top-right canvas)
      const front = frontRef.current;
      if (front) {
        const fctx = front.getContext("2d");
        if (fctx) {
          const targetW = 160, targetH = 160;
          if (front.width !== targetW || front.height !== targetH) {
            front.width = targetW; front.height = targetH;
          }
          fctx.clearRect(0, 0, front.width, front.height);

          // Source control points (pixels) with light smoothing for stability
          let lx = m.eyes.left.center.x * canvas.width;
          let ly = m.eyes.left.center.y * canvas.height;
          let rx = m.eyes.right.center.x * canvas.width;
          let ry = m.eyes.right.center.y * canvas.height;
          let mx = m.mouth.center.x * canvas.width;
          let my = m.mouth.center.y * canvas.height;

          const prevCtrl = smoothCtrlRef.current;
          const alpha = 0.8;
          if (prevCtrl) {
            lx = alpha * prevCtrl.lx + (1 - alpha) * lx;
            ly = alpha * prevCtrl.ly + (1 - alpha) * ly;
            rx = alpha * prevCtrl.rx + (1 - alpha) * rx;
            ry = alpha * prevCtrl.ry + (1 - alpha) * ry;
            mx = alpha * prevCtrl.mx + (1 - alpha) * mx;
            my = alpha * prevCtrl.my + (1 - alpha) * my;
          }
          smoothCtrlRef.current = { lx, ly, rx, ry, mx, my };

          // Canonical destination points
          const dst = [
            { x: targetW * 0.32, y: targetH * 0.40 }, // left eye
            { x: targetW * 0.68, y: targetH * 0.40 }, // right eye
            { x: targetW * 0.50, y: targetH * 0.75 }, // mouth
          ];

          const src = [ { x: lx, y: ly }, { x: rx, y: ry }, { x: mx, y: my } ];
          const T = estimateSimilarity2D(src, dst);

          fctx.save();
          fctx.setTransform(T[0], T[2], T[1], T[3], T[4], T[5]);
          fctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          fctx.restore();

          // Border
          fctx.strokeStyle = "hsl(var(--ring))";
          fctx.strokeRect(0, 0, front.width, front.height);
        }
      }

      // Heart-rate ROI: forehead band (top 15% of face box) with smoothing
      const roi = {
        x: Math.round(box.x + box.w * 0.3),
        y: Math.round(box.y + box.h * 0.08),
        w: Math.round(box.w * 0.4),
        h: Math.round(box.h * 0.12),
      };
      let sroi = roi;
      const pb = smoothBoxRef.current;
      const aR = 0.85;
      if (pb) {
        sroi = {
          x: Math.round(aR * pb.x + (1 - aR) * roi.x),
          y: Math.round(aR * pb.y + (1 - aR) * roi.y),
          w: Math.round(aR * pb.w + (1 - aR) * roi.w),
          h: Math.round(aR * pb.h + (1 - aR) * roi.h),
        };
      }
      smoothBoxRef.current = sroi;

      // ROI visual disabled

      // Sample mean green from ROI
      if (hiddenCtx) {
        hiddenCanvas.width = sroi.w;
        hiddenCanvas.height = sroi.h;
        hiddenCtx.drawImage(
          video,
          sroi.x,
          sroi.y,
          sroi.w,
          sroi.h,
          0,
          0,
          sroi.w,
          sroi.h
        );
        // Show ROI preview below frontalized crop
        const hrCanvas = hrRef.current;
        if (hrCanvas) {
          const hctx = hrCanvas.getContext("2d");
          const targetW = 160, targetH = 64;
          if (hrCanvas.width !== targetW || hrCanvas.height !== targetH) {
            hrCanvas.width = targetW; hrCanvas.height = targetH;
          }
          hctx?.clearRect(0, 0, hrCanvas.width, hrCanvas.height);
          hctx?.drawImage(video, sroi.x, sroi.y, sroi.w, sroi.h, 0, 0, hrCanvas.width, hrCanvas.height);
        }
        const img = hiddenCtx.getImageData(0, 0, sroi.w, sroi.h);
        const data = img.data;
        let sumG = 0;
        const step = 4 * 4; // stride to reduce work
        for (let i = 1; i < data.length; i += step) sumG += data[i];
        const samples = data.length / step;
        const meanG = sumG / Math.max(1, samples);
        samplesRef.current.push({ t: performance.now(), g: meanG });

        // keep last 20s of data max
        const cutoff = performance.now() - 20000;
        while (samplesRef.current.length && samplesRef.current[0].t < cutoff) {
          samplesRef.current.shift();
        }

        // Compute HR every ~1s
        if (samplesRef.current.length > 64 && (!lastHRCompute || ts - lastHRCompute > 1000)) {
          lastHRCompute = ts;
          const hr = computeHeartRate(samplesRef.current, { minBpm: 45, maxBpm: 170 });
          setBpm(hr.bpm ? Math.round(hr.bpm) : null);
          setHrConfidence(hr.confidence);

          // Simple signal quality: normalized variance
          const gVals = samplesRef.current.map((s) => s.g);
          const mu = gVals.reduce((a, b) => a + b, 0) / gVals.length;
          const variance = gVals.reduce((a, b) => a + (b - mu) * (b - mu), 0) / gVals.length;
          const sq = Math.max(0, Math.min(1, variance / 100));
          setSignalQuality(sq);

          // Respiration (every ~2s)
          if (!lastRespCompute || ts - lastRespCompute > 2000) {
            const rr = estimateRespirationRate(samplesRef.current);
            setRespRate(rr.bpm ? Math.round(rr.bpm) : null);
            lastRespCompute = ts;
          }

          // HRV (every ~5s)
          if (!lastHRVCompute || ts - lastHRVCompute > 5000) {
            const hrv = estimateHRV(samplesRef.current);
            setRmssd(hrv.rmssd ?? null);
            lastHRVCompute = ts;
          }
        }
      }

      // Optional: record jaw-opening training samples
      if (jawRecording) {
        if (!lastJawSaveRef.current || ts - lastJawSaveRef.current > 200) {
          jawDatasetRef.current.push({ t: ts, mar: m.mouth.openRatio, yaw: m.head.yaw, pitch: m.head.pitch });
          lastJawSaveRef.current = ts;
        }
      }

      // Facial fat estimate (lightweight heuristics)
      const adip = estimateFacialAdiposity(points);
      setFullness(adip.fullnessIndex);
      setFatCategory(adip.category);
    }
  }

  let lastHRCompute = 0;
  let lastRespCompute = 0;
  let lastHRVCompute = 0;

  const confidenceBadge = hrConfidence > 0.66 ? "High" : hrConfidence > 0.33 ? "Medium" : "Low";
  const confVariant = hrConfidence > 0.66 ? "default" : hrConfidence > 0.33 ? "secondary" : "outline";

  return (
    <div className="w-full grid gap-6">
      <div className="space-y-2">
        <h1 className="text-3xl md:text-4xl font-bold tracking-tight">Face Keypoint & Vital Analyzer</h1>
        <p className="text-muted-foreground">Real‑time face landmarks, remote heart rate (rPPG), and facial fullness heuristics from your camera. Not medical advice.</p>
      </div>

      <Card className="p-4 md:p-6">
        <div className="relative w-full aspect-video rounded-lg overflow-hidden ring-1 ring-border">
          <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover" playsInline muted />
          <canvas ref={overlayRef} className="absolute inset-0 w-full h-full" />
          <canvas
            ref={frontRef}
            className="absolute top-3 right-3 w-40 h-40 rounded-md bg-background/70 backdrop-blur-sm ring-1 ring-border shadow-sm"
            aria-label="Frontalized face preview"
          />
          <canvas
            ref={hrRef}
            className="absolute top-44 right-3 w-40 h-20 rounded-md bg-background/70 backdrop-blur-sm ring-1 ring-border shadow-sm"
            aria-label="Heart-rate ROI preview"
          />
          {!streaming && (
            <div className="absolute inset-0 grid place-items-center bg-background/80">
              <p className="text-sm text-muted-foreground">{initializing ? "Initializing camera & model…" : "Camera unavailable"}</p>
            </div>
          )}
        </div>

        <div className="mt-4 flex items-center gap-3">
          <Button variant="default" onClick={() => (streaming ? stopLoop() : startLoop())}>
            {rafRef.current ? "Pause" : "Resume"}
          </Button>
          <Button variant={jawRecording ? "destructive" : "secondary"} onClick={() => setJawRecording((v) => !v)}>
            {jawRecording ? "Stop Jaw Recording" : "Record Jaw Opening"}
          </Button>
          <Badge variant="secondary">Signal quality: {(signalQuality * 100).toFixed(0)}%</Badge>
          <Badge variant="outline">Clips: {jawDatasetRef.current.length}</Badge>
        </div>

        <Separator className="my-4" />

        <div className="grid md:grid-cols-3 gap-4">
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Heart rate</h3>
              <Badge variant={confVariant as any}>{confidenceBadge} confidence</Badge>
            </div>
            <div className="mt-2 text-4xl font-bold">
              {bpm ? `${bpm} BPM` : "—"}
            </div>
            <p className="text-xs text-muted-foreground mt-2">Hold steady with good lighting. A stable forehead ROI is auto-tracked; wait ~15–30 seconds.</p>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Facial fullness</h3>
              <Badge>{fatCategory}</Badge>
            </div>
            <div className="mt-2">
              <div className="h-3 w-full rounded-full bg-secondary overflow-hidden">
                <div
                  className="h-full bg-primary transition-all"
                  style={{ width: `${Math.round(fullness * 100)}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground mt-2">Heuristic index based on face geometry. Lighting, camera angle, and expression affect results.</p>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Eye & Respiration</h3>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
              <div>
                <div className="text-muted-foreground">Blink rate</div>
                <div className="font-semibold">{blinkRate != null ? `${blinkRate.toFixed(0)} /min` : "—"}</div>
              </div>
              <div>
                <div className="text-muted-foreground">PERCLOS</div>
                <div className="font-semibold">{(perclos * 100).toFixed(0)}%</div>
              </div>
              <div>
                <div className="text-muted-foreground">Yawn</div>
                <div className="font-semibold">{(yawnProb * 100).toFixed(0)}%</div>
              </div>
              <div>
                <div className="text-muted-foreground">Resp. rate</div>
                <div className="font-semibold">{respRate != null ? `${respRate} brpm` : "—"}</div>
              </div>
              <div className="col-span-2">
                <div className="text-muted-foreground">HRV (RMSSD)</div>
                <div className="font-semibold">{rmssd != null ? `${rmssd.toFixed(0)} ms` : "—"}</div>
              </div>
            </div>
          </Card>
        </div>
        <div className="mt-4">
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Facial motion over time</h3>
              <Badge variant="secondary">Now: {(motionNow * 100).toFixed(1)}</Badge>
            </div>
            <div className="mt-2 text-muted-foreground text-xs">Relative landmark movement (rigid head motion removed).</div>
            <div className="mt-2">
              <Sparkline data={motionSeries.map((d) => d.v)} />
            </div>
          </Card>
        </div>
      </Card>
    </div>
  );
}

function Sparkline({ data, width = 600, height = 80 }: { data: number[]; width?: number; height?: number }) {
  const n = data.length || 1;
  const max = Math.max(1e-6, ...data);
  const step = width / Math.max(1, n - 1);
  let d = "";
  data.forEach((v, i) => {
    const x = i * step;
    const y = height - (v / max) * height;
    d += `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)} `;
  });
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <path d={d} fill="none" stroke="hsl(var(--primary))" strokeWidth="2" />
    </svg>
  );
}

function getBounds(points: { x: number; y: number }[], width: number, height: number) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const x = minX * width;
  const y = minY * height;
  const w = (maxX - minX) * width;
  const h = (maxY - minY) * height;
  return { x, y, w, h };
}
