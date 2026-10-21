// Audio visuals for the call screen, driven by the Web Audio API.
//
// We analyse the tracks ourselves (AnalyserNode) instead of relying on the
// transport's level events: it gives real frequency data for the orb and
// works the same whichever Pipecat transport is used. Analysis only reads the
// audio; playback still happens through the page's <audio> element.

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export class TrackMeter {
  private readonly analyser: AnalyserNode;
  private readonly source: MediaStreamAudioSourceNode;
  private readonly bins: Uint8Array<ArrayBuffer>;

  constructor(context: AudioContext, track: MediaStreamTrack) {
    this.source = context.createMediaStreamSource(new MediaStream([track]));
    this.analyser = context.createAnalyser();
    this.analyser.fftSize = 256;
    this.analyser.smoothingTimeConstant = 0.75;
    this.source.connect(this.analyser);
    this.bins = new Uint8Array(new ArrayBuffer(this.analyser.frequencyBinCount));
  }

  /** Frequency magnitudes, 0..1, for the speech-relevant low/mid range. */
  spectrum(count: number): number[] {
    this.analyser.getByteFrequencyData(this.bins);
    const usable = Math.floor(this.bins.length * 0.6); // skip the mostly-empty top end
    const step = usable / count;
    return Array.from({ length: count }, (_, i) => this.bins[Math.floor(i * step)] / 255);
  }

  /** Overall loudness, 0..1. */
  level(): number {
    const values = this.spectrum(24);
    return Math.min(1, (values.reduce((a, b) => a + b, 0) / values.length) * 1.8);
  }

  dispose(): void {
    this.source.disconnect();
  }
}

/** The bot's orb: a glowing core ringed by bars that follow its voice. */
export class Orb {
  private readonly ctx: CanvasRenderingContext2D;
  private meter: TrackMeter | null = null;
  private frame = 0;
  private phase = 0;
  private smoothed: number[] = [];

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly accent: string,
  ) {
    this.ctx = canvas.getContext("2d")!;
  }

  setMeter(meter: TrackMeter | null): void {
    this.meter = meter;
  }

  start(): void {
    cancelAnimationFrame(this.frame);
    const draw = () => {
      this.draw();
      this.frame = requestAnimationFrame(draw);
    };
    draw();
  }

  stop(): void {
    cancelAnimationFrame(this.frame);
    this.meter = null;
    this.draw();
  }

  private draw(): void {
    const { canvas, ctx } = this;
    const dpr = window.devicePixelRatio || 1;
    const size = canvas.clientWidth;
    if (canvas.width !== size * dpr) {
      canvas.width = canvas.height = size * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const bars = 64;
    const raw = this.meter ? this.meter.spectrum(bars / 2) : new Array(bars / 2).fill(0);
    // Mirror so the ring is symmetric, then ease each bar toward its target.
    const target = [...raw, ...[...raw].reverse()];
    this.smoothed = target.map((v, i) => (this.smoothed[i] ?? 0) * 0.6 + v * 0.4);
    const loudness = this.smoothed.reduce((a, b) => a + b, 0) / bars;

    this.phase += prefersReducedMotion ? 0 : 0.02;
    const breathe = prefersReducedMotion ? 0 : Math.sin(this.phase) * 0.02;
    const center = size / 2;
    const core = size * (0.24 + breathe + loudness * 0.08);

    // Glow + core.
    const glow = ctx.createRadialGradient(center, center, core * 0.2, center, center, core * 1.9);
    glow.addColorStop(0, this.withAlpha(0.55 + loudness * 0.4));
    glow.addColorStop(1, this.withAlpha(0));
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(center, center, core * 1.9, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = this.withAlpha(0.95);
    ctx.beginPath();
    ctx.arc(center, center, core * 0.62, 0, Math.PI * 2);
    ctx.fill();

    // Ring of bars.
    const inner = core * 1.05;
    ctx.strokeStyle = this.withAlpha(0.9);
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(2, size / 160);
    this.smoothed.forEach((value, i) => {
      const angle = (i / bars) * Math.PI * 2 - Math.PI / 2;
      const length = size * (0.015 + value * 0.16);
      ctx.beginPath();
      ctx.moveTo(center + Math.cos(angle) * inner, center + Math.sin(angle) * inner);
      ctx.lineTo(
        center + Math.cos(angle) * (inner + length),
        center + Math.sin(angle) * (inner + length),
      );
      ctx.stroke();
    });
  }

  private withAlpha(alpha: number): string {
    // accent is "r g b" (space-separated), read from the --accent-rgb CSS token.
    return `rgb(${this.accent} / ${alpha.toFixed(3)})`;
  }
}

/** Your mic level as a row of small bars. */
export class LevelBars {
  private frame = 0;
  private meter: TrackMeter | null = null;
  private readonly bars: HTMLElement[];

  constructor(container: HTMLElement, count = 12) {
    container.replaceChildren();
    this.bars = Array.from({ length: count }, () => {
      const bar = document.createElement("span");
      container.append(bar);
      return bar;
    });
  }

  setMeter(meter: TrackMeter | null): void {
    this.meter = meter;
  }

  start(): void {
    cancelAnimationFrame(this.frame);
    const tick = () => {
      const values = this.meter ? this.meter.spectrum(this.bars.length) : [];
      this.bars.forEach((bar, i) => {
        const v = Math.min(1, (values[i] ?? 0) * 1.6);
        bar.style.transform = `scaleY(${0.12 + v * 0.88})`;
      });
      this.frame = requestAnimationFrame(tick);
    };
    tick();
  }

  stop(): void {
    cancelAnimationFrame(this.frame);
    this.meter = null;
    this.bars.forEach((bar) => (bar.style.transform = "scaleY(0.12)"));
  }
}
