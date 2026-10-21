// Latency panel: per-turn bar chart (inline SVG) plus headline numbers.

import type { TurnMetrics } from "./call";

const SVG = "http://www.w3.org/2000/svg";
const CHART_H = 120;
const MAX_BARS = 16;

export class LatencyPanel {
  private turns: TurnMetrics[] = [];

  constructor(
    private readonly chart: SVGSVGElement,
    private readonly lastEl: HTMLElement,
    private readonly medianEl: HTMLElement,
    private readonly countEl: HTMLElement,
    private readonly breakdownEl: HTMLElement,
  ) {
    this.render();
    new ResizeObserver(() => this.drawChart()).observe(chart);
  }

  reset(): void {
    this.turns = [];
    this.render();
  }

  add(metrics: TurnMetrics): void {
    this.turns.push(metrics);
    this.render();
  }

  /** Turns timed from "VAD stop" with audio: comparable end-to-end numbers. */
  private get comparable(): number[] {
    return this.turns
      .filter((t) => t.anchor === "VAD stop" && t.total_ms != null)
      .map((t) => t.total_ms as number);
  }

  get median(): number | null {
    const values = [...this.comparable].sort((a, b) => a - b);
    return values.length ? values[Math.floor(values.length / 2)] : null;
  }

  get count(): number {
    return this.turns.length;
  }

  private render(): void {
    const last = [...this.turns].reverse().find((t) => t.total_ms != null);
    this.lastEl.textContent = seconds(last?.total_ms ?? null);
    this.medianEl.textContent = seconds(this.median);
    this.countEl.textContent = String(this.turns.length);
    this.breakdownEl.textContent = last
      ? `turn end ${ms(last.turn_end_ms)} · LLM first token ${ms(last.first_token_ms)} · ` +
        `token → audio ${ms(last.token_to_audio_ms)}`
      : "Numbers appear after your first exchange.";
    this.drawChart();
  }

  private drawChart(): void {
    const chart = this.chart;
    chart.replaceChildren();
    const shown = this.turns.slice(-MAX_BARS);
    // Draw in real pixels (viewBox = rendered size) so text isn't stretched.
    const width = chart.clientWidth || 300;
    chart.setAttribute("viewBox", `0 0 ${width} ${CHART_H}`);
    const slot = width / Math.max(shown.length, 8);
    const barWidth = Math.min(22, slot * 0.7);

    const max = Math.max(2000, ...shown.map((t) => t.total_ms ?? 0));
    const y = (value: number) => CHART_H - (value / max) * (CHART_H - 14);

    // 1-second gridline for scale.
    const grid = document.createElementNS(SVG, "line");
    grid.setAttribute("x1", "0");
    grid.setAttribute("x2", String(width));
    grid.setAttribute("y1", String(y(1000)));
    grid.setAttribute("y2", String(y(1000)));
    grid.setAttribute("class", "grid");
    const gridLabel = document.createElementNS(SVG, "text");
    gridLabel.setAttribute("x", "0");
    gridLabel.setAttribute("y", String(y(1000) - 3));
    gridLabel.setAttribute("class", "grid-label");
    gridLabel.textContent = "1 s";
    chart.append(grid, gridLabel);

    shown.forEach((turn, i) => {
      const value = turn.total_ms ?? 0;
      const bar = document.createElementNS(SVG, "rect");
      const height = Math.max(3, CHART_H - y(value));
      bar.setAttribute("x", String(i * slot + (slot - barWidth) / 2));
      bar.setAttribute("y", String(CHART_H - height));
      bar.setAttribute("width", String(barWidth));
      bar.setAttribute("height", String(height));
      bar.setAttribute("rx", "4");
      bar.setAttribute(
        "class",
        turn.total_ms == null ? "bar empty" : turn.anchor === "VAD stop" ? "bar" : "bar alt",
      );
      const title = document.createElementNS(SVG, "title");
      title.textContent =
        turn.total_ms == null
          ? `Turn ${i + 1}: interrupted before audio`
          : `Turn ${i + 1}: ${ms(turn.total_ms)} (from ${turn.anchor})`;
      bar.append(title);
      chart.append(bar);
    });
  }
}

function ms(value: number | null): string {
  return value == null ? "–" : `${Math.round(value)} ms`;
}

function seconds(value: number | null): string {
  return value == null ? "–" : `${(value / 1000).toFixed(2)} s`;
}
