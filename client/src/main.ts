// CALL-E web client: a three-step wizard (product -> brief -> call).

import "./style.css";

import { ApiError, createCall, generateBrief, getConfig, getDemoBrief, type Config } from "./api";
import { Call, type BotState } from "./call";
import { LatencyPanel } from "./latency";
import { LevelBars, Orb, TrackMeter } from "./visualizer";

const $ = <T extends Element = HTMLElement>(selector: string) =>
  document.querySelector(selector) as T;

type Screen = "product" | "brief" | "call";

const productForm = $<HTMLFormElement>("#product-form");
const fileInput = $<HTMLInputElement>("#file-input");
const dropzone = $<HTMLLabelElement>("#dropzone");
const generateBtn = $<HTMLButtonElement>("#generate-btn");
const productStatus = $("#product-status");
const briefEl = $<HTMLTextAreaElement>("#brief");
const briefCount = $("#brief-count");
const briefProblems = $<HTMLUListElement>("#brief-problems");
const callBtn = $<HTMLButtonElement>("#call-btn");
const muteBtn = $<HTMLButtonElement>("#mute-btn");
const callStatus = $("#call-status");
const callTimer = $("#call-timer");
const callNote = $("#call-note");
const botStateEl = $("#bot-state");
const micSelect = $<HTMLSelectElement>("#mic-select");
const speakerSelect = $<HTMLSelectElement>("#speaker-select");
const botAudio = $<HTMLAudioElement>("#bot-audio");

const accentRgb = getComputedStyle(document.documentElement).getPropertyValue("--accent-rgb").trim();
const orb = new Orb($<HTMLCanvasElement>("#orb"), accentRgb);
const micBars = new LevelBars($("#mic-level"));
const latency = new LatencyPanel(
  $<SVGSVGElement>("#lat-chart"),
  $("#lat-last"),
  $("#lat-median"),
  $("#lat-count"),
  $("#lat-breakdown"),
);

let config: Config | null = null;
let call: Call | null = null;
let audioContext: AudioContext | null = null;
let meters: TrackMeter[] = [];
let countdown: number | undefined;
let callStartedAt = 0;

// ------------------------------------------------------------- helpers

function setNote(el: HTMLElement, text: string, kind: "info" | "error" | "ok" = "info") {
  el.textContent = text;
  el.dataset.kind = kind;
}

function describe(error: unknown): string {
  if (error instanceof ApiError && error.status === 429 && error.retryAfter) {
    return `${error.message} (about ${Math.ceil(error.retryAfter / 60)} min)`;
  }
  return error instanceof Error ? error.message : String(error);
}

function showProblems(problems: string[]) {
  briefProblems.replaceChildren(
    ...problems.map((p) => Object.assign(document.createElement("li"), { textContent: p })),
  );
}

function updateBriefCount() {
  const max = config?.brief_max_chars ?? 3000;
  const len = briefEl.value.length;
  briefCount.textContent = `${len.toLocaleString()} / ${max.toLocaleString()}`;
  briefCount.dataset.kind = len > max ? "error" : "info";
}

function setBrief(text: string) {
  briefEl.value = text;
  showProblems([]);
  updateBriefCount();
}

const formatClock = (secs: number) =>
  `${Math.floor(secs / 60)}:${String(Math.max(0, secs) % 60).padStart(2, "0")}`;

// ------------------------------------------------------------- wizard

let current: Screen = "product";

function show(screen: Screen) {
  if (call) return; // no navigating away mid-call
  if (screen !== "product" && !briefEl.value.trim()) screen = "product";
  current = screen;
  document.querySelectorAll<HTMLElement>("[data-screen]").forEach((el) => {
    el.hidden = el.dataset.screen !== screen;
  });
  const order: Screen[] = ["product", "brief", "call"];
  document.querySelectorAll<HTMLButtonElement>("#stepper button").forEach((button) => {
    const step = button.dataset.goto as Screen;
    button.setAttribute("aria-current", step === screen ? "step" : "false");
    button.dataset.done = String(order.indexOf(step) < order.indexOf(screen));
    // Steps after the first need a brief.
    button.disabled = step !== "product" && !briefEl.value.trim();
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest<HTMLElement>("[data-goto]");
  if (target) show(target.dataset.goto as Screen);
});

// ------------------------------------------------------------- step 1: product

function showFileName() {
  const file = fileInput.files?.[0];
  $("#file-name").textContent = file ? file.name : "Drop a product sheet here, or click to choose";
  dropzone.dataset.filled = String(Boolean(file));
}

fileInput.addEventListener("change", showFileName);
for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.dataset.dragging = "true";
  });
}
for (const type of ["dragleave", "drop"]) {
  dropzone.addEventListener(type, () => (dropzone.dataset.dragging = "false"));
}
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  if (event.dataTransfer?.files.length) {
    fileInput.files = event.dataTransfer.files;
    showFileName();
  }
});

productForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(productForm);
  if (!fileInput.files?.length) form.delete("file");
  generateBtn.disabled = true;
  generateBtn.dataset.busy = "true";
  setNote(productStatus, "Reading your material and writing the brief… this takes 10–30 s.");
  try {
    const result = await generateBrief(form);
    setBrief(result.brief);
    $("#bot-company").textContent = String(form.get("company_name") || "AI sales rep");
    const notes = [result.used_web ? "Includes web results." : "", ...result.warnings].filter(Boolean);
    setNote(productStatus, notes.join(" "), "ok");
    show("brief");
  } catch (error) {
    setNote(productStatus, describe(error), "error");
  } finally {
    generateBtn.disabled = false;
    delete generateBtn.dataset.busy;
  }
});

$("#demo-btn").addEventListener("click", async () => {
  try {
    setBrief((await getDemoBrief()).brief);
    $("#bot-company").textContent = "Brightfern Labs";
    setNote(productStatus, "");
    show("brief");
  } catch (error) {
    setNote(productStatus, describe(error), "error");
  }
});

briefEl.addEventListener("input", updateBriefCount);

// ------------------------------------------------------------- step 3: call

const BOT_STATE_TEXT: Record<BotState, string> = {
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking",
};

function setCallStatus(state: "idle" | "connecting" | "live" | "ended" | "error", text: string) {
  callStatus.dataset.state = state;
  callStatus.textContent = text;
}

function startCountdown(seconds: number) {
  const endsAt = Date.now() + seconds * 1000;
  const tick = () => {
    const left = Math.max(0, Math.round((endsAt - Date.now()) / 1000));
    callTimer.textContent = `${formatClock(left)} left`;
    callTimer.dataset.warn = String(left <= 30);
    if (left === 0) window.clearInterval(countdown);
  };
  tick();
  countdown = window.setInterval(tick, 1000);
}

async function fillDevices(active: Call) {
  const mics = await active.listMics();
  micSelect.replaceChildren(
    ...mics.map((m, i) => new Option(m.label || `Microphone ${i + 1}`, m.deviceId)),
  );
  micSelect.value = active.selectedMicId || micSelect.value;
  micSelect.disabled = mics.length < 2;

  // Bot audio plays through our own <audio> element, so pick its output
  // device with setSinkId (not supported everywhere, e.g. Safari).
  if (!("setSinkId" in HTMLMediaElement.prototype)) {
    $("#speaker-field").hidden = true;
    return;
  }
  const speakers = (await navigator.mediaDevices.enumerateDevices()).filter(
    (d) => d.kind === "audiooutput",
  );
  speakerSelect.replaceChildren(
    ...speakers.map((s, i) => new Option(s.label || `Speaker ${i + 1}`, s.deviceId)),
  );
  speakerSelect.disabled = speakers.length < 2;
}

micSelect.addEventListener("change", () => call?.useMic(micSelect.value));
speakerSelect.addEventListener("change", () => {
  void (botAudio as HTMLAudioElement & { setSinkId(id: string): Promise<void> })
    .setSinkId(speakerSelect.value)
    .catch((error) => console.warn("Could not switch speaker", error));
});

muteBtn.addEventListener("click", () => {
  if (!call) return;
  const muted = call.micEnabled;
  call.setMicEnabled(!muted);
  muteBtn.setAttribute("aria-pressed", String(muted));
  $("#mute-label").textContent = muted ? "Unmute" : "Mute";
});

function attachMeter(track: MediaStreamTrack): TrackMeter | null {
  if (!audioContext) return null;
  const meter = new TrackMeter(audioContext, track);
  meters.push(meter);
  return meter;
}

function teardownCall() {
  window.clearInterval(countdown);
  orb.setMeter(null);
  micBars.stop();
  meters.forEach((m) => m.dispose());
  meters = [];
  callBtn.textContent = "Start call";
  callBtn.dataset.live = "false";
  callBtn.disabled = false;
  muteBtn.disabled = true;
  muteBtn.setAttribute("aria-pressed", "false");
  $("#mute-label").textContent = "Mute";
  micSelect.disabled = speakerSelect.disabled = true;
  document.body.dataset.inCall = "false";
}

async function startCall() {
  callBtn.disabled = true;
  setCallStatus("connecting", "Checking brief");
  setNote(callNote, "");
  // Audio analysis needs an AudioContext created during a user gesture.
  audioContext ??= new AudioContext();
  void audioContext.resume();

  let info;
  try {
    info = await createCall(briefEl.value);
  } catch (error) {
    setCallStatus("idle", "Ready");
    callBtn.disabled = false;
    if (error instanceof ApiError && error.problems.length) {
      show("brief");
      showProblems(error.problems);
      return;
    }
    setNote(callNote, describe(error), "error");
    return;
  }

  latency.reset();
  setCallStatus("connecting", "Connecting");
  botStateEl.textContent = "Allow the microphone when your browser asks.";
  document.body.dataset.inCall = "true";
  micBars.start();

  call = new Call(info.call_id, config?.ice_servers ?? [], botAudio, {
    onStatus: (status) => {
      if (status === "error") setCallStatus("error", "Error");
    },
    onReady: () => {
      callStartedAt = Date.now();
      setCallStatus("live", "Live");
      botStateEl.textContent = "Alex is calling you…";
      startCountdown(info.max_call_secs);
      callBtn.textContent = "End call";
      callBtn.dataset.live = "true";
      callBtn.disabled = false;
      muteBtn.disabled = false;
      if (call) void fillDevices(call);
    },
    onBotState: (state) => {
      botStateEl.textContent = BOT_STATE_TEXT[state];
      botStateEl.dataset.state = state;
    },
    onBotTrack: (track) => orb.setMeter(attachMeter(track)),
    onMicTrack: (track) => micBars.setMeter(attachMeter(track)),
    onMetrics: (metrics) => latency.add(metrics),
    onEnded: () => {
      call = null;
      const duration = callStartedAt ? Math.round((Date.now() - callStartedAt) / 1000) : 0;
      callStartedAt = 0;
      teardownCall();
      setCallStatus("ended", "Call ended");
      callTimer.textContent = "";
      botStateEl.textContent = "Call ended.";
      delete botStateEl.dataset.state;
      const median = latency.median;
      setNote(
        callNote,
        [
          `Lasted ${formatClock(duration)}`,
          `${latency.count} turns`,
          median != null ? `median latency ${(median / 1000).toFixed(2)} s` : "",
        ]
          .filter(Boolean)
          .join(" · "),
      );
      callBtn.textContent = "Call again";
    },
  });

  try {
    await call.start();
  } catch (error) {
    call = null;
    teardownCall();
    setCallStatus("error", "Couldn't connect");
    botStateEl.textContent = "";
    setNote(callNote, describe(error), "error");
  }
}

callBtn.addEventListener("click", async () => {
  if (call) {
    callBtn.disabled = true;
    await call.hangUp();
  } else {
    await startCall();
  }
});

// ------------------------------------------------------------- init

async function init() {
  show("product");
  orb.start(); // idles gently until a call gives it a voice to follow
  try {
    config = await getConfig();
    fileInput.accept = config.supported_files.join(",");
    $("#file-hint").textContent =
      `${config.supported_files.join(", ")} · up to ${Math.round(config.max_upload_bytes / 1048576)} MB`;
    $("#web-option").hidden = !config.web_search_available;
    setNote(
      callNote,
      `Calls last up to ${Math.round(config.call_max_secs / 60)} minutes, ` +
        `${config.calls_per_hour} per hour.`,
    );
  } catch (error) {
    setNote(productStatus, describe(error), "error");
  }
  show(current);
}

void init();
