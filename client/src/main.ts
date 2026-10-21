// CALL-E web client: product material -> editable brief -> voice call.

import "./style.css";

import { ApiError, createCall, generateBrief, getConfig, getDemoBrief, type Config } from "./api";
import { Call, type TurnMetrics } from "./call";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const productForm = $<HTMLFormElement>("product-form");
const fileInput = $<HTMLInputElement>("file-input");
const generateBtn = $<HTMLButtonElement>("generate-btn");
const demoBtn = $<HTMLButtonElement>("demo-btn");
const productStatus = $("product-status");
const briefEl = $<HTMLTextAreaElement>("brief");
const briefCount = $("brief-count");
const briefProblems = $<HTMLUListElement>("brief-problems");
const callBtn = $<HTMLButtonElement>("call-btn");
const callStatus = $("call-status");
const callTimer = $("call-timer");
const callNote = $("call-note");
const transcript = $<HTMLOListElement>("transcript");
const latencyRows = $<HTMLTableSectionElement>("latency-rows");
const latencySummary = $("latency-summary");
const botAudio = $<HTMLAudioElement>("bot-audio");

let config: Config | null = null;
let call: Call | null = null;
let countdown: number | undefined;

// ------------------------------------------------------------- helpers

function setStatus(el: HTMLElement, text: string, kind: "info" | "error" | "ok" = "info") {
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
  briefCount.textContent = `${len.toLocaleString()} / ${max.toLocaleString()} characters`;
  briefCount.dataset.kind = len > max ? "error" : "info";
}

function setBrief(text: string) {
  briefEl.value = text;
  showProblems([]);
  updateBriefCount();
}

const ms = (value: number | null) => (value == null ? "–" : `${Math.round(value)} ms`);

// ------------------------------------------------------------- step 1: brief

productForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(productForm);
  if (!fileInput.files?.length) form.delete("file");
  generateBtn.disabled = true;
  setStatus(productStatus, "Reading your material and writing the brief… (10–30 s)");
  try {
    const result = await generateBrief(form);
    setBrief(result.brief);
    const notes = [result.used_web ? "Includes web results." : "", ...result.warnings].filter(Boolean);
    setStatus(productStatus, ["Brief ready. Review it below.", ...notes].join(" "), "ok");
  } catch (error) {
    setStatus(productStatus, describe(error), "error");
  } finally {
    generateBtn.disabled = false;
  }
});

demoBtn.addEventListener("click", async () => {
  try {
    setBrief((await getDemoBrief()).brief);
    setStatus(productStatus, "Loaded the demo brief (Quillo, a fictional AI meeting assistant).", "ok");
  } catch (error) {
    setStatus(productStatus, describe(error), "error");
  }
});

briefEl.addEventListener("input", updateBriefCount);

// ------------------------------------------------------------- step 3: call

function addLine(who: "user" | "bot", text: string, interim = false): HTMLLIElement {
  const li = document.createElement("li");
  li.className = `line ${who}${interim ? " interim" : ""}`;
  li.textContent = text;
  transcript.append(li);
  li.scrollIntoView({ block: "nearest" });
  return li;
}

let interimLine: HTMLLIElement | null = null;
let botLine: HTMLLIElement | null = null;

function onUserText(text: string, final: boolean) {
  if (!text.trim()) return;
  if (!interimLine) interimLine = addLine("user", text, true);
  interimLine.textContent = text;
  if (final) {
    interimLine.classList.remove("interim");
    interimLine = null;
    botLine = null; // the next bot sentence starts a new bubble
  }
}

function onBotText(text: string) {
  if (!botLine) botLine = addLine("bot", text);
  else botLine.textContent = `${botLine.textContent} ${text}`;
}

const totals: number[] = [];

function onMetrics(m: TurnMetrics) {
  const row = latencyRows.insertRow();
  const cells = [
    String(latencyRows.rows.length),
    ms(m.total_ms) + (m.interrupted ? " ✂" : ""),
    ms(m.first_token_ms),
    ms(m.token_to_audio_ms),
  ];
  for (const text of cells) row.insertCell().textContent = text;
  if (m.anchor !== "VAD stop") row.title = `Timed from ${m.anchor}`;
  if (m.total_ms != null && m.anchor === "VAD stop") totals.push(m.total_ms);
  if (totals.length) {
    const sorted = [...totals].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    latencySummary.textContent = `Median from you stopping to the rep's first audio: ${Math.round(median)} ms over ${totals.length} turns.`;
  }
}

function startCountdown(seconds: number) {
  const endsAt = Date.now() + seconds * 1000;
  const tick = () => {
    const left = Math.max(0, Math.round((endsAt - Date.now()) / 1000));
    callTimer.textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")} left`;
    if (left === 0) window.clearInterval(countdown);
  };
  tick();
  countdown = window.setInterval(tick, 1000);
}

function resetCallUi() {
  window.clearInterval(countdown);
  callTimer.textContent = "";
  callBtn.textContent = "Start call";
  callBtn.disabled = false;
  callStatus.textContent = "idle";
  interimLine = botLine = null;
}

async function startCall() {
  showProblems([]);
  callBtn.disabled = true;
  setStatus(callNote, "Checking the brief…");
  try {
    const info = await createCall(briefEl.value);
    transcript.replaceChildren();
    latencyRows.replaceChildren();
    latencySummary.textContent = "";
    totals.length = 0;
    setStatus(callNote, "Connecting… allow the microphone when asked.");

    call = new Call(info.call_id, config?.ice_servers ?? [], botAudio, {
      onStatus: (status) => (callStatus.textContent = status),
      onReady: () => {
        setStatus(callNote, "Connected. The rep will greet you first.", "ok");
        startCountdown(info.max_call_secs);
        callBtn.textContent = "End call";
        callBtn.disabled = false;
      },
      onUserText,
      onBotText,
      onMetrics,
      onEnded: () => {
        call = null;
        resetCallUi();
        setStatus(callNote, "Call ended.");
      },
    });
    await call.start();
  } catch (error) {
    call = null;
    resetCallUi();
    if (error instanceof ApiError && error.problems.length) showProblems(error.problems);
    setStatus(callNote, describe(error), "error");
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
  try {
    config = await getConfig();
    fileInput.accept = config.supported_files.join(",");
    $("file-hint").textContent =
      `${config.supported_files.join(", ")} up to ${Math.round(config.max_upload_bytes / 1048576)} MB`;
    $("web-option").hidden = !config.web_search_available;
    setStatus(
      callNote,
      `Calls last up to ${Math.round(config.call_max_secs / 60)} minutes; ` +
        `${config.calls_per_hour} calls per hour per visitor.`,
    );
    setBrief((await getDemoBrief()).brief);
  } catch (error) {
    setStatus(productStatus, describe(error), "error");
    callBtn.disabled = true;
  }
}

void init();
