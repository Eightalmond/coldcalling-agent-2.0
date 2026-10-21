// One voice call: wraps Pipecat's PipecatClient + SmallWebRTCTransport and
// turns its events into a few simple callbacks for the UI.

import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";

import { API_BASE } from "./api";

// Sent by the bot after every turn (see on_turn_metrics in src/calle/bot.py).
export interface TurnMetrics {
  anchor: string;
  interrupted: boolean;
  vad_wait_ms: number | null;
  turn_end_ms: number | null;
  first_token_ms: number | null;
  token_to_audio_ms: number | null;
  total_ms: number | null;
  filler_ms: number | null;
}

export type BotState = "listening" | "thinking" | "speaking";

export interface CallHandlers {
  onStatus: (status: string) => void;
  onReady: () => void;
  onBotState: (state: BotState) => void;
  onBotTrack: (track: MediaStreamTrack) => void;
  onMicTrack: (track: MediaStreamTrack) => void;
  onMetrics: (metrics: TurnMetrics) => void;
  onEnded: () => void;
}

export class Call {
  private readonly client: PipecatClient;
  private ended = false;

  constructor(
    private readonly callId: string,
    iceServers: RTCIceServer[],
    audioEl: HTMLAudioElement,
    handlers: CallHandlers,
  ) {
    const end = () => {
      if (this.ended) return;
      this.ended = true;
      handlers.onEnded();
    };

    this.client = new PipecatClient({
      transport: new SmallWebRTCTransport({ iceServers }),
      enableMic: true,
      enableCam: false,
      callbacks: {
        onTransportStateChanged: (state) => handlers.onStatus(state),
        onBotReady: () => handlers.onReady(),
        // The SmallWebRTC transport reports the bot's (remote) tracks WITHOUT a
        // participant; only local tracks carry one, so "not local" = the bot.
        onTrackStarted: (track, participant) => {
          if (track.kind !== "audio") return;
          if (participant?.local) {
            handlers.onMicTrack(track);
          } else {
            audioEl.srcObject = new MediaStream([track]);
            audioEl.play().catch((error) => console.warn("Bot audio playback blocked", error));
            handlers.onBotTrack(track);
          }
        },
        // Turn-taking state for the status line: the user's turn ending means
        // the bot is now working on a reply until its audio starts.
        onUserStoppedSpeaking: () => handlers.onBotState("thinking"),
        onUserStartedSpeaking: () => handlers.onBotState("listening"),
        onBotStartedSpeaking: () => handlers.onBotState("speaking"),
        onBotStoppedSpeaking: () => handlers.onBotState("listening"),
        onServerMessage: (data) => {
          if (data?.type === "turn_metrics") handlers.onMetrics(data as TurnMetrics);
        },
        onDisconnected: end,
        onBotDisconnected: end,
        onError: (message) => {
          console.error("Call error", message);
          handlers.onStatus("error");
        },
      },
    });
  }

  async start(): Promise<void> {
    await this.client.connect({
      webrtcRequestParams: {
        endpoint: `${API_BASE}/api/offer`,
        // The server only accepts offers carrying a fresh call id from /api/calls.
        requestData: { call_id: this.callId },
      },
    });
  }

  async hangUp(): Promise<void> {
    await this.client.disconnect();
  }

  get micEnabled(): boolean {
    return this.client.isMicEnabled;
  }

  setMicEnabled(enabled: boolean): void {
    this.client.enableMic(enabled);
  }

  listMics(): Promise<MediaDeviceInfo[]> {
    return this.client.getAllMics();
  }

  get selectedMicId(): string {
    return this.client.selectedMic.deviceId ?? "";
  }

  useMic(deviceId: string): void {
    this.client.updateMic(deviceId);
  }
}
