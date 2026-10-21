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

export interface CallHandlers {
  onStatus: (status: string) => void;
  onReady: () => void;
  onUserText: (text: string, final: boolean) => void;
  onBotText: (text: string) => void;
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
        // The bot's voice arrives as a remote audio track; play it. The
        // SmallWebRTC transport reports remote tracks WITHOUT a participant
        // (only local tracks carry one), so "not local" is the right test.
        onTrackStarted: (track, participant) => {
          if (track.kind === "audio" && !participant?.local) {
            audioEl.srcObject = new MediaStream([track]);
            audioEl.play().catch((error) => console.warn("Bot audio playback blocked", error));
          }
        },
        onUserInput: (data) => handlers.onUserText(data.text, data.final),
        // Pipecat sends each sentence twice: "new" when it's queued for
        // speech and "completed" once its audio has played. Show only the
        // latter, so the transcript matches what the caller actually heard.
        onBotOutput: (data) => {
          if (data.spoken_status === "completed") handlers.onBotText(data.text);
        },
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
}
