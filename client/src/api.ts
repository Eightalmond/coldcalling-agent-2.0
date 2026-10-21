// Thin typed wrapper around the CALL-E server API (src/calle/server.py).

// Empty = same origin (local: the Python server serves this UI, or Vite's dev
// proxy forwards /api). In production, point it at the deployed server.
export const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? "";

export interface Config {
  web_search_available: boolean;
  call_max_secs: number;
  calls_per_hour: number;
  calls_per_day: number;
  brief_max_chars: number;
  max_upload_bytes: number;
  supported_files: string[];
  ice_servers: RTCIceServer[];
}

export interface BriefResult {
  brief: string;
  warnings: string[];
  used_web: boolean;
}

export interface CallInfo {
  call_id: string;
  max_call_secs: number;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly problems: string[] = [],
    readonly retryAfter?: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError("Can't reach the CALL-E server. Is it running?", 0);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : response.statusText;
    throw new ApiError(detail, response.status, body.problems ?? [], body.retry_after);
  }
  return body as T;
}

export const getConfig = () => request<Config>("/api/config");

export const getDemoBrief = () => request<{ brief: string }>("/api/demo-brief");

export const generateBrief = (form: FormData) =>
  request<BriefResult>("/api/briefs", { method: "POST", body: form });

export const createCall = (brief: string) =>
  request<CallInfo>("/api/calls", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ brief }),
  });
