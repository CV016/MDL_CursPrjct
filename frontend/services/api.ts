/**
 * services/api.ts
 * JWT-authenticated fetch utility.
 *
 * All outbound requests to the gateway are funnelled through `apiFetch`,
 * which:
 *  1. Reads the JWT from the httpOnly cookie server-side, or from
 *     the request cookie header in client components.
 *  2. Attaches the token as a Bearer Authorization header.
 *  3. Throws a typed ApiError on non-2xx responses so callers can handle
 *     authentication failures (401) distinctly from server errors (5xx).
 *
 * The base URL is read from NEXT_PUBLIC_GATEWAY_URL at build time,
 * allowing local, staging, and production deployments to point at different
 * gateway addresses without code changes.
 */

const GATEWAY_URL =
  process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  public readonly status: number;
  public readonly detail: string;

  constructor(status: number, detail: string) {
    super(`API error ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

// ---------------------------------------------------------------------------
// Token retrieval
// ---------------------------------------------------------------------------

/**
 * Read the access token from document.cookie (client-side only).
 * The cookie is httpOnly so it will not appear here from client JS —
 * this function is kept for non-httpOnly fallback scenarios and SSR
 * contexts where cookies() from next/headers is used instead.
 *
 * Returns an empty string if no token is found.
 */
function getTokenFromCookie(): string {
  if (typeof document === "undefined") {
    // Server-side rendering: caller must supply the token via headers
    return "";
  }
  const match = document.cookie.match(/(?:^|;\s*)access_token=([^;]*)/);
  return match ? decodeURIComponent(match[1]) : "";
}

// ---------------------------------------------------------------------------
// Core fetch wrapper
// ---------------------------------------------------------------------------

/**
 * Authenticated fetch to the AI-DOC INTERACT gateway.
 *
 * @param path    - Gateway endpoint path, e.g. "/upload" or "/feedback".
 * @param init    - Standard RequestInit options (method, body, headers, etc.).
 * @param token   - Optional explicit token; falls back to cookie if omitted.
 * @returns       Parsed JSON body of the response.
 * @throws ApiError on non-2xx status codes.
 */
export async function apiFetch<T = unknown>(
  path: string,
  init: RequestInit = {},
  token?: string,
): Promise<T> {
  const resolvedToken = token ?? getTokenFromCookie();

  const headers = new Headers(init.headers);
  if (resolvedToken) {
    headers.set("Authorization", `Bearer ${resolvedToken}`);
  }

  // Do not override Content-Type when sending FormData — the browser must set
  // it with the correct multipart boundary automatically.
  if (!(init.body instanceof FormData)) {
    if (!headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
  }

  const url = `${GATEWAY_URL}${path}`;
  // credentials: "include" is required so the browser automatically attaches
  // the httpOnly access_token cookie on cross-origin requests to the gateway.
  // Manually reading document.cookie will never surface an httpOnly cookie.
  const response = await fetch(url, { ...init, headers, credentials: "include" });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errorBody = (await response.json()) as { detail?: string };
      detail = errorBody.detail ?? detail;
    } catch {
      // Response body is not JSON — use statusText as fallback
    }
    throw new ApiError(response.status, detail);
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/**
 * POST JSON payload to the gateway.
 */
export async function apiPost<T = unknown>(
  path: string,
  body: unknown,
  token?: string,
): Promise<T> {
  return apiFetch<T>(path, { method: "POST", body: JSON.stringify(body) }, token);
}
