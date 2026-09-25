/**
 * Calls to the GEOlytics API, made server-side.
 *
 * Everything the browser asks for goes through here, so this is the one place
 * a credential is attached and the one place an expired session is renewed.
 */

import { API_BASE_URL, UPSTREAM_TIMEOUT_MS } from "@/lib/config";
import {
  readAccessToken,
  readRefreshToken,
  storeSession,
  type TokenPair,
} from "@/lib/session";

export type UpstreamResult = {
  status: number;
  body: unknown;
  headers: Headers;
};

/** Headers worth passing back to the browser; the rest are noise or unsafe. */
const FORWARDED_RESPONSE_HEADERS = [
  "x-request-id",
  "x-ratelimit-limit",
  "x-ratelimit-remaining",
  "retry-after",
];

export async function callApi(
  path: string,
  init: RequestInit = {},
  token?: string | null,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  try {
    return await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init.headers ?? {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      signal: controller.signal,
      // Never cache an authenticated response: one customer's data must not
      // be served to the next request that happens to match the URL.
      cache: "no-store",
    });
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Call the API with the session credential, renewing it once on a 401.
 *
 * Access tokens are short-lived by design, so a silent refresh is the
 * difference between a 15-minute session and a usable product. A single retry
 * is deliberate: if the refreshed token is also rejected, the session is
 * genuinely over and looping would only hammer the auth endpoint.
 */
export async function callApiWithSession(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const access = await readAccessToken();
  let response = await callApi(path, init, access);

  if (response.status !== 401) return response;

  const refreshed = await refreshSession();
  if (!refreshed) return response;

  response = await callApi(path, init, refreshed);
  return response;
}

async function refreshSession(): Promise<string | null> {
  const refresh = await readRefreshToken();
  if (!refresh) return null;

  const response = await callApi("/auth/refresh", {
    method: "POST",
    body: JSON.stringify({ refresh_token: refresh }),
  });
  if (!response.ok) return null;

  const tokens = (await response.json()) as TokenPair;
  await storeSession(tokens);
  return tokens.access_token;
}

export function passthroughHeaders(source: Headers): Headers {
  const headers = new Headers();
  for (const name of FORWARDED_RESPONSE_HEADERS) {
    const value = source.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

/** Read a JSON body, tolerating an empty or non-JSON response. */
export async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { error: "upstream_error", message: text.slice(0, 500) };
  }
}
