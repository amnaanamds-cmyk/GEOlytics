"use client";

/** Browser-side calls. Always to the proxy, never to the API directly. */

import type { ApiError } from "@/lib/types";

export class RequestError extends Error {
  constructor(
    readonly status: number,
    readonly payload: ApiError | Record<string, unknown> | null,
    message: string,
  ) {
    super(message);
  }
}

function describe(status: number, payload: unknown): string {
  if (payload && typeof payload === "object") {
    const body = payload as Record<string, unknown>;
    // The API returns a quota refusal as a structured `detail` object.
    const detail = body.detail;
    if (detail && typeof detail === "object") {
      const inner = detail as Record<string, unknown>;
      if (typeof inner.message === "string") return inner.message;
    }
    if (typeof detail === "string") return detail;
    if (typeof body.message === "string") return body.message;
  }
  return `Request failed (${status})`;
}

export async function api<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`/api/proxy/${path.replace(/^\//, "")}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
  });

  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;

  if (!response.ok) {
    throw new RequestError(response.status, payload, describe(response.status, payload));
  }
  return payload as T;
}

export const get = <T>(path: string) => api<T>(path);
export const post = <T>(path: string, body?: unknown) =>
  api<T>(path, { method: "POST", body: JSON.stringify(body ?? {}) });
export const patch = <T>(path: string, body?: unknown) =>
  api<T>(path, { method: "PATCH", body: JSON.stringify(body ?? {}) });
export const del = <T>(path: string) => api<T>(path, { method: "DELETE" });
