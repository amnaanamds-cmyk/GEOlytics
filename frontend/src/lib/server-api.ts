/**
 * Server-side data loading for page components.
 *
 * Initial data is fetched here, on the server, with the session cookie already
 * in hand. That removes the client waterfall (render, then fetch, then render
 * again) and the loading flash that comes with it; client components are left
 * to do only what they must — mutations and polling.
 */

import { callApiWithSession, readJson } from "@/lib/upstream";

export type Loaded<T> = { data: T; error: null } | { data: null; error: string };

export async function load<T>(path: string): Promise<Loaded<T>> {
  try {
    const response = await callApiWithSession(`/${path.replace(/^\//, "")}`);
    const body = await readJson(response);

    if (!response.ok) {
      return { data: null, error: describe(response.status, body) };
    }
    return { data: body as T, error: null };
  } catch {
    return { data: null, error: "Could not reach the GEOlytics API." };
  }
}

/** Load something the page can render without, e.g. a panel a viewer may not see. */
export async function loadOptional<T>(path: string): Promise<T | null> {
  const result = await load<T>(path);
  return result.data;
}

function describe(status: number, payload: unknown): string {
  if (payload && typeof payload === "object") {
    const body = payload as Record<string, unknown>;
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
