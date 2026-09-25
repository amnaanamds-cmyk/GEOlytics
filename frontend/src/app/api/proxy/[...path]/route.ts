/**
 * Credential-attaching proxy to the GEOlytics API.
 *
 * The browser calls `/api/proxy/audits`; this forwards it with the session
 * token from an httpOnly cookie. Nothing client-side ever holds a credential.
 *
 * The path is rebuilt from the matched segments rather than taken from the
 * raw URL, so `..` cannot walk out of the API's namespace, and only the
 * methods listed below are exposed.
 */

import { NextResponse } from "next/server";

import { callApiWithSession, passthroughHeaders, readJson } from "@/lib/upstream";

type Context = { params: Promise<{ path: string[] }> };

// Paths the dashboard must never reach through the proxy, whatever a client
// asks for. The webhook is signature-authenticated and has no business being
// callable with a user's session.
const BLOCKED = [/^billing\/webhook$/];

async function forward(request: Request, context: Context, method: string) {
  const { path } = await context.params;
  const segments = (path ?? []).filter((s) => s && s !== "." && s !== "..");
  const joined = segments.join("/");

  if (!joined || BLOCKED.some((pattern) => pattern.test(joined))) {
    return NextResponse.json(
      { error: "not_found", message: "no such endpoint" },
      { status: 404 },
    );
  }

  const search = new URL(request.url).search;
  const hasBody = method !== "GET" && method !== "DELETE";

  const response = await callApiWithSession(`/${joined}${search}`, {
    method,
    ...(hasBody ? { body: await request.text() } : {}),
  });

  const body = await readJson(response);
  return NextResponse.json(body, {
    status: response.status,
    headers: passthroughHeaders(response.headers),
  });
}

export async function GET(request: Request, context: Context) {
  return forward(request, context, "GET");
}

export async function POST(request: Request, context: Context) {
  return forward(request, context, "POST");
}

export async function PATCH(request: Request, context: Context) {
  return forward(request, context, "PATCH");
}

export async function DELETE(request: Request, context: Context) {
  return forward(request, context, "DELETE");
}
