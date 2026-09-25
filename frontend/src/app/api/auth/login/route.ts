import { NextResponse } from "next/server";

import { callApi, readJson } from "@/lib/upstream";
import { storeSession, type TokenPair } from "@/lib/session";

export async function POST(request: Request) {
  const payload = await request.json().catch(() => null);
  if (!payload) {
    return NextResponse.json(
      { error: "invalid_request", message: "expected a JSON body" },
      { status: 400 },
    );
  }

  const response = await callApi("/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  const body = await readJson(response);

  if (!response.ok) {
    return NextResponse.json(body, { status: response.status });
  }

  await storeSession(body as TokenPair);
  // The tokens stay in httpOnly cookies; only the display fields come back.
  return NextResponse.json({ organization: (body as TokenPair).organization });
}
