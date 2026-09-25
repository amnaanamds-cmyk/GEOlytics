/**
 * Session cookies.
 *
 * Tokens live in httpOnly cookies rather than localStorage. That is the whole
 * reason this app proxies the API instead of calling it from the browser: a
 * token readable by JavaScript is a token any XSS can exfiltrate, and an
 * access token is a bearer credential for a customer's entire organisation.
 *
 * `sameSite: "lax"` rather than "strict": strict would drop the cookie on
 * every inbound link into the dashboard, logging people out whenever they
 * arrive from an email. Lax still withholds it from cross-site POSTs, which
 * is the CSRF case that matters, and every mutating route here is a POST.
 */

import { cookies } from "next/headers";

import { IS_PRODUCTION } from "@/lib/config";

export const ACCESS_COOKIE = "geo_at";
export const REFRESH_COOKIE = "geo_rt";
export const ORG_COOKIE = "geo_org";

const BASE_OPTIONS = {
  httpOnly: true,
  secure: IS_PRODUCTION,
  sameSite: "lax" as const,
  path: "/",
};

export type TokenPair = {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  organization: { id: number; slug: string; name: string; plan: string; role?: string };
};

export async function storeSession(tokens: TokenPair): Promise<void> {
  const store = await cookies();

  // The access cookie is given a slightly shorter life than the token itself,
  // so the browser discards it before the API would start rejecting it.
  store.set(ACCESS_COOKIE, tokens.access_token, {
    ...BASE_OPTIONS,
    maxAge: Math.max(60, tokens.expires_in - 30),
  });
  store.set(REFRESH_COOKIE, tokens.refresh_token, {
    ...BASE_OPTIONS,
    maxAge: 60 * 60 * 24 * 30,
  });
  // Readable by the UI: it is only a display label, never a credential.
  store.set(ORG_COOKIE, tokens.organization.slug, {
    ...BASE_OPTIONS,
    httpOnly: false,
    maxAge: 60 * 60 * 24 * 30,
  });
}

export async function clearSession(): Promise<void> {
  const store = await cookies();
  for (const name of [ACCESS_COOKIE, REFRESH_COOKIE, ORG_COOKIE]) {
    store.set(name, "", { ...BASE_OPTIONS, httpOnly: name !== ORG_COOKIE, maxAge: 0 });
  }
}

export async function readAccessToken(): Promise<string | null> {
  return (await cookies()).get(ACCESS_COOKIE)?.value ?? null;
}

export async function readRefreshToken(): Promise<string | null> {
  return (await cookies()).get(REFRESH_COOKIE)?.value ?? null;
}

export async function hasSession(): Promise<boolean> {
  const store = await cookies();
  return Boolean(store.get(ACCESS_COOKIE)?.value ?? store.get(REFRESH_COOKIE)?.value);
}
