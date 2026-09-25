/**
 * Server-only configuration.
 *
 * The backend URL is deliberately not a NEXT_PUBLIC_ variable: the browser
 * never talks to the API directly. Every call goes through this app's route
 * handlers, which attach the credential from an httpOnly cookie.
 */

export const API_BASE_URL = (
  process.env.GEOLYTICS_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export const IS_PRODUCTION = process.env.NODE_ENV === "production";

/** Upper bound on a proxied request, so a hung backend cannot pin a Node worker. */
export const UPSTREAM_TIMEOUT_MS = 30_000;
