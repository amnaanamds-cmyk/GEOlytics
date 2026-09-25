"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

import { Alert, Button, Field } from "@/components/ui";

type Mode = "login" | "signup";

export function AuthForm({ mode }: { mode: Mode }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setBusy(true);

    const form = new FormData(event.currentTarget);
    const payload =
      mode === "signup"
        ? {
            email: form.get("email"),
            password: form.get("password"),
            full_name: form.get("full_name") || null,
            organization_name: form.get("organization_name"),
          }
        : { email: form.get("email"), password: form.get("password") };

    try {
      const response = await fetch(`/api/auth/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await response.json().catch(() => null);

      if (!response.ok) {
        const detail =
          (body?.detail && typeof body.detail === "string" && body.detail) ||
          body?.message ||
          body?.details?.[0]?.problem ||
          "Something went wrong. Please try again.";
        setError(detail);
        return;
      }
      // Replace, so the back button does not return to a form that is now stale.
      router.replace("/dashboard");
      router.refresh();
    } catch {
      setError("Could not reach the server. Check your connection and try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      {error && <Alert tone="error">{error}</Alert>}

      {mode === "signup" && (
        <>
          <Field
            label="Your name"
            name="full_name"
            autoComplete="name"
            placeholder="Amna Ahmad"
          />
          <Field
            label="Organisation"
            name="organization_name"
            required
            minLength={2}
            placeholder="Acme Plumbing"
            hint="You can rename this later."
          />
        </>
      )}

      <Field
        label="Email"
        name="email"
        type="email"
        required
        autoComplete="email"
        placeholder="you@company.com"
      />
      <Field
        label="Password"
        name="password"
        type="password"
        required
        minLength={mode === "signup" ? 12 : undefined}
        autoComplete={mode === "signup" ? "new-password" : "current-password"}
        hint={mode === "signup" ? "At least 12 characters. Length beats symbols." : undefined}
      />

      <Button type="submit" loading={busy} className="w-full">
        {mode === "signup" ? "Create account" : "Sign in"}
      </Button>

      <p className="text-center text-xs" style={{ color: "var(--text-secondary)" }}>
        {mode === "signup" ? (
          <>
            Already have an account?{" "}
            <Link href="/login" className="underline">
              Sign in
            </Link>
          </>
        ) : (
          <>
            No account yet?{" "}
            <Link href="/signup" className="underline">
              Create one
            </Link>
          </>
        )}
      </p>
    </form>
  );
}
