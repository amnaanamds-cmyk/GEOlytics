import { redirect } from "next/navigation";

import { AuthForm } from "@/components/auth-form";
import { AuthShell } from "@/components/auth-shell";
import { hasSession } from "@/lib/session";

export default async function LoginPage() {
  if (await hasSession()) redirect("/dashboard");
  return (
    <AuthShell title="Sign in" subtitle="Audit how your site reads to AI answer engines.">
      <AuthForm mode="login" />
    </AuthShell>
  );
}
