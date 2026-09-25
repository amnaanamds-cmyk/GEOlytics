import { redirect } from "next/navigation";

import { AuthForm } from "@/components/auth-form";
import { AuthShell } from "@/components/auth-shell";
import { hasSession } from "@/lib/session";

export default async function SignupPage() {
  if (await hasSession()) redirect("/dashboard");
  return (
    <AuthShell
      title="Create your account"
      subtitle="Start on the free plan. No card required."
    >
      <AuthForm mode="signup" />
    </AuthShell>
  );
}
