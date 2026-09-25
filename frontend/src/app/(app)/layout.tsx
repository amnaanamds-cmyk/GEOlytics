import { redirect } from "next/navigation";

import { AppNav } from "@/components/app-nav";
import { hasSession } from "@/lib/session";

export default async function AppLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  // Checked on the server so an unauthenticated visitor never receives the
  // dashboard shell at all. The API enforces this again on every call; this is
  // the navigation guard, not the security boundary.
  if (!(await hasSession())) redirect("/login");

  return (
    <div className="min-h-dvh">
      <AppNav />
      <main className="mx-auto w-full max-w-6xl px-4 py-8">{children}</main>
    </div>
  );
}
