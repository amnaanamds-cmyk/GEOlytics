"use client";

import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";

import { Button, NavLink } from "@/components/ui";

const LINKS = [
  { href: "/dashboard", label: "Audits" },
  { href: "/experiments", label: "Experiments" },
  { href: "/settings", label: "Settings" },
];

export function AppNav() {
  const pathname = usePathname();
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  async function signOut() {
    setBusy(true);
    await fetch("/api/auth/logout", { method: "POST" });
    router.replace("/login");
    router.refresh();
  }

  return (
    <header style={{ borderBottom: "1px solid var(--border)" }}>
      <nav
        className="mx-auto flex w-full max-w-6xl flex-wrap items-center gap-3 px-4 py-3"
        aria-label="Main"
      >
        <span className="mr-2 text-sm font-semibold tracking-tight">GEOlytics</span>
        <div className="flex flex-wrap gap-1">
          {LINKS.map((link) => (
            <NavLink
              key={link.href}
              href={link.href}
              active={pathname === link.href || pathname.startsWith(`${link.href}/`)}
            >
              {link.label}
            </NavLink>
          ))}
        </div>
        <div className="ml-auto">
          <Button variant="secondary" onClick={signOut} loading={busy}>
            Sign out
          </Button>
        </div>
      </nav>
    </header>
  );
}
