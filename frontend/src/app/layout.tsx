import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "GEOlytics",
  description:
    "Automated Generative Engine Optimization and RAG auditing for your website.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-dvh antialiased">{children}</body>
    </html>
  );
}
