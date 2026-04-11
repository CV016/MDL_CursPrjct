/**
 * app/layout.tsx
 * Root layout for the Next.js 14 App Router.
 * Applies global styles and wraps every page with a consistent shell.
 */

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI-DOC INTERACT",
  description: "Intelligent document analysis — upload, summarise, and generate questions.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col">
        {/* Top navigation bar */}
        <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur-sm">
          <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-6">
            <a href="/" className="flex items-center gap-2 font-semibold text-brand-700 text-lg">
              <span className="rounded bg-brand-600 px-2 py-0.5 text-white text-sm font-bold tracking-wide">
                AI-DOC
              </span>
              <span className="text-slate-800">INTERACT</span>
            </a>
            <nav className="flex items-center gap-4 text-sm text-slate-600">
              <a href="/" className="hover:text-brand-600 transition-colors">Upload</a>
              <a href="/login" className="hover:text-brand-600 transition-colors">Sign In</a>
            </nav>
          </div>
        </header>

        {/* Main content */}
        <main className="flex-1 mx-auto w-full max-w-5xl px-6 py-10">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-200 bg-white">
          <div className="mx-auto max-w-5xl px-6 py-4 text-center text-xs text-slate-400">
            AI-DOC INTERACT — MLOps-first intelligent document platform
          </div>
        </footer>
      </body>
    </html>
  );
}
