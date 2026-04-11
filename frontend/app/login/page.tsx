/**
 * app/login/page.tsx
 * Login form — POSTs credentials to the auth service and stores the
 * returned JWT in an httpOnly cookie via the /api/auth/set-cookie route.
 *
 * For SSR-safe authentication the JWT is stored server-side in an httpOnly
 * cookie (not accessible from JavaScript) using a lightweight Next.js API
 * route that sets the cookie header.
 */

"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

interface LoginFormState {
  username: string;
  password: string;
}

export default function LoginPage() {
  const router = useRouter();
  const [form, setForm] = useState<LoginFormState>({ username: "", password: "" });
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      // POST credentials directly to the auth service via the Next.js rewrite proxy
      const resp = await fetch("/api/auth/token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: form.username, password: form.password }),
      });

      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error((body as { detail?: string }).detail ?? "Login failed.");
      }

      const data = (await resp.json()) as { access_token: string; token_type: string };

      // Persist the JWT via a server-side API route that sets an httpOnly cookie,
      // keeping the raw token out of client-side JavaScript.
      await fetch("/api/auth/set-cookie", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: data.access_token }),
      });

      router.push("/");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "An unexpected error occurred.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="card w-full max-w-md">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-slate-900">Sign in</h1>
          <p className="text-sm text-slate-400 mt-1">
            Enter your credentials to access the document analysis platform.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          {/* Username */}
          <div className="flex flex-col gap-1.5">
            <label htmlFor="username" className="text-sm font-medium text-slate-700">
              Username
            </label>
            <input
              id="username"
              name="username"
              type="text"
              autoComplete="username"
              required
              value={form.username}
              onChange={handleChange}
              className="input-field"
              placeholder="your-username"
            />
          </div>

          {/* Password */}
          <div className="flex flex-col gap-1.5">
            <label htmlFor="password" className="text-sm font-medium text-slate-700">
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              required
              value={form.password}
              onChange={handleChange}
              className="input-field"
              placeholder="••••••••"
            />
          </div>

          {/* Error message */}
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {/* Submit */}
          <button type="submit" disabled={loading} className="btn-primary py-3 mt-2">
            {loading ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}
