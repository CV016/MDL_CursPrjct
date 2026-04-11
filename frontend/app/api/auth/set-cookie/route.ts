/**
 * app/api/auth/set-cookie/route.ts
 * Next.js Route Handler — receives the JWT from the client-side login form
 * and stores it in an httpOnly, Secure, SameSite=Lax cookie so that the raw
 * token is never accessible to client-side JavaScript (mitigates XSS risk).
 */

import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest): Promise<NextResponse> {
  let body: { token?: string };
  try {
    body = (await req.json()) as { token?: string };
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const token = body.token;
  if (!token || typeof token !== "string") {
    return NextResponse.json({ error: "Missing token field." }, { status: 400 });
  }

  const response = NextResponse.json({ message: "Cookie set." });

  // httpOnly prevents JavaScript access; Secure ensures the cookie is only
  // sent over HTTPS in production; SameSite=Lax guards against CSRF.
  response.cookies.set("access_token", token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    // 1 hour — matches the JWT expiry
    maxAge: 60 * 60,
    path: "/",
  });

  return response;
}
