/** @type {import('next').NextConfig} */
const nextConfig = {
  // Output as a standalone bundle for Docker — copies only the necessary files
  // into .next/standalone, which dramatically reduces the final image size.
  output: "standalone",

  // Allow the gateway URL to be configured at build time via environment variable
  env: {
    NEXT_PUBLIC_GATEWAY_URL: process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8000",
  },

  // Forward /api/* requests to the gateway during development to avoid CORS issues
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.GATEWAY_INTERNAL_URL ?? "http://gateway:8000"}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
