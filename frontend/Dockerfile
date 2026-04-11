# =============================================================================
# Frontend Dockerfile — Next.js 14 App Router
# Uses the official Node.js Alpine image.
# Multi-stage build: deps -> builder -> runtime
# The standalone output mode copies only the minimum required files.
# =============================================================================

# ---- Stage 1: install dependencies ----
FROM node:20-alpine AS deps

WORKDIR /app

# Copy package manifests only to leverage Docker layer caching
COPY package.json package-lock.json* ./
RUN npm ci --frozen-lockfile

# ---- Stage 2: build the Next.js app ----
FROM node:20-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build args injected by docker-compose
ARG NEXT_PUBLIC_GATEWAY_URL=http://localhost:8000
ENV NEXT_PUBLIC_GATEWAY_URL=$NEXT_PUBLIC_GATEWAY_URL

RUN npm run build

# ---- Stage 3: minimal runtime image ----
FROM node:20-alpine AS runtime

WORKDIR /app

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy the standalone bundle produced by `output: "standalone"` in next.config.js
COPY --from=builder --chown=appuser:appgroup /app/.next/standalone ./
COPY --from=builder --chown=appuser:appgroup /app/.next/static ./.next/static
COPY --from=builder --chown=appuser:appgroup /app/public ./public

USER appuser
EXPOSE 3000

ENV PORT=3000
ENV NODE_ENV=production

CMD ["node", "server.js"]
