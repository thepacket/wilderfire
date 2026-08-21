# WilderFire — static build served by nginx.
# All rendering happens client-side in WebGPU; this image only serves files.

FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Assets only: tsc + vitest are pre-push checks and need tests/ + scripts/, which
# .dockerignore keeps out of the build context (they are 340 MB of dev tooling).
RUN npx vite build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
