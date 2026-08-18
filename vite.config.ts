import { defineConfig, type Plugin } from 'vite';
import { writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';

/** Dev-only: lets the in-browser variation oracle harness (src/dev/varTest.ts)
 *  persist its verdicts to scripts/jwf-port/verified.json. */
function jwfReportSink(): Plugin {
  return {
    name: 'wilderfire-jwf-report-sink',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__jwf/verified', (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }
        let body = '';
        req.on('data', (c) => { body += c; });
        req.on('end', () => {
          try {
            const parsed = JSON.parse(body);
            const file = resolve(import.meta.dirname, 'scripts/jwf-port/verified.json');
            writeFileSync(file, JSON.stringify(parsed, null, 1) + '\n');
            res.setHeader('content-type', 'application/json');
            res.end(JSON.stringify({ ok: true, file }));
          } catch (err) {
            res.statusCode = 400;
            res.end(String(err));
          }
        });
      });
    },
  };
}

/** Dev-only: file sink for the flame comparison harness (src/dev/flameCompare.ts) —
 *  writes PNG/XML into compare-out/<name> (gitignored; the name is sanitised). */
function compareSink(): Plugin {
  return {
    name: 'wilderfire-compare-sink',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__jwf/save', (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }
        const url = new URL(req.url ?? '', 'http://x');
        const name = (url.searchParams.get('name') ?? '').replace(/[^A-Za-z0-9_.-]/g, '_');
        if (!name || name.startsWith('.')) { res.statusCode = 400; res.end('bad name'); return; }
        const chunks: Buffer[] = [];
        req.on('data', (c) => chunks.push(Buffer.from(c)));
        req.on('end', () => {
          const dir = resolve(import.meta.dirname, 'compare-out');
          mkdirSync(dir, { recursive: true });
          const file = resolve(dir, name);
          writeFileSync(file, Buffer.concat(chunks));
          res.setHeader('content-type', 'application/json');
          res.end(JSON.stringify({ ok: true, file, bytes: chunks.reduce((a, b) => a + b.length, 0) }));
        });
      });
    },
  };
}

/** Dev-only: writes the render-regression baseline (src/dev/renderCheck.ts) to scripts/jwf-port/render-baseline.json. */
function baselineSink(): Plugin {
  return {
    name: 'wilderfire-baseline-sink',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__jwf/baseline', (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }
        let body = '';
        req.on('data', (c) => { body += c; });
        req.on('end', () => {
          try {
            JSON.parse(body);
            const file = resolve(import.meta.dirname, 'scripts/jwf-port/render-baseline.json');
            writeFileSync(file, body.endsWith('\n') ? body : body + '\n');
            res.end(JSON.stringify({ ok: true, file }));
          } catch (err) { res.statusCode = 400; res.end(String(err)); }
        });
      });
    },
  };
}

/**
 * Build-only: emits `sw.js`, a service worker that precaches this build's hashed assets
 * (+ the page, manifest, icon) so the app opens offline. Strategy: the page itself is
 * network-first (a deploy takes effect on the next load; the cached copy is the offline
 * fallback), hashed /assets/ are cache-first (immutable), /flames/ samples are
 * stale-while-revalidate. Each build gets its own cache name; activation deletes the rest.
 * Not registered in dev (main.ts guards on import.meta.env.PROD).
 */
function serviceWorker(): Plugin {
  return {
    name: 'wilderfire-service-worker',
    apply: 'build',
    generateBundle(_o, bundle) {
      // (the dev harness chunks — varTest/flameTest/flameCompare/renderCheck — are lazy dev tools, not precached)
      const files = Object.keys(bundle).filter((f) => !f.endsWith('.map') && !/\/(varTest|flameTest|flameCompare|renderCheck)-/.test(f));
      const version = files.map((f) => f.split('-').pop()).join('|');
      let hash = 0;
      for (const c of version) hash = (hash * 31 + c.charCodeAt(0)) | 0;
      const cacheName = `wilderfire-${(hash >>> 0).toString(36)}`;
      // the bundled sample flames too (~1 MB), so the Tests menu works offline
      const flames = readdirSync(resolve(import.meta.dirname, 'public/flames')).filter((f) => f.endsWith('.flame')).map((f) => '/flames/' + f);
      const precache = ['/', '/index.html', '/manifest.webmanifest', '/icon.svg', ...files.map((f) => '/' + f), ...flames];
      const sw = `// WilderFire service worker (generated at build time by vite.config.ts)
const CACHE = ${JSON.stringify(cacheName)};
const PRECACHE = ${JSON.stringify(precache)};
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
// ignoreVary: the server may answer with \`Vary: Accept-Encoding\`, and a request created inside the worker
// (precache) carries no such header — a strict match would then miss every precached asset offline
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return; // OpenRouter / local AI servers: never touched
  if (req.mode === 'navigate' || url.pathname === '/' || url.pathname === '/index.html') {
    // page: network first (deploys land on the next load), cached copy offline
    e.respondWith(fetch(req).then((r) => { const copy = r.clone(); caches.open(CACHE).then((c) => c.put('/', copy)); return r; }).catch(() => caches.match('/', { ignoreVary: true })));
    return;
  }
  if (url.pathname.startsWith('/assets/')) {
    // hashed, immutable: cache first
    e.respondWith(caches.match(req, { ignoreVary: true }).then((hit) => hit || fetch(req).then((r) => { const copy = r.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); return r; })));
    return;
  }
  if (url.pathname.startsWith('/flames/') || url.pathname === '/manifest.webmanifest' || url.pathname === '/icon.svg') {
    // samples: stale-while-revalidate
    e.respondWith(caches.match(req, { ignoreVary: true }).then((hit) => {
      const net = fetch(req).then((r) => { const copy = r.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); return r; }).catch(() => hit);
      return hit || net;
    }));
  }
});
`;
      this.emitFile({ type: 'asset', fileName: 'sw.js', source: sw });
    },
  };
}

export default defineConfig({
  plugins: [jwfReportSink(), compareSink(), baselineSink(), serviceWorker()],
  build: {
    // the JWildfire variation registry (variations.jwf.ts) is loaded lazily and is ~1.9 MB on purpose
    chunkSizeWarningLimit: 2100,
  },
  server: {
    port: Number(process.env.PORT) || 5173,
    strictPort: false,
  },
});
