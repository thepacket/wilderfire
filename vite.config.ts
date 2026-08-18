import { defineConfig, type Plugin } from 'vite';
import { writeFileSync, mkdirSync } from 'node:fs';
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

export default defineConfig({
  plugins: [jwfReportSink(), compareSink(), baselineSink()],
  build: {
    // the JWildfire variation registry (variations.jwf.ts) is loaded lazily and is ~1.9 MB on purpose
    chunkSizeWarningLimit: 2100,
  },
  server: {
    port: Number(process.env.PORT) || 5173,
    strictPort: false,
  },
});
