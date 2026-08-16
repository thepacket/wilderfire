import { defineConfig, type Plugin } from 'vite';
import { writeFileSync } from 'node:fs';
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

export default defineConfig({
  plugins: [jwfReportSink()],
  server: {
    port: Number(process.env.PORT) || 5173,
    strictPort: false,
  },
});
