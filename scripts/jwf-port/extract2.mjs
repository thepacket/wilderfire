// usage: node extract2.mjs '<regex on the flame xml>' <outId> [skip]  — first corpus flame matching, with no unportable variation
import fs from 'fs'; import { execSync } from 'child_process';
const [,, rx, out, skipS] = process.argv; let skip = +(skipS||0); const re = new RegExp(rx);
const unpNames = new Set([...fs.readFileSync('src/core/variations.unportable.ts','utf8').matchAll(/^\s+"([A-Za-z0-9_]+)":/gm)].map(m=>m[1]));
const files = execSync(`grep -rlE '${rx.replace(/'/g, "'\\''")}' ~/Projects/frames/packs`, {encoding:'utf8'}).trim().split('\n').filter(Boolean);
for (const f of files) {
  const xml = fs.readFileSync(f,'utf8');
  for (const m of xml.matchAll(/<(?:jwf-)?flame [\s\S]*?<\/(?:jwf-)?flame>/g)) {
    const fl = m[0]; if (!re.test(fl)) continue;
    const attrs = new Set([...fl.matchAll(/ ([A-Za-z0-9_]+)="/g)].map(x=>x[1]));
    const unp = [...attrs].filter(a=>unpNames.has(a));
    if (unp.length) continue;
    if (skip-- > 0) continue;
    const name = /name="([^"]*)"/.exec(fl)?.[1];
    console.log('PICK', out, '|', f.replace(/.*packs\//,''), '|', name, '| xforms', (fl.match(/<xform /g)||[]).length, 'finals', (fl.match(/<finalxform /g)||[]).length, '| size', /size="([^"]*)"/.exec(fl)?.[1], '| layers', (fl.match(/<layer /g)||[]).length);
    fs.writeFileSync('scripts/jwf-port/testflames/'+out+'.flame', '<flames>\n'+fl+'\n</flames>\n');
    process.exit(0);
  }
}
console.log('NONE for', rx);
