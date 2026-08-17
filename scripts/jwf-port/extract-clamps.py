#!/usr/bin/env python3
"""Extract per-parameter clamps from JWildfire's Java setParameter() bodies
(Tools.limitValue / limitIntValue / limitVal / limitIntVal) into
data/param-clamps.json. The GPU snippets read params raw; gen.ts wraps reads
in the same clamps so out-of-range values behave like JWildfire's CPU.
Usage: extract-clamps.py <path to jwf/src/org/jwildfire/create/tina/variation>"""
import re, sys, json, glob, os
J = sys.argv[1]
names = set()
for line in open(os.path.join(os.path.dirname(__file__), 'data/jwf-variations.jsonl')):
    if line.strip(): names.add(json.loads(line)['name'])
files = {}
for f in glob.glob(J + '/*.java'):
    s = open(f, encoding='latin-1').read()
    for m in re.finditer(r'return\s+"([A-Za-z0-9_]+)"\s*;', s): files.setdefault(m.group(1), f)
found = {}
for n in sorted(names):
    f = files.get(n)
    if not f: continue
    s = open(f, encoding='latin-1').read()
    consts = dict(re.findall(r'String\s+(PARAM_[A-Za-z0-9_]+)\s*=\s*"([^"]+)"', s))
    i = s.find('void setParameter(')
    if i < 0: continue
    body = s[i:i + 6000]
    clamps = {}
    for stmt in body.split(';'):
        pm = re.search(r'(PARAM_[A-Za-z0-9_]+)', stmt)
        lm = re.search(r'limit(?:Int)?Val(?:ue)?\s*\(\s*(?:Tools\.FTOI\()?\s*pValue\s*\)?\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)', stmt)
        if pm and lm and consts.get(pm.group(1)):
            clamps[consts[pm.group(1)]] = [float(lm.group(1)), float(lm.group(2))]
    if clamps: found[n] = clamps
out = os.path.join(os.path.dirname(__file__), 'data/param-clamps.json')
json.dump(found, open(out, 'w'), indent=0, sort_keys=True)
print(f'{len(found)} variations with clamps → {out}')
