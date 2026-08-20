#!/usr/bin/env node
/**
 * Static-export checks for the MoCoO package-index site.
 * Usage: node scripts/verify-export.mjs
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const out = join(process.cwd(), 'out');
const required = [
  'index.html',
  'results/index.html',
  'methods/index.html',
  'evidence/index.html',
  'claims/index.html',
  '.nojekyll',
];
const forbidden = ['abstract', 'cite', 'team'];
const denylist = ['PEERJ_REVIEWER_FAQ.md', 'PEERJ_PORTAL_INPUTS.txt', 'superpowers'];
const leak = [
  /unpublished results/i,
  /iLISI/i,
  /0\.117/,
  /0\.167/,
  /Fig\.?\s*7/i,
  /jbhi/i,
  /NUMBER-LOCK/i,
  /Science Gateway/i,
  /cell-state proof/i,
  /IRALL/i,
  /Tables?\s+[IVXL]+(?:[–-]\s*[IVXL]+)?\b/i,
  /\/figures\//i,
  /F0[0-9]_/i,
];

let failed = 0;

for (const rel of required) {
  const p = join(out, rel);
  if (!existsSync(p)) {
    console.error(`FAIL: missing ${rel}`);
    failed += 1;
  }
}

for (const dir of forbidden) {
  if (existsSync(join(out, dir))) {
    console.error(`FAIL: forbidden route directory out/${dir}/`);
    failed += 1;
  }
}

if (existsSync(join(out, 'figures'))) {
  console.error('FAIL: unpublished figure directory out/figures/');
  failed += 1;
}

function walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (denylist.includes(entry.name)) {
        console.error(`FAIL: denylist dir ${p}`);
        failed += 1;
      }
      walk(p);
    } else if (denylist.some((d) => entry.name.includes(d))) {
      console.error(`FAIL: denylist file ${p}`);
      failed += 1;
    } else if (/\.(html|txt|js|css)$/.test(entry.name)) {
      const text = readFileSync(p, 'utf8');
      for (const re of leak) {
        if (re.test(text)) {
          console.error(`FAIL: leak ${re} in ${p}`);
          failed += 1;
        }
      }
    }
  }
}

if (existsSync(out)) {
  walk(out);
  const html = readFileSync(join(out, 'index.html'), 'utf8');
  if (/github\.com\/PeterPonyu\/HetCLOP/i.test(html)) {
    console.error('FAIL: private HetCLOP Code href in index.html');
    failed += 1;
  }
  for (const label of ['Abstract', 'Cite', 'Team']) {
    if (new RegExp(`>${label}<`, 'i').test(html)) {
      console.error(`FAIL: journal nav label "${label}" in index.html`);
      failed += 1;
    }
  }
  if (/Get started|Try now|Launch/i.test(html)) {
    console.error('FAIL: product headline pattern in index.html');
    failed += 1;
  }
  if (!/data-site-binding="mocoo-pypi-index"/.test(html)) {
    console.error('FAIL: missing mocoo-pypi-index binding on home');
    failed += 1;
  }
  if (!statSync(join(out, 'index.html')).size) {
    console.error('FAIL: empty index.html');
    failed += 1;
  }
}

if (failed) {
  process.exit(1);
}

console.log(`verify-export: ok (${required.length} required paths, no result gallery)`);
