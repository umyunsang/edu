#!/usr/bin/env node
/**
 * check_mermaid.mjs — 마크다운 노트 안의 ```mermaid 블록을 실제 Mermaid 파서로 검증한다.
 *
 * Obsidian에서 다이어그램이 깨진 채로 커밋되는 것을 막기 위한 검사기다.
 * 렌더링은 하지 않고 parse 단계만 돌리므로 빠르다.
 *
 * 사용법:
 *   node scripts/check_mermaid.mjs <file.md ...>
 *   node scripts/check_mermaid.mjs --dir "ComputerScience/04_systems-infrastructure/parallel-distributed-computing"
 *
 * 의존성: mermaid, jsdom (전역 설치 대신 --modules 로 경로 지정 가능)
 *   npm i mermaid@11 jsdom
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, extname } from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const argv = process.argv.slice(2);
let modulesDir = process.env.MERMAID_MODULES ?? null;
const files = [];
const dirs = [];

for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--dir") dirs.push(argv[++i]);
  else if (argv[i] === "--modules") modulesDir = argv[++i];
  else files.push(argv[i]);
}

function walk(dir) {
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".")) continue;
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p);
    else if (extname(p) === ".md") files.push(p);
  }
}
dirs.forEach(walk);

if (files.length === 0) {
  console.error("검사할 마크다운 파일이 없습니다.");
  process.exit(2);
}

const require_ = createRequire(
  modulesDir ? pathToFileURL(join(modulesDir, "noop.js")) : import.meta.url,
);
const { JSDOM } = require_("jsdom");
const dom = new JSDOM("<!DOCTYPE html><body></body>", { pretendToBeVisual: true });
for (const k of ["window", "document", "navigator", "DOMPurify", "Element", "SVGElement"]) {
  if (k === "window") globalThis.window = dom.window;
  else if (k === "document") globalThis.document = dom.window.document;
  else if (globalThis[k] === undefined && dom.window[k] !== undefined) globalThis[k] = dom.window[k];
}

const mermaid = (await import(pathToFileURL(require_.resolve("mermaid")).href)).default;
mermaid.initialize({ startOnLoad: false, securityLevel: "loose" });

const BLOCK = /```mermaid\r?\n([\s\S]*?)```/g;
let total = 0;
let failed = 0;

for (const file of files) {
  const src = readFileSync(file, "utf8");
  const lines = src.split("\n");
  let m;
  BLOCK.lastIndex = 0;
  while ((m = BLOCK.exec(src)) !== null) {
    total++;
    const lineNo = src.slice(0, m.index).split("\n").length;
    const code = m[1];
    try {
      await mermaid.parse(code);
    } catch (err) {
      failed++;
      const kind = code.trim().split(/\s|\n/)[0];
      console.log(`FAIL ${file}:${lineNo} [${kind}]`);
      console.log(`     ${String(err?.message ?? err).split("\n").slice(0, 4).join("\n     ")}`);
    }
  }
  void lines;
}

console.log(`\nmermaid 블록 ${total}개 검사 · 실패 ${failed}개`);
process.exit(failed > 0 ? 1 : 0);
