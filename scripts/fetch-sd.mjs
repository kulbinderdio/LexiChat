#!/usr/bin/env node
// Fetch the prebuilt stable-diffusion.cpp engine for THIS platform and stage it as a Tauri resource
// (src-tauri/resources/image-gen/), so `tauri build` bundles it and the app provisions it on first
// launch (see image_gen::provision_from_resources). Run in CI before `tauri build`, and locally to
// test bundling. Not committed — the staged files are gitignored.
//
// Prebuilts exist only for macOS-arm64 (Metal), Windows-x64 (Vulkan), and Linux-x64 (Vulkan). On
// any other target (Intel Mac, Linux-arm64) this is a graceful no-op: the app simply falls back to
// the in-app model/engine download or a setup message.
//
// Usage: node scripts/fetch-sd.mjs   (optional env SD_RELEASE_TAG to pin a different release)
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";

const TAG = process.env.SD_RELEASE_TAG || "master-820-de298c2";
const REPO = "leejet/stable-diffusion.cpp";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const destDir = path.join(root, "src-tauri", "resources", "image-gen");

// Pick the release asset for this platform. `match` runs over asset names; `null` = unsupported.
function pickMatcher() {
  const p = process.platform, a = process.arch;
  if (p === "darwin" && a === "arm64") return n => /Darwin/i.test(n) && /arm64/i.test(n);
  if (p === "win32" && a === "x64")
    return [n => /win/i.test(n) && /vulkan/i.test(n) && /x64/i.test(n),   // prefer GPU (Vulkan)
            n => /win/i.test(n) && /cpu/i.test(n) && /x64/i.test(n)];      // fallback CPU
  if (p === "linux" && a === "x64")
    return [n => /Linux/i.test(n) && /vulkan/i.test(n) && /x86_64/i.test(n),
            n => /Linux/i.test(n) && /x86_64/i.test(n) && !/rocm|cuda/i.test(n)];
  return null; // Intel Mac, Linux-arm64, etc. — no prebuilt
}

function extract(zip, into) {
  fs.mkdirSync(into, { recursive: true });
  if (process.platform === "linux") execSync(`unzip -o "${zip}" -d "${into}"`, { stdio: "inherit" });
  else execSync(`tar -xf "${zip}" -C "${into}"`, { stdio: "inherit" }); // bsdtar reads zip on mac/win
}

// Recursively list files under a dir.
function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(e => {
    const fp = path.join(dir, e.name);
    return e.isDirectory() ? walk(fp) : [fp];
  });
}

async function main() {
  const matcher = pickMatcher();
  if (!matcher) {
    console.log(`[fetch-sd] no prebuilt engine for ${process.platform}/${process.arch} — skipping (app will fall back).`);
    return;
  }
  const matchers = Array.isArray(matcher) ? matcher : [matcher];

  console.log(`[fetch-sd] release ${TAG} for ${process.platform}/${process.arch}…`);
  const rel = await fetch(`https://api.github.com/repos/${REPO}/releases/tags/${TAG}`, {
    headers: { "User-Agent": "lexichat-build", ...(process.env.GITHUB_TOKEN ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` } : {}) },
  }).then(r => r.json());
  const assets = rel.assets || [];
  let asset;
  for (const m of matchers) { asset = assets.find(x => m(x.name)); if (asset) break; }
  if (!asset) {
    console.log(`[fetch-sd] no matching asset in ${TAG} (${assets.map(a => a.name).join(", ")}) — skipping.`);
    return;
  }
  console.log(`[fetch-sd] downloading ${asset.name} (${Math.round(asset.size / 1e6)} MB)…`);

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "sd-"));
  const zip = path.join(tmp, asset.name);
  const buf = Buffer.from(await fetch(asset.browser_download_url, { headers: { "User-Agent": "lexichat-build" } }).then(r => r.arrayBuffer()));
  fs.writeFileSync(zip, buf);
  const unz = path.join(tmp, "x");
  extract(zip, unz);

  // Locate the CLI executable + shared libraries in the extracted tree.
  const files = walk(unz);
  const exe = files.find(f => /(^|[\/\\])(sd-cli|sd)(\.exe)?$/i.test(f));
  if (!exe) { console.error("[fetch-sd] could not find the sd executable in the archive"); process.exit(1); }
  const libs = files.filter(f => /\.(dylib|dll)$/i.test(f) || /\.so(\.\d+)*$/i.test(f));

  fs.rmSync(destDir, { recursive: true, force: true });
  fs.mkdirSync(destDir, { recursive: true });
  const exeName = process.platform === "win32" ? "sd.exe" : "sd";
  fs.copyFileSync(exe, path.join(destDir, exeName));
  if (process.platform !== "win32") fs.chmodSync(path.join(destDir, exeName), 0o755);
  for (const l of libs) fs.copyFileSync(l, path.join(destDir, path.basename(l)));

  console.log(`[fetch-sd] staged ${[exeName, ...libs.map(l => path.basename(l))].join(", ")} → ${path.relative(root, destDir)}`);
  fs.rmSync(tmp, { recursive: true, force: true });
}

main().catch(e => { console.error("[fetch-sd] failed:", e); process.exit(1); });
