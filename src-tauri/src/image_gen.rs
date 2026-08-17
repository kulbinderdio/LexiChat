// Local, offline image generation by shelling out to a `stable-diffusion.cpp` ("sd") binary — the
// "llama.cpp of images". No cloud API and no companion GUI (ComfyUI/A1111): LexiChat invokes the
// located `sd` CLI with a GGUF/safetensors diffusion model and reads back a PNG, which then renders
// inline via the same tool-image path as matplotlib charts and Mapbox static maps.
//
// The binary and model are NOT bundled (they're large + platform-specific); they're located from,
// in order: the user's explicit config path, the app data dir (`<data>/lexichat/image-gen/…`), or
// PATH (binary only). When neither is found, generate() returns a clear setup message the model
// relays to the user rather than a cryptic failure.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Per-install image-generation settings (held in AppState, pushed from the frontend). All fields
/// optional/defaulted so an empty config is valid and simply triggers auto-detection.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ImageGenConfig {
    /// Path to the `sd` (stable-diffusion.cpp) executable. Empty → auto-detect (data dir, then PATH).
    #[serde(default)] pub binary_path: String,
    /// Path to the diffusion model (.gguf / .safetensors / .ckpt). Empty → first model file found
    /// in `<data>/lexichat/image-gen/models/`.
    #[serde(default)] pub model_path: String,
    /// Default sampling steps. Turbo models want ~4; classic SD ~20. 0 → 20.
    #[serde(default)] pub steps: u32,
    /// Default square image size in px (512 / 768 / 1024). 0 → 512.
    #[serde(default)] pub size: u32,
    /// CFG (classifier-free guidance) scale. Turbo models want ~1.0; classic ~7. 0 → 7.
    #[serde(default)] pub cfg_scale: f32,
    /// Extra raw CLI args for advanced users (space-split), e.g. "--vae path --clip_l path".
    #[serde(default)] pub extra_args: String,
}

fn image_gen_dir() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("lexichat").join("image-gen"))
}

/// Where downloaded models live (auto-detected by resolve_model).
pub fn models_dir() -> Option<PathBuf> {
    image_gen_dir().map(|d| d.join("models"))
}

/// A sensible, well-known default model to pre-fill the download field. SD-Turbo is compact and
/// fast (~1-4 steps). It's editable in the UI — some models require sign-in/licence acceptance, in
/// which case the user pastes a direct URL to an ungated .safetensors/.gguf instead.
pub const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/stabilityai/sd-turbo/resolve/main/sd_turbo.safetensors";

/// Stream-download a model file into the models dir, reporting progress via `on_progress(received,
/// total)` and honoring `cancel`. Writes to a `.part` file and renames on success (so a cancelled
/// or failed download never looks like a complete model). Returns the final path.
pub async fn download_model(
    url: &str,
    filename: Option<&str>,
    cancel: &std::sync::atomic::AtomicBool,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    use std::io::Write;
    use std::sync::atomic::Ordering;

    let dir = models_dir().ok_or("Cannot resolve the app data directory.")?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create models dir: {e}"))?;

    // Derive a safe filename from the arg or the URL's last path segment.
    let raw = filename
        .map(str::to_string)
        .filter(|s| !s.trim().is_empty())
        .or_else(|| url.split('/').next_back().map(|s| s.split('?').next().unwrap_or(s).to_string()))
        .filter(|s| !s.is_empty())
        .ok_or("Cannot determine a filename from the URL — set one explicitly.")?;
    let name = std::path::Path::new(&raw)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.safetensors")
        .to_string();
    let dest = dir.join(&name);
    let part = dir.join(format!("{name}.part"));

    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    // Already have a complete copy? Skip the (multi-GB) transfer — HEAD the URL and, if the local
    // file matches the server's size, just report done. Makes "Download & use" idempotent so
    // re-selecting an installed model is instant instead of re-downloading it.
    if dest.is_file() {
        if let Ok(head) = client.head(url).send().await {
            if let (Some(remote), Ok(meta)) = (head.content_length(), std::fs::metadata(&dest)) {
                if remote > 0 && meta.len() == remote {
                    on_progress(remote, Some(remote));
                    return Ok(dest);
                }
            }
        }
    }

    let mut resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "Download failed: HTTP {}. The model URL may be wrong, or the model requires \
             sign-in/licence acceptance — paste a direct link to an ungated .safetensors/.gguf.",
            resp.status()
        ));
    }
    let total = resp.content_length();
    let mut file = std::fs::File::create(&part).map_err(|e| format!("Cannot write file: {e}"))?;
    let mut received: u64 = 0;
    on_progress(0, total);
    loop {
        if cancel.load(Ordering::SeqCst) {
            drop(file);
            let _ = std::fs::remove_file(&part);
            return Err("Download cancelled.".to_string());
        }
        match resp.chunk().await.map_err(|e| format!("Download interrupted: {e}"))? {
            Some(chunk) => {
                file.write_all(&chunk).map_err(|e| format!("Write error: {e}"))?;
                received += chunk.len() as u64;
                on_progress(received, total);
            }
            None => break,
        }
    }
    file.flush().ok();
    drop(file);
    // Guard against a truncated download (connection closed early): if the server told us the size,
    // require we got all of it before promoting the .part file — a short file yields a corrupt model
    // that the engine rejects cryptically.
    if let Some(total) = total {
        if received < total {
            let _ = std::fs::remove_file(&part);
            return Err(format!("Download incomplete: received {received} of {total} bytes — please try again."));
        }
    }
    std::fs::rename(&part, &dest).map_err(|e| format!("Cannot finalize the model file: {e}"))?;
    Ok(dest)
}

/// Find an executable by name on the PATH (no extra crate). Returns the first hit.
fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Resolve the sd binary: explicit config → app data dir (where the bundled engine is provisioned
/// on first launch, or the user dropped it) → PATH.
fn resolve_binary(cfg: &ImageGenConfig) -> Option<PathBuf> {
    let explicit = cfg.binary_path.trim();
    if !explicit.is_empty() {
        // An explicit path is authoritative: use it, or fail (surfacing the config error) rather
        // than silently auto-detecting something else. Matches resolve_model.
        let p = PathBuf::from(explicit);
        return p.is_file().then_some(p);
    }
    if let Some(dir) = image_gen_dir() {
        for name in ["sd", "sd.exe", "stable-diffusion", "stable-diffusion.exe"] {
            let p = dir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    find_on_path("sd").or_else(|| find_on_path("stable-diffusion"))
}

/// Resolve the model: explicit config → first model file in `<data>/…/image-gen/models/`.
fn resolve_model(cfg: &ImageGenConfig) -> Option<PathBuf> {
    let explicit = cfg.model_path.trim();
    if !explicit.is_empty() {
        let p = PathBuf::from(explicit);
        return p.is_file().then_some(p);
    }
    let models = image_gen_dir()?.join("models");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&models)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| matches!(x.to_ascii_lowercase().as_str(), "gguf" | "safetensors" | "ckpt"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort();
    entries.into_iter().next()
}

/// Pinned stable-diffusion.cpp release the engine is fetched from on first use.
const ENGINE_RELEASE_TAG: &str = "master-820-de298c2";
const ENGINE_REPO: &str = "leejet/stable-diffusion.cpp";

/// Whether an sd engine is installed in the app data dir (drives the Images-tab status).
pub fn image_gen_dir_has_engine() -> bool {
    image_gen_dir()
        .map(|d| d.join(if cfg!(windows) { "sd.exe" } else { "sd" }).is_file())
        .unwrap_or(false)
}

/// Is a prebuilt engine available for this platform? (macOS-arm64/Metal, Windows-x64/Vulkan,
/// Linux-x64/Vulkan.) Other targets have no upstream prebuilt.
pub fn engine_supported() -> bool {
    use std::env::consts::{ARCH, OS};
    matches!((OS, ARCH), ("macos", "aarch64") | ("windows", "x86_64") | ("linux", "x86_64"))
}

/// Whether a release-asset name is the right engine build for this platform. Returns a priority
/// (lower = preferred: GPU build before CPU) or None if it doesn't match.
fn engine_asset_priority(name: &str) -> Option<u8> {
    use std::env::consts::{ARCH, OS};
    let n = name.to_ascii_lowercase();
    match (OS, ARCH) {
        ("macos", "aarch64") if n.contains("darwin") && n.contains("arm64") => Some(0),
        ("windows", "x86_64") if n.contains("win") && n.contains("vulkan") && n.contains("x64") => Some(0),
        ("windows", "x86_64") if n.contains("win") && n.contains("cpu") && n.contains("x64") => Some(1),
        ("linux", "x86_64") if n.contains("linux") && n.contains("vulkan") && n.contains("x86_64") => Some(0),
        ("linux", "x86_64") if n.contains("linux") && n.contains("x86_64") && !n.contains("rocm") && !n.contains("cuda") => Some(1),
        _ => None,
    }
}

/// Download and install the offline image engine (stable-diffusion.cpp) for this platform into the
/// app data dir on first use — the app ships without it (bundling unsigned binaries breaks macOS
/// notarization). Small (~50 MB), one-time. Reports progress and honors `cancel`.
pub async fn download_engine(
    cancel: &std::sync::atomic::AtomicBool,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    use std::io::Write;
    use std::sync::atomic::Ordering;

    if !engine_supported() {
        return Err("The offline image engine isn't available for this platform yet.".to_string());
    }
    let dest = image_gen_dir().ok_or("Cannot resolve the app data directory.")?;
    std::fs::create_dir_all(&dest).map_err(|e| format!("Cannot create image-gen dir: {e}"))?;

    let client = reqwest::Client::builder().build().map_err(|e| format!("HTTP client error: {e}"))?;

    // 1. Find the matching asset in the pinned release.
    let rel_url = format!("https://api.github.com/repos/{ENGINE_REPO}/releases/tags/{ENGINE_RELEASE_TAG}");
    let rel: serde_json::Value = client.get(&rel_url).header("User-Agent", "lexichat")
        .send().await.map_err(|e| format!("Cannot reach the release server: {e}"))?
        .json().await.map_err(|e| format!("Unexpected release data: {e}"))?;
    let mut best: Option<(u8, String)> = None;
    for a in rel.get("assets").and_then(|a| a.as_array()).map(|v| v.as_slice()).unwrap_or(&[]) {
        let name = a.get("name").and_then(|n| n.as_str()).unwrap_or("");
        if let Some(p) = engine_asset_priority(name) {
            if best.as_ref().map(|(bp, _)| p < *bp).unwrap_or(true) {
                let url = a.get("browser_download_url").and_then(|u| u.as_str()).unwrap_or("").to_string();
                best = Some((p, url));
            }
        }
    }
    let (_, url) = best.ok_or("No matching engine build found in the release.")?;

    // 2. Download the zip to a temp file (progress + cancel + truncation guard).
    let mut resp = client.get(&url).header("User-Agent", "lexichat").send().await
        .map_err(|e| format!("Engine download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Engine download failed: HTTP {}", resp.status()));
    }
    let total = resp.content_length();
    let tmp_zip = dest.join("engine-download.zip.part");
    let mut file = std::fs::File::create(&tmp_zip).map_err(|e| format!("Cannot write engine file: {e}"))?;
    let mut received: u64 = 0;
    on_progress(0, total);
    loop {
        if cancel.load(Ordering::SeqCst) {
            drop(file); let _ = std::fs::remove_file(&tmp_zip);
            return Err("Engine download cancelled.".to_string());
        }
        match resp.chunk().await.map_err(|e| format!("Engine download interrupted: {e}"))? {
            Some(chunk) => { file.write_all(&chunk).map_err(|e| format!("Write error: {e}"))?; received += chunk.len() as u64; on_progress(received, total); }
            None => break,
        }
    }
    file.flush().ok(); drop(file);
    if let Some(t) = total { if received < t {
        let _ = std::fs::remove_file(&tmp_zip);
        return Err(format!("Engine download incomplete ({received}/{t} bytes) — please retry."));
    }}

    // 3. Extract the sd executable + its shared libraries into the image-gen dir.
    let sd = extract_engine(&tmp_zip, &dest);
    let _ = std::fs::remove_file(&tmp_zip);
    let sd = sd?;

    // 4. Executable bit + de-quarantine (so a downloaded binary can run on macOS).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(&sd) { let mut p = meta.permissions(); p.set_mode(0o755); let _ = std::fs::set_permissions(&sd, p); }
    }
    #[cfg(target_os = "macos")]
    { let _ = std::process::Command::new("xattr").arg("-dr").arg("com.apple.quarantine").arg(&dest).status(); }

    Ok(sd)
}

/// Extract the sd executable (normalized to `sd`/`sd.exe`) and its shared libraries from the release
/// zip into `dest`. Returns the path to the installed executable.
fn extract_engine(zip_path: &Path, dest: &Path) -> Result<PathBuf, String> {
    let f = std::fs::File::open(zip_path).map_err(|e| format!("Cannot open engine archive: {e}"))?;
    let mut archive = zip::ZipArchive::new(f).map_err(|e| format!("Bad engine archive: {e}"))?;
    let dest_exe = dest.join(if cfg!(windows) { "sd.exe" } else { "sd" });
    let mut found_exe = false;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
        if !entry.is_file() { continue; }
        let raw = entry.name().to_string();
        let base = raw.rsplit(['/', '\\']).next().unwrap_or(&raw).to_string();
        let low = base.to_ascii_lowercase();
        let is_exe = matches!(low.as_str(), "sd-cli" | "sd" | "sd-cli.exe" | "sd.exe");
        let is_lib = low.ends_with(".dylib") || low.ends_with(".dll") || low.ends_with(".so") || low.contains(".so.");
        if !is_exe && !is_lib { continue; }
        let out = if is_exe { dest_exe.clone() } else { dest.join(&base) };
        let mut w = std::fs::File::create(&out).map_err(|e| format!("Cannot write {base}: {e}"))?;
        std::io::copy(&mut entry, &mut w).map_err(|e| format!("Extract error: {e}"))?;
        if is_exe { found_exe = true; }
    }
    if !found_exe { return Err("Engine archive did not contain the sd executable.".to_string()); }
    Ok(dest_exe)
}

const SETUP_HELP: &str = "Image generation isn't set up yet. Open Settings → the Images tab and \
    click \"Install image engine\" (a small one-time download), then pick and download a model. \
    Everything runs offline on your machine after that.";

const MODEL_HELP: &str = "No image model found. Put a diffusion model (.gguf or .safetensors — e.g. \
    an SD/SDXL-Turbo model) in the app's image-gen/models folder, or set its path in Settings → \
    Image Generation.";

/// How long a single generation may run before it's killed (a stuck/huge job shouldn't hang a turn).
const GENERATE_TIMEOUT_SECS: u64 = 300;

/// Generate one image. Returns PNG bytes on success, or a human-readable error the model relays.
/// `size` overrides the config default; 0 falls back to config → 512.
pub async fn generate(
    cfg: &ImageGenConfig,
    prompt: &str,
    negative: Option<&str>,
    size: u32,
    steps: u32,
    seed: Option<i64>,
) -> Result<Vec<u8>, String> {
    let bin = resolve_binary(cfg).ok_or_else(|| SETUP_HELP.to_string())?;
    let model = resolve_model(cfg).ok_or_else(|| MODEL_HELP.to_string())?;

    let dim = if size > 0 { size } else if cfg.size > 0 { cfg.size } else { 512 };
    let steps = if steps > 0 { steps } else if cfg.steps > 0 { cfg.steps } else { 20 };
    let cfg_scale = if cfg.cfg_scale > 0.0 { cfg.cfg_scale } else { 7.0 };

    let out_dir = std::env::temp_dir().join("lexichat-imagegen");
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("Cannot create output dir: {e}"))?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let out = out_dir.join(format!("img-{nanos}.png"));

    let mut cmd = tokio::process::Command::new(&bin);
    cmd.arg("-m").arg(&model)
        .arg("-p").arg(prompt)
        .arg("-o").arg(&out)
        .arg("-W").arg(dim.to_string())
        .arg("-H").arg(dim.to_string())
        .arg("--steps").arg(steps.to_string())
        .arg("--cfg-scale").arg(format!("{cfg_scale}"));
    if let Some(n) = negative.map(str::trim).filter(|n| !n.is_empty()) {
        cmd.arg("-n").arg(n);
    }
    if let Some(s) = seed {
        cmd.arg("-s").arg(s.to_string());
    }
    for a in cfg.extra_args.split_whitespace() {
        cmd.arg(a);
    }

    // Make the dynamic loader find sd's runtime library, which sits in the same dir as the binary
    // (both the data-dir install and the provisioned-from-resources engine put them together).
    if let Some(dir) = bin.parent() {
        #[cfg(target_os = "macos")]
        cmd.env("DYLD_FALLBACK_LIBRARY_PATH", dir);
        #[cfg(target_os = "linux")]
        cmd.env("LD_LIBRARY_PATH", dir);
        #[cfg(target_os = "windows")]
        {
            // Windows resolves DLLs from PATH — prepend the binary's dir.
            let mut all = vec![dir.to_path_buf()];
            if let Some(p) = std::env::var_os("PATH") { all.extend(std::env::split_paths(&p)); }
            if let Ok(joined) = std::env::join_paths(all) { cmd.env("PATH", joined); }
        }
    }

    cmd.kill_on_drop(true);

    let run = cmd.output();
    let output = match tokio::time::timeout(std::time::Duration::from_secs(GENERATE_TIMEOUT_SECS), run).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => return Err(format!("Failed to launch image generator ({}): {e}", bin.display())),
        Err(_) => return Err(format!("Image generation timed out after {GENERATE_TIMEOUT_SECS}s — try fewer steps or a smaller size.")),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let last = stderr.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("unknown error");
        return Err(format!("Image generation failed: {last}"));
    }

    let bytes = std::fs::read(&out)
        .map_err(|e| format!("Image generator ran but produced no output file: {e}"))?;
    let _ = std::fs::remove_file(&out);
    if bytes.is_empty() {
        return Err("Image generator produced an empty file.".to_string());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_binary_gives_setup_help() {
        // An empty config with no sd binary anywhere resolves to None → SETUP_HELP.
        let cfg = ImageGenConfig { binary_path: "/definitely/not/here/sd".into(), ..Default::default() };
        assert!(resolve_binary(&cfg).is_none());
    }

    #[test]
    fn explicit_missing_model_is_none() {
        let cfg = ImageGenConfig { model_path: "/nope/model.gguf".into(), ..Default::default() };
        assert!(resolve_model(&cfg).is_none());
    }

    #[tokio::test]
    async fn generate_reports_setup_when_unconfigured() {
        let cfg = ImageGenConfig { binary_path: "/no/sd".into(), model_path: "/no/m.gguf".into(), ..Default::default() };
        let err = generate(&cfg, "a cat", None, 512, 4, None).await.unwrap_err();
        assert!(err.contains("Image generation isn't set up") || err.contains("No image model"));
    }

    #[test]
    fn engine_asset_priority_matches_this_platform_or_none() {
        // Sanity: on a supported platform at least one representative asset name matches; on an
        // unsupported one, nothing does. (Names mirror the stable-diffusion.cpp release assets.)
        let samples = [
            "sd-master-bin-Darwin-macOS-arm64.zip",
            "sd-master-bin-win-vulkan-x64.zip",
            "sd-master-bin-Linux-Ubuntu-x86_64-vulkan.zip",
        ];
        let any = samples.iter().any(|s| engine_asset_priority(s).is_some());
        assert_eq!(any, engine_supported());
    }

    // Real end-to-end: fetch + extract the engine from GitHub and confirm the binary runs. Hits the
    // network and installs into the real app-data dir, so it's #[ignore]d — run explicitly with
    // `cargo test download_engine_real -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn download_engine_real() {
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let sd = download_engine(&cancel, |_, _| {}).await.expect("engine download+extract");
        assert!(sd.is_file(), "sd should be installed");
        let out = std::process::Command::new(&sd).arg("--help").output().expect("sd should run");
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(text.contains("model") || text.contains("prompt"), "sd --help should print usage; got: {}", &text[..text.len().min(200)]);
    }
}
