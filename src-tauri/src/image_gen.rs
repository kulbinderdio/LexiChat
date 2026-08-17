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

/// Install the bundled sd engine (binary + runtime lib, shipped as app resources under
/// `<resource_dir>/image-gen/`) into the writable app-data image-gen dir on first launch. Resources
/// are read-only and lose the executable bit, so we can't run them in place — we copy once, set
/// +x (and de-quarantine on macOS), and the normal data-dir resolution takes over. No-op if already
/// installed, if the user provided their own, or if this platform ships no bundled engine.
pub fn provision_from_resources(resource_dir: &Path) {
    if let Some(dest) = image_gen_dir() {
        provision_into(resource_dir, &dest);
    }
}

fn provision_into(resource_dir: &Path, dest: &Path) {
    let dest_exe = dest.join(if cfg!(windows) { "sd.exe" } else { "sd" });
    if dest_exe.is_file() {
        return; // already provisioned or user-installed
    }
    let Some(src) = [resource_dir.join("image-gen"), resource_dir.join("resources").join("image-gen")]
        .into_iter()
        .find(|c| c.is_dir())
    else {
        return; // no bundled engine on this platform/build
    };
    let exe_names = ["sd", "sd.exe", "sd-cli", "sd-cli.exe"];
    if !exe_names.iter().any(|n| src.join(n).is_file()) {
        return; // resource dir present but no executable — nothing to install
    }
    if std::fs::create_dir_all(&dest).is_err() {
        return;
    }
    if let Ok(rd) = std::fs::read_dir(&src) {
        for entry in rd.flatten() {
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let name = p.file_name().unwrap_or_default();
            let target = if exe_names.iter().any(|n| std::ffi::OsStr::new(n) == name) {
                dest_exe.clone() // normalize the executable name to `sd`/`sd.exe`
            } else {
                dest.join(name)
            };
            let _ = std::fs::copy(&p, &target);
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(&dest_exe) {
            let mut perms = meta.permissions();
            perms.set_mode(0o755);
            let _ = std::fs::set_permissions(&dest_exe, perms);
        }
    }
    // macOS Gatekeeper: strip quarantine so the copied binary/lib can run.
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("xattr")
            .arg("-dr").arg("com.apple.quarantine").arg(&dest).status();
    }
}

const SETUP_HELP: &str = "Image generation isn't set up yet. It needs a stable-diffusion.cpp \
    executable (the `sd` binary — the offline, GUI-free image engine) and a diffusion model. \
    Either put the `sd` binary and a `.gguf`/`.safetensors` model under the app's image-gen folder, \
    or set their paths in Settings → Image Generation. (Tip: an SD/SDXL-Turbo model generates in \
    ~4 steps and is fast on Apple Silicon.)";

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
    fn provision_installs_engine_into_dest() {
        let base = std::env::temp_dir().join(format!("lexi-prov-{}-{}", std::process::id(), line!()));
        let res = base.join("res").join("image-gen");
        let dest = base.join("dest");
        std::fs::create_dir_all(&res).unwrap();
        std::fs::write(res.join("sd"), b"#!/bin/sh\necho hi\n").unwrap();
        std::fs::write(res.join("libstable-diffusion.dylib"), b"lib").unwrap();

        provision_into(&base.join("res"), &dest);

        let exe = dest.join(if cfg!(windows) { "sd.exe" } else { "sd" });
        assert!(exe.is_file(), "sd should be provisioned into dest");
        assert!(dest.join("libstable-diffusion.dylib").is_file(), "runtime lib should be copied");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&exe).unwrap().permissions().mode();
            assert!(mode & 0o111 != 0, "provisioned sd should be executable");
        }
        // Idempotent: a second call is a no-op (dest already has sd) and doesn't error.
        provision_into(&base.join("res"), &dest);
        let _ = std::fs::remove_dir_all(&base);
    }
}
