use std::fs;
use std::path::{Path, PathBuf};
use serde_json::Value;
use serde::Serialize;

// ── Wiki directory ────────────────────────────────────────────────────────────

/// Per-thread redirect used only by tests. Rust runs each `#[test]` on its own thread, so a
/// thread-local gives every test its own wiki without them racing — which an env var or a
/// global would not. Without it the tests write into, and delete from, the user's real wiki.
#[cfg(test)]
thread_local! {
    static TEST_WIKI_DIR: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_test_wiki_dir(dir: Option<PathBuf>) {
    TEST_WIKI_DIR.with(|d| *d.borrow_mut() = dir);
}

pub fn wiki_dir() -> PathBuf {
    #[cfg(test)]
    if let Some(dir) = TEST_WIKI_DIR.with(|d| d.borrow().clone()) {
        let _ = fs::create_dir_all(&dir);
        return dir;
    }
    let base = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lexichat")
        .join("wiki");
    let _ = fs::create_dir_all(&base);
    base
}

/// Resolve a model-supplied path to an absolute path inside the wiki dir.
/// Adds .md extension if the path has none. Rejects traversal attempts.
fn resolve(raw: &str) -> Result<PathBuf, String> {
    let raw = raw.trim().trim_start_matches('/');
    if raw.is_empty() {
        return Err("Path must not be empty.".into());
    }
    if raw.contains("..") {
        return Err("Path must not contain '..'.".into());
    }

    let base = wiki_dir();
    let mut p = base.join(raw);

    // Add .md if no extension present
    if p.extension().is_none() {
        p.set_extension("md");
    }

    // Verify the resolved path stays inside wiki_dir
    let canonical_base = fs::canonicalize(&base).unwrap_or(base.clone());
    // For new files, walk up to the first existing ancestor to canonicalize
    let canonical_p = if p.exists() {
        fs::canonicalize(&p).unwrap_or(p.clone())
    } else {
        let mut cur: &Path = &p;
        loop {
            if cur.exists() {
                let mut resolved = fs::canonicalize(cur).unwrap_or(cur.to_path_buf());
                // Append the remaining non-existent suffix
                if let Ok(suffix) = p.strip_prefix(cur) {
                    resolved = resolved.join(suffix);
                }
                break resolved;
            }
            match cur.parent() {
                Some(parent) => cur = parent,
                None => break p.clone(),
            }
        }
    };

    if !canonical_p.starts_with(&canonical_base) {
        return Err(format!("Path '{}' is outside the wiki directory.", raw));
    }

    Ok(p)
}

// ── Tool handlers ─────────────────────────────────────────────────────────────

pub fn wiki_list() -> String {
    let dir = wiki_dir();
    let mut pages: Vec<String> = Vec::new();
    collect_pages(&dir, &dir, &mut pages);
    if pages.is_empty() {
        return "Wiki is empty. Use wiki_write to create your first page.".into();
    }
    pages.sort();
    format!("Wiki pages ({}):\n{}", pages.len(), pages.join("\n"))
}

fn collect_pages(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_pages(root, &path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            if let Ok(rel) = path.strip_prefix(root) {
                out.push(rel.to_string_lossy().to_string());
            }
        }
    }
}

/// Exact-substring search. High precision and zero setup, but it cannot find a page about
/// "sibling DOB" from a query about "my sister's birthday" — that is what the semantic pass
/// in `wiki_search` is for.
pub fn wiki_search(args: &Value) -> String {
    let query = match args["query"].as_str() {
        Some(q) if !q.trim().is_empty() => q.to_lowercase(),
        _ => return "Error: query is required.".into(),
    };

    let dir = wiki_dir();
    let mut results: Vec<String> = Vec::new();
    search_pages(&dir, &dir, &query, &mut results);

    if results.is_empty() {
        return format!("No wiki pages match '{query}'.");
    }
    format!("Search results for '{query}':\n\n{}", results.join("\n---\n"))
}

/// How many semantically related pages to add beyond the exact matches. Enough to rescue a
/// missed recall, few enough that the model isn't handed half the wiki.
const SEMANTIC_LIMIT: usize = 5;

/// Hybrid search: exact matches first, then pages that are merely *about* the same thing.
///
/// The two passes answer different questions and neither subsumes the other. Substring
/// matching is precise and finds nothing when the wording differs; embeddings find the right
/// page from different words but will happily rank a loosely-related one. Running both, with
/// the exact hits first, keeps the precision and adds the recall.
///
/// Falls back to lexical-only whenever the semantic pass cannot run — no embedding model
/// installed, Ollama unreachable, a corrupt index. Memory search must never fail outright.
pub async fn wiki_search_hybrid(args: &Value, backend: &crate::ollama::Backend) -> String {
    let lexical = wiki_search(args);
    let Some(query) = args["query"].as_str().map(str::trim).filter(|q| !q.is_empty()) else {
        return lexical;
    };

    let hits = match crate::wiki_index::semantic_search(backend, query, SEMANTIC_LIMIT).await {
        Ok(Some(hits)) => hits,
        // Ok(None) = no embedding model installed; Err = it exists but failed. Either way the
        // lexical result is still a valid answer, so return it rather than surfacing an error.
        _ => return lexical,
    };
    if hits.is_empty() {
        return lexical;
    }

    // Don't repeat a page the exact pass already returned.
    let already = |path: &str| lexical.contains(&format!("**{path}**"));
    let extra: Vec<String> = hits
        .iter()
        .filter(|h| !already(&h.path))
        .map(|h| {
            let label = if h.heading.is_empty() { h.path.clone() } else { format!("{} — {}", h.path, h.heading) };
            format!("**{}**\n  > {}", label, h.snippet)
        })
        .collect();
    if extra.is_empty() {
        return lexical;
    }

    let header = if lexical.starts_with("No wiki pages match") {
        format!("No page contains '{query}' word-for-word, but these are about it:")
    } else {
        "Also related (matched by meaning, not wording):".to_string()
    };
    format!("{lexical}\n\n{header}\n\n{}", extra.join("\n---\n"))
}

fn search_pages(root: &Path, dir: &Path, query: &str, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            search_pages(root, &path, query, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            let Ok(content) = fs::read_to_string(&path) else { continue };
            if content.to_lowercase().contains(query) {
                if let Ok(rel) = path.strip_prefix(root) {
                    // Return up to 3 matching lines with context
                    let matches: Vec<String> = content.lines()
                        .filter(|l| l.to_lowercase().contains(query))
                        .take(3)
                        .map(|l| format!("  > {l}"))
                        .collect();
                    out.push(format!("**{}**\n{}", rel.display(), matches.join("\n")));
                }
            }
        }
    }
}

pub fn wiki_read(args: &Value) -> String {
    let path_str = match args["path"].as_str() {
        Some(p) => p,
        None => return "Error: path is required.".into(),
    };
    let path = match resolve(path_str) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    match fs::read_to_string(&path) {
        Ok(content) => {
            if content.is_empty() {
                format!("Page '{}' exists but is empty.", path_str)
            } else {
                content
            }
        }
        Err(_) => {
            // Give a more helpful hint for the index file specifically
            if path_str.trim_end_matches(".md") == "index" {
                "index.md not found — wiki is empty or index hasn't been created yet. Call wiki_list to see what's stored.".into()
            } else {
                format!("Page '{}' not found. Use wiki_list or wiki_search to find available pages.", path_str)
            }
        }
    }
}

pub fn wiki_write(args: &Value) -> String {
    let path_str = match args["path"].as_str() {
        Some(p) => p,
        None => return "Error: path is required.".into(),
    };
    let content = match args["content"].as_str() {
        Some(c) => c,
        None => return "Error: content is required.".into(),
    };
    let path = match resolve(path_str) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return format!("Error creating directory: {e}");
        }
    }
    match fs::write(&path, content) {
        Ok(()) => {
            append_log_entry("write", path_str);
            format!("Written {} chars to '{}'.", content.len(), path_str)
        }
        Err(e) => format!("Error writing '{}': {e}", path_str),
    }
}

pub fn wiki_patch(args: &Value) -> String {
    let path_str = match args["path"].as_str() {
        Some(p) => p,
        None => return "Error: path is required.".into(),
    };
    let find = match args["find"].as_str() {
        Some(f) if !f.is_empty() => f,
        _ => return "Error: find is required and must not be empty.".into(),
    };
    let replace = args["replace"].as_str().unwrap_or("");

    let path = match resolve(path_str) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return format!("Page '{}' not found.", path_str),
    };
    if !content.contains(find) {
        return format!("Text not found in '{}'. No changes made.", path_str);
    }
    let patched = content.replacen(find, replace, 1);
    match fs::write(&path, &patched) {
        Ok(()) => {
            append_log_entry("patch", path_str);
            format!("Patched '{}': replaced first occurrence.", path_str)
        }
        Err(e) => format!("Error writing '{}': {e}", path_str),
    }
}

pub fn wiki_append(args: &Value) -> String {
    let path_str = match args["path"].as_str() {
        Some(p) => p,
        None => return "Error: path is required.".into(),
    };
    let content = match args["content"].as_str() {
        Some(c) => c,
        None => return "Error: content is required.".into(),
    };
    let path = match resolve(path_str) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let existing = fs::read_to_string(&path).unwrap_or_default();
    let sep = if existing.is_empty() || existing.ends_with('\n') { "" } else { "\n" };
    let new_content = format!("{existing}{sep}{content}\n");
    match fs::write(&path, &new_content) {
        Ok(()) => format!("Appended {} chars to '{}'.", content.len(), path_str),
        Err(e) => format!("Error appending to '{}': {e}", path_str),
    }
}

pub fn wiki_lint() -> String {
    let dir = wiki_dir();
    let mut all_pages: Vec<String> = Vec::new();
    collect_pages(&dir, &dir, &mut all_pages);

    if all_pages.is_empty() {
        return "Wiki is empty — nothing to lint.".into();
    }

    let content_pages: Vec<&str> = all_pages.iter()
        .map(String::as_str)
        .filter(|p| *p != "index.md" && *p != "log.md")
        .collect();

    let mut issues: Vec<String> = Vec::new();
    let mut stats_words = 0usize;

    // Count words and find empty pages
    for page in &all_pages {
        let path = dir.join(page);
        let content = fs::read_to_string(&path).unwrap_or_default();
        let words: usize = content.split_whitespace().count();
        stats_words += words;
        if words == 0 {
            issues.push(format!("  • Empty page: {page}"));
        }
    }

    // Check index.md coverage
    let index_path = dir.join("index.md");
    if !index_path.exists() {
        if !content_pages.is_empty() {
            issues.push(format!("  • index.md missing — {} pages have no index entry.", content_pages.len()));
        }
    } else {
        let index_content = fs::read_to_string(&index_path).unwrap_or_default();

        // Pages not mentioned in index
        let mut unindexed: Vec<&str> = Vec::new();
        for page in &content_pages {
            // Check if page basename or full path appears anywhere in index
            let stem = std::path::Path::new(page)
                .file_stem().and_then(|s| s.to_str()).unwrap_or(page);
            if !index_content.contains(page) && !index_content.contains(stem) {
                unindexed.push(page);
            }
        }
        if !unindexed.is_empty() {
            issues.push(format!("  • Pages not in index.md ({}):\n{}",
                unindexed.len(),
                unindexed.iter().map(|p| format!("      - {p}")).collect::<Vec<_>>().join("\n")));
        }

        // Markdown links in index that point to missing files
        let mut broken: Vec<String> = Vec::new();
        for cap in index_content.split("](") {
            if let Some(end) = cap.find(')') {
                let link = cap[..end].trim();
                if link.ends_with(".md") && !link.starts_with("http") {
                    let linked = dir.join(link);
                    if !linked.exists() {
                        broken.push(link.to_string());
                    }
                }
            }
        }
        if !broken.is_empty() {
            issues.push(format!("  • Broken links in index.md ({}):\n{}",
                broken.len(),
                broken.iter().map(|l| format!("      - {l}")).collect::<Vec<_>>().join("\n")));
        }
    }

    // Log freshness
    let log_path = dir.join("log.md");
    let log_summary = if log_path.exists() {
        let log = fs::read_to_string(&log_path).unwrap_or_default();
        let entries: Vec<&str> = log.lines().filter(|l| l.starts_with("## [")).collect();
        format!("log.md: {} entries, last: {}", entries.len(),
            entries.last().copied().unwrap_or("(none)"))
    } else {
        "log.md: not created yet".into()
    };

    let summary = format!(
        "Wiki health check\n\
         Pages: {} ({} content + index + log)\n\
         Words: ~{stats_words}\n\
         {log_summary}",
        all_pages.len(), content_pages.len()
    );

    if issues.is_empty() {
        format!("{summary}\n\nNo issues found.")
    } else {
        format!("{summary}\n\nIssues ({}):\n{}", issues.len(), issues.join("\n"))
    }
}

// ── Auto-logging helper ───────────────────────────────────────────────────────

fn append_log_entry(action: &str, detail: &str) {
    let log_path = wiki_dir().join("log.md");
    let now = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let entry = format!("## [{now}] {action} | {detail}\n");
    let existing = fs::read_to_string(&log_path).unwrap_or_default();
    let _ = fs::write(&log_path, format!("{existing}{entry}"));
}

pub fn wiki_delete(args: &Value) -> String {
    let path_str = match args["path"].as_str() {
        Some(p) => p,
        None => return "Error: path is required.".into(),
    };
    let path = match resolve(path_str) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    if !path.exists() {
        return format!("Page '{}' not found.", path_str);
    }
    match fs::remove_file(&path) {
        Ok(()) => {
            append_log_entry("delete", path_str);
            format!("Deleted '{}'.", path_str)
        }
        Err(e) => format!("Error deleting '{}': {e}", path_str),
    }
}

// ── Schema helpers (available for future Rust-side use) ──────────────────────

#[allow(dead_code)]
pub fn wiki_schemas() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_list",
                "description": "List all pages in the persistent wiki. Use this to discover what knowledge has been stored.",
                "parameters": { "type": "object", "properties": {}, "required": [] }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_search",
                "description": "Search the wiki for pages containing a keyword or phrase. Returns matching page names and relevant lines. Always search before writing to avoid duplicates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Keyword or phrase to search for." }
                    },
                    "required": ["query"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_read",
                "description": "Read the full contents of a wiki page. Use wiki_list or wiki_search first to find the correct path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Page path relative to the wiki root, e.g. 'people/alice.md' or 'projects.md'. The .md extension is optional." }
                    },
                    "required": ["path"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_write",
                "description": "Create or overwrite a wiki page with markdown content. Use wiki_search first to avoid duplicates. Use clear structured markdown with headings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Page path, e.g. 'people/alice.md' or 'projects.md'." },
                        "content": { "type": "string", "description": "Full markdown content for the page." }
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_patch",
                "description": "Update part of an existing wiki page by replacing the first occurrence of a specific string. Use this for small targeted updates rather than rewriting the entire page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Page path." },
                        "find": { "type": "string", "description": "Exact text to find (must be an exact substring of the page content)." },
                        "replace": { "type": "string", "description": "Text to replace it with." }
                    },
                    "required": ["path", "find", "replace"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_delete",
                "description": "Permanently delete a wiki page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Page path to delete." }
                    },
                    "required": ["path"]
                }
            }
        }),
    ]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// ── Graph ─────────────────────────────────────────────────────────────────────

/// A page as a node: enough to draw and label it without reading the file again.
#[derive(Debug, Clone, Serialize)]
pub struct WikiNode {
    pub path: String,
    /// First path component ("people", "projects"), or "" for a page at the root. Drives colour.
    pub folder: String,
    /// First heading in the page, falling back to the filename.
    pub title: String,
    pub bytes: usize,
    /// Indexed chunks — a rough measure of how much is written, and 0 when unindexed.
    pub chunks: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct WikiLink {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WikiGraph {
    pub nodes: Vec<WikiNode>,
    /// Links actually written in the Markdown.
    pub links: Vec<WikiLink>,
    /// Pages that are merely about similar things. Empty when nothing has been indexed.
    pub related: Vec<crate::wiki_index::SemanticEdge>,
    /// True when semantic edges are unavailable, so the UI can explain the sparser graph
    /// instead of implying the wiki has no structure.
    pub unindexed: bool,
}

/// Wiki links written in a page, as relative page paths.
///
/// Handles both spellings the wiki uses: Markdown `[label](people/alice.md)` and the
/// `[[people/alice]]` wiki style. Anything with a scheme is an external URL, not a page.
pub fn parse_links(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let bytes: Vec<char> = text.chars().collect();
    let mut i = 0usize;

    let normalise = |raw: &str| -> Option<String> {
        let t = raw.trim().trim_start_matches("./");
        if t.is_empty() || t.contains("://") || t.starts_with('#') {
            return None;
        }
        // Drop any anchor, then ensure the .md the wiki resolver would add.
        let t = t.split('#').next().unwrap_or(t).trim();
        if t.is_empty() { return None; }
        Some(if t.ends_with(".md") { t.to_string() } else { format!("{t}.md") })
    };

    while i < bytes.len() {
        // [[wiki style]]
        if bytes[i] == '[' && bytes.get(i + 1) == Some(&'[') {
            if let Some(end) = (i + 2..bytes.len()).find(|&j| bytes[j] == ']' && bytes.get(j + 1) == Some(&']')) {
                let inner: String = bytes[i + 2..end].iter().collect();
                // "[[path|label]]" — the target is the part before the pipe.
                let target = inner.split('|').next().unwrap_or(&inner).to_string();
                if let Some(p) = normalise(&target) { out.push(p); }
                i = end + 2;
                continue;
            }
        }
        // [label](target)
        if bytes[i] == '[' {
            if let Some(close) = (i + 1..bytes.len()).find(|&j| bytes[j] == ']') {
                if bytes.get(close + 1) == Some(&'(') {
                    if let Some(end) = (close + 2..bytes.len()).find(|&j| bytes[j] == ')') {
                        let target: String = bytes[close + 2..end].iter().collect();
                        if let Some(p) = normalise(&target) { out.push(p); }
                        i = end + 1;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    out.sort();
    out.dedup();
    out
}

/// First `# heading` in a page, or the filename stem if it has none.
fn page_title(text: &str, path: &str) -> String {
    for line in text.lines().take(40) {
        let level = line.chars().take_while(|c| *c == '#').count();
        if (1..=6).contains(&level) && line.chars().nth(level) == Some(' ') {
            let t = line[level + 1..].trim();
            if !t.is_empty() { return t.to_string(); }
        }
    }
    Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or(path).replace('_', " ")
}

/// Build the whole graph: one node per page, edges for real links, and edges for pages that
/// are merely about similar things.
///
/// Link edges are kept only when both ends exist, so a broken link shows up in `wiki_lint`
/// rather than as a phantom node here.
pub fn wiki_graph(min_score: f32) -> WikiGraph {
    let root = wiki_dir();
    let mut pages: Vec<(String, String)> = Vec::new();
    read_all_pages(&root, &root, &mut pages);

    let chunk_counts = crate::wiki_index::chunk_counts();
    let known: std::collections::HashSet<String> = pages.iter().map(|(p, _)| p.clone()).collect();

    let mut nodes = Vec::new();
    let mut links = Vec::new();
    for (rel, text) in &pages {
        let folder = Path::new(rel).parent()
            .map(|p| p.to_string_lossy().to_string())
            .filter(|p| !p.is_empty())
            .unwrap_or_default();
        nodes.push(WikiNode {
            path: rel.clone(),
            folder,
            title: page_title(text, rel),
            bytes: text.len(),
            chunks: chunk_counts.get(rel).copied().unwrap_or(0),
        });
        for target in parse_links(text) {
            if target != *rel && known.contains(&target) {
                links.push(WikiLink { from: rel.clone(), to: target });
            }
        }
    }
    nodes.sort_by(|a, b| a.path.cmp(&b.path));

    let related = crate::wiki_index::semantic_edges(min_score);
    WikiGraph { nodes, links, unindexed: chunk_counts.is_empty(), related }
}

/// Every `.md` page under the wiki, as (relative path, contents).
fn read_all_pages(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            read_all_pages(root, &path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            if let (Ok(rel), Ok(text)) = (path.strip_prefix(root), fs::read_to_string(&path)) {
                out.push((rel.to_string_lossy().to_string(), text));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Run `f` against a throwaway wiki.
    ///
    /// This used to create a TempDir and then never use it, so every wiki test read from and
    /// wrote to the user's real wiki — leaving `_test_*` pages behind, appending to their
    /// log.md, and in one case deleting their index.md and restoring it from memory. The
    /// redirect is now actually installed, and cleared afterwards even if the test panics.
    fn with_wiki_dir(f: impl FnOnce(&TempDir)) {
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) { set_test_wiki_dir(None); }
        }
        let tmp = TempDir::new().unwrap();
        set_test_wiki_dir(Some(tmp.path().to_path_buf()));
        let _guard = Guard;
        f(&tmp);
    }

    /// The isolation itself, pinned. If this fails, every other test in this module is
    /// quietly operating on the user's real wiki again.
    #[test]
    fn tests_never_touch_the_real_wiki() {
        let real = {
            set_test_wiki_dir(None);
            wiki_dir()
        };
        with_wiki_dir(|tmp| {
            assert_eq!(wiki_dir(), tmp.path(), "wiki_dir must resolve to the temp wiki");
            assert_ne!(wiki_dir(), real, "wiki_dir must not be the user's wiki");
            let _ = wiki_write(&serde_json::json!({ "path": "canary", "content": "x" }));
            assert!(tmp.path().join("canary.md").exists());
            assert!(!real.join("canary.md").exists(), "wrote outside the temp wiki");
        });
        // …and the redirect is torn down afterwards.
        assert_eq!(wiki_dir(), real);
    }

    #[test]
    fn resolve_adds_md_extension() {
        with_wiki_dir(|_| {
            let p = wiki_dir().join("test");
            let mut with_ext = p.clone();
            with_ext.set_extension("md");
            assert_eq!(with_ext.extension().unwrap(), "md");
        });
    }

    #[test]
    fn wiki_write_and_read_roundtrip() {
        with_wiki_dir(|_| {
            let result = wiki_write(&serde_json::json!({ "path": "roundtrip", "content": "# Hello\nworld" }));
            assert!(result.contains("Written"), "write failed: {result}");
            let content = wiki_read(&serde_json::json!({ "path": "roundtrip" }));
            assert!(content.contains("Hello"), "read failed: {content}");
        });
    }

    #[test]
    fn wiki_patch_updates_content() {
        with_wiki_dir(|_| {
            let path = "patch";
            let _ = wiki_write(&serde_json::json!({ "path": path, "content": "old text here" }));
            let result = wiki_patch(&serde_json::json!({ "path": path, "find": "old text", "replace": "new text" }));
            assert!(result.contains("Patched"), "patch failed: {result}");
            let content = wiki_read(&serde_json::json!({ "path": path }));
            assert!(content.contains("new text"), "content not updated: {content}");
            assert!(!content.contains("old text"), "old text still present");
        });
    }

    #[test]
    fn wiki_patch_reports_not_found() {
        with_wiki_dir(|_| {
            let result = wiki_patch(&serde_json::json!({ "path": "nonexistent", "find": "x", "replace": "y" }));
            assert!(result.contains("not found"), "expected not-found: {result}");
        });
    }

    #[test]
    fn wiki_read_missing_page() {
        with_wiki_dir(|_| {
            let result = wiki_read(&serde_json::json!({ "path": "surely_missing" }));
            assert!(result.contains("not found"), "expected not-found: {result}");
        });
    }

    #[test]
    fn wiki_search_finds_content() {
        with_wiki_dir(|_| {
            let _ = wiki_write(&serde_json::json!({ "path": "search", "content": "# Alice\nBirthday: 14th March" }));
            let result = wiki_search(&serde_json::json!({ "query": "birthday" }));
            assert!(result.to_lowercase().contains("birthday"), "search failed: {result}");
        });
    }

    #[test]
    fn wiki_delete_removes_page() {
        with_wiki_dir(|_| {
            let path = "to_delete";
            let _ = wiki_write(&serde_json::json!({ "path": path, "content": "to delete" }));
            let result = wiki_delete(&serde_json::json!({ "path": path }));
            assert!(result.contains("Deleted"), "delete failed: {result}");
            let read = wiki_read(&serde_json::json!({ "path": path }));
            assert!(read.contains("not found"), "page still exists after delete");
        });
    }

    #[test]
    fn resolve_rejects_traversal() {
        let result = resolve("../etc/passwd");
        assert!(result.is_err(), "expected traversal rejection");
    }

    #[test]
    fn wiki_append_creates_and_grows() {
        with_wiki_dir(|_| {
            let path = "append";
            let r1 = wiki_append(&serde_json::json!({ "path": path, "content": "line one" }));
            assert!(r1.contains("Appended"), "first append failed: {r1}");
            let r2 = wiki_append(&serde_json::json!({ "path": path, "content": "line two" }));
            assert!(r2.contains("Appended"), "second append failed: {r2}");
            let content = wiki_read(&serde_json::json!({ "path": path }));
            assert!(content.contains("line one"), "first line missing: {content}");
            assert!(content.contains("line two"), "second line missing: {content}");
        });
    }

    #[test]
    fn wiki_lint_reports_empty_wiki() {
        with_wiki_dir(|_| {
            let result = wiki_lint();
            assert!(!result.is_empty(), "lint returned empty string");
        });
    }

    #[test]
    fn wiki_lint_detects_page_missing_from_index() {
        // Previously this deleted the user's real index.md and restored it from a backup —
        // a panic in between would have lost it. On a temp wiki there is simply no index.
        with_wiki_dir(|_| {
            let _ = wiki_write(&serde_json::json!({ "path": "orphan", "content": "# Orphan page" }));
            let result = wiki_lint();
            assert!(result.contains("index.md"), "lint should flag the missing index: {result}");
        });
    }

    // ── Graph ─────────────────────────────────────────────────────────────────

    #[test]
    fn links_are_parsed_in_both_spellings() {
        let md = "See [Alice](people/alice.md) and [[projects/bionic]] and [Ext](https://x.com).";
        assert_eq!(parse_links(md), vec!["people/alice.md", "projects/bionic.md"]);
    }

    #[test]
    fn link_targets_are_normalised() {
        // A missing .md, a leading ./ and a trailing anchor all point at the same page.
        assert_eq!(parse_links("[a](./people/alice) [b](people/alice.md#birthday)"),
                   vec!["people/alice.md"]);
    }

    #[test]
    fn the_graph_only_links_pages_that_exist() {
        with_wiki_dir(|_| {
            let _ = wiki_write(&serde_json::json!({ "path": "index", "content":
                "# Index\n- [Real](real.md)\n- [Gone](missing.md)" }));
            let _ = wiki_write(&serde_json::json!({ "path": "real", "content": "# Real page" }));
            let g = wiki_graph(0.6);
            let paths: Vec<&str> = g.nodes.iter().map(|n| n.path.as_str()).collect();
            assert!(paths.contains(&"index.md") && paths.contains(&"real.md"));
            // Writing also touches log.md, so assert on what is absent rather than a count.
            assert!(!paths.contains(&"missing.md"), "a broken link must not invent a node");
            // The dangling link is wiki_lint's business, not a phantom edge here.
            assert_eq!(g.links.len(), 1);
            assert_eq!(g.links[0].to, "real.md");
        });
    }

    #[test]
    fn a_node_takes_its_title_from_the_first_heading() {
        with_wiki_dir(|_| {
            let _ = wiki_write(&serde_json::json!({ "path": "some_page", "content": "intro\n## Real Title\nbody" }));
            let _ = wiki_write(&serde_json::json!({ "path": "no_heading", "content": "just body text" }));
            let g = wiki_graph(0.6);
            let title = |p: &str| g.nodes.iter().find(|n| n.path == p).unwrap().title.clone();
            assert_eq!(title("some_page.md"), "Real Title");
            // No heading: fall back to a readable filename rather than showing nothing.
            assert_eq!(title("no_heading.md"), "no heading");
        });
    }

    #[test]
    fn nodes_carry_their_folder_for_colouring() {
        with_wiki_dir(|_| {
            let _ = wiki_write(&serde_json::json!({ "path": "people/alice", "content": "# Alice" }));
            let _ = wiki_write(&serde_json::json!({ "path": "root_page", "content": "# Root" }));
            let g = wiki_graph(0.6);
            let folder = |p: &str| g.nodes.iter().find(|n| n.path == p).unwrap().folder.clone();
            assert_eq!(folder("people/alice.md"), "people");
            assert_eq!(folder("root_page.md"), "", "a root page has no folder, not a fake one");
        });
    }
}
