// Semantic recall for the wiki.
//
// `wiki_search` matched lowercase substrings, so asking about "my sister's birthday" when
// the page says "sibling DOB" returned nothing — the memory was there and simply could not
// be found. This module adds meaning-based retrieval on top, without giving up the exact
// matching that substring search is genuinely good at.
//
// Design constraints that shaped it:
//   * Nothing leaves the machine. Embeddings come from the user's own Ollama instance.
//   * Zero configuration. If an embedding model is installed we use it; if not, search
//     silently stays lexical rather than erroring or nagging.
//   * The wiki stays plain Markdown the user owns. The index is a separate file in the app
//     data directory, never inside the wiki folder.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::ollama::Backend;

/// Chunks larger than this are split further. Big enough to keep a section's meaning intact,
/// small enough that one stray paragraph doesn't drown a chunk's vector.
const MAX_CHUNK_CHARS: usize = 1400;

/// Cosine similarity below this is noise — an unrelated page will still score ~0.3 against
/// any query, and returning those makes recall look worse than saying nothing.
const MIN_SCORE: f32 = 0.45;

// ── Index shape ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Chunk {
    /// Heading path the chunk sits under ("Family > Birthdays"), used as the result snippet
    /// label and prefixed to the embedded text so a bare list of dates keeps its context.
    pub heading: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexedPage {
    /// Content hash — re-embed only what actually changed, not everything on every search.
    hash: u64,
    chunks: Vec<Chunk>,
    vectors: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct WikiIndex {
    /// Vectors from different models are not comparable, so switching model invalidates all.
    #[serde(default)]
    model: String,
    #[serde(default)]
    pages: HashMap<String, IndexedPage>,
}

fn index_path() -> PathBuf {
    // Deliberately NOT inside the wiki folder: that folder is the user's own Markdown, and a
    // multi-megabyte JSON blob of floats has no business sitting in it.
    crate::dirs_path().join("wiki-index.json")
}

fn load_index() -> WikiIndex {
    std::fs::read_to_string(index_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_index(index: &WikiIndex) {
    if let Ok(json) = serde_json::to_string(index) {
        let _ = std::fs::write(index_path(), json);
    }
}

// ── Pure helpers ──────────────────────────────────────────────────────────────

/// Stable content hash. Only used to detect "has this page changed since we embedded it",
/// so speed matters and collision resistance does not.
pub fn content_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Split a Markdown page into chunks along its headings.
///
/// Headings carry most of a wiki page's structure, so they are the natural seam: a chunk is
/// one section, labelled with its heading trail. Sections longer than `MAX_CHUNK_CHARS` are
/// split again at blank lines, each part keeping the heading so it stays self-describing.
pub fn chunk_markdown(text: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut trail: Vec<String> = Vec::new();   // heading stack, by level
    let mut heading = String::new();
    let mut body = String::new();

    let flush = |heading: &str, body: &str, out: &mut Vec<Chunk>| {
        let trimmed = body.trim();
        if trimmed.is_empty() {
            return;
        }
        for part in split_long(trimmed, MAX_CHUNK_CHARS) {
            out.push(Chunk { heading: heading.to_string(), text: part });
        }
    };

    for line in text.lines() {
        let level = line.chars().take_while(|c| *c == '#').count();
        let is_heading = level >= 1 && level <= 6 && line.chars().nth(level) == Some(' ');
        if is_heading {
            flush(&heading, &body, &mut chunks);
            body.clear();
            let title = line[level + 1..].trim().to_string();
            trail.truncate(level.saturating_sub(1));
            trail.push(title);
            heading = trail.join(" > ");
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }
    flush(&heading, &body, &mut chunks);
    chunks
}

/// Break an over-long section at paragraph boundaries, falling back to a hard split only if a
/// single paragraph is itself larger than the limit.
fn split_long(text: &str, max: usize) -> Vec<String> {
    if text.chars().count() <= max {
        return vec![text.to_string()];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for para in text.split("\n\n") {
        if !cur.is_empty() && cur.chars().count() + para.chars().count() > max {
            out.push(std::mem::take(&mut cur).trim().to_string());
        }
        if para.chars().count() > max {
            // One enormous paragraph: hard-split on character count.
            let chars: Vec<char> = para.chars().collect();
            for slice in chars.chunks(max) {
                out.push(slice.iter().collect::<String>().trim().to_string());
            }
        } else {
            cur.push_str(para);
            cur.push_str("\n\n");
        }
    }
    let tail = cur.trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out.retain(|s| !s.is_empty());
    out
}

/// Cosine similarity. Returns 0.0 for mismatched or empty vectors rather than panicking —
/// a corrupt index entry should degrade one result, not take down the search.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Choose an embedding model from what Ollama has installed.
///
/// Ranked by retrieval quality rather than alphabetically, so a machine with several
/// installed gets the best one. Returns `None` when nothing suitable is present, which is
/// the signal to stay on lexical search.
pub fn pick_embed_model(models: &[String]) -> Option<String> {
    const PREFERRED: [&str; 5] = [
        "nomic-embed-text",
        "mxbai-embed-large",
        "snowflake-arctic-embed",
        "bge-m3",
        "all-minilm",
    ];
    for want in PREFERRED {
        if let Some(m) = models.iter().find(|m| m.starts_with(want)) {
            return Some(m.clone());
        }
    }
    // Anything self-describing as an embedding model. Chat models are excluded on purpose:
    // they will happily return a vector, and it will retrieve badly.
    models.iter().find(|m| m.contains("embed")).cloned()
}

// ── Ollama embeddings ─────────────────────────────────────────────────────────

/// Embed a batch of texts via Ollama's `/api/embed`.
pub(crate) async fn embed_batch(backend: &Backend, model: &str, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let url = format!("{}/api/embed", backend.base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder().use_rustls_tls().build()?;
    let resp = client
        .post(&url)
        .json(&serde_json::json!({ "model": model, "input": texts }))
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("embed HTTP {}", resp.status().as_u16());
    }
    #[derive(Deserialize)]
    struct EmbedResponse {
        embeddings: Vec<Vec<f32>>,
    }
    let parsed: EmbedResponse = resp.json().await?;
    Ok(parsed.embeddings)
}

// ── Index maintenance ─────────────────────────────────────────────────────────

/// Every `.md` page in the wiki, as (relative path, contents).
fn read_pages(root: &Path) -> Vec<(String, String)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
                if let (Ok(rel), Ok(text)) = (path.strip_prefix(root), std::fs::read_to_string(&path)) {
                    out.push((rel.to_string_lossy().to_string(), text));
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out
}

/// Bring the index in line with what's on disk, embedding only pages that are new or changed.
/// Returns the number of pages re-embedded, which the caller can report on first build.
async fn refresh_index(backend: &Backend, model: &str, index: &mut WikiIndex) -> anyhow::Result<usize> {
    if index.model != model {
        // Vectors from a different model are meaningless here — start clean.
        index.pages.clear();
        index.model = model.to_string();
    }
    let root = crate::wiki::wiki_dir();
    let pages = read_pages(&root);
    let live: std::collections::HashSet<&str> = pages.iter().map(|(p, _)| p.as_str()).collect();
    index.pages.retain(|path, _| live.contains(path.as_str()));

    let mut embedded = 0usize;
    for (rel, text) in &pages {
        let hash = content_hash(text);
        if index.pages.get(rel).is_some_and(|p| p.hash == hash) {
            continue;
        }
        let chunks = chunk_markdown(text);
        if chunks.is_empty() {
            index.pages.remove(rel);
            continue;
        }
        // Prefix the heading trail so a chunk that is only "3 March" still embeds as a
        // birthday rather than a bare date.
        let inputs: Vec<String> = chunks
            .iter()
            .map(|c| if c.heading.is_empty() { c.text.clone() } else { format!("{}\n{}", c.heading, c.text) })
            .collect();
        let vectors = embed_batch(backend, model, &inputs).await?;
        if vectors.len() != chunks.len() {
            anyhow::bail!("embed returned {} vectors for {} chunks", vectors.len(), chunks.len());
        }
        index.pages.insert(rel.clone(), IndexedPage { hash, chunks, vectors });
        embedded += 1;
    }
    Ok(embedded)
}

// ── Search ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub path: String,
    pub heading: String,
    pub snippet: String,
    pub score: f32,
}

/// Rank chunks against a query vector, keeping only the best chunk per page so one long page
/// cannot crowd out every other result.
pub fn rank(query: &[f32], pages: &[(String, &[Chunk], &[Vec<f32>])], limit: usize) -> Vec<Hit> {
    let mut best: Vec<Hit> = Vec::new();
    for (path, chunks, vectors) in pages {
        let mut top: Option<Hit> = None;
        for (i, chunk) in chunks.iter().enumerate() {
            let Some(vec) = vectors.get(i) else { continue };
            let score = cosine(query, vec);
            if score < MIN_SCORE {
                continue;
            }
            if top.as_ref().is_none_or(|t| score > t.score) {
                top = Some(Hit {
                    path: path.clone(),
                    heading: chunk.heading.clone(),
                    snippet: snippet(&chunk.text),
                    score,
                });
            }
        }
        if let Some(hit) = top {
            best.push(hit);
        }
    }
    best.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    best.truncate(limit);
    best
}

/// First few non-empty lines of a chunk — enough for the model to judge relevance without
/// pulling the whole page into context.
fn snippet(text: &str) -> String {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .take(3)
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(240)
        .collect()
}

/// Semantically related wiki pages for `query`.
///
/// Returns `Ok(None)` when no embedding model is installed — the caller should carry on with
/// lexical search. `Err` means an embedding model exists but something went wrong, which is
/// also non-fatal to the caller but worth distinguishing.
pub async fn semantic_search(backend: &Backend, query: &str, limit: usize) -> anyhow::Result<Option<Vec<Hit>>> {
    let models = crate::ollama::list_models(backend).await.unwrap_or_default();
    let Some(model) = pick_embed_model(&models) else {
        return Ok(None);
    };

    let mut index = load_index();
    let embedded = refresh_index(backend, &model, &mut index).await?;
    if embedded > 0 {
        save_index(&index);
    }

    let query_vec = embed_batch(backend, &model, &[query.to_string()])
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no query embedding returned"))?;

    let pages: Vec<(String, &[Chunk], &[Vec<f32>])> = index
        .pages
        .iter()
        .map(|(path, p)| (path.clone(), p.chunks.as_slice(), p.vectors.as_slice()))
        .collect();
    Ok(Some(rank(&query_vec, &pages, limit)))
}

// ── Graph edges ───────────────────────────────────────────────────────────────

/// Two pages that are about similar things, with how similar they are.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SemanticEdge {
    pub a: String,
    pub b: String,
    pub score: f32,
}

/// Mean of a page's chunk vectors — one point standing for the whole page.
///
/// Averaging loses the detail that per-chunk search needs, which is exactly why it suits a
/// graph: a page about two subjects lands between them rather than appearing twice.
pub fn centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    let Some(dim) = vectors.first().map(|v| v.len()) else { return Vec::new() };
    if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
        return Vec::new();
    }
    let n = vectors.len() as f32;
    (0..dim).map(|i| vectors.iter().map(|v| v[i]).sum::<f32>() / n).collect()
}

/// Every pair of pages whose centroids are at least `min_score` alike, strongest first.
///
/// Returns an empty list when the wiki has never been indexed — the graph then falls back to
/// explicit links only, rather than showing nothing.
pub fn semantic_edges(min_score: f32) -> Vec<SemanticEdge> {
    let index = load_index();
    let centroids: Vec<(String, Vec<f32>)> = index
        .pages
        .iter()
        .map(|(path, page)| (path.clone(), centroid(&page.vectors)))
        .filter(|(_, c)| !c.is_empty())
        .collect();

    let mut edges = Vec::new();
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let score = cosine(&centroids[i].1, &centroids[j].1);
            if score >= min_score {
                edges.push(SemanticEdge {
                    a: centroids[i].0.clone(),
                    b: centroids[j].0.clone(),
                    score,
                });
            }
        }
    }
    edges.sort_by(|x, y| y.score.partial_cmp(&x.score).unwrap_or(std::cmp::Ordering::Equal));
    edges
}

/// Chunk count per page, so the graph can size a node by how much is written on it.
pub fn chunk_counts() -> std::collections::HashMap<String, usize> {
    load_index().pages.into_iter().map(|(p, page)| (p, page.chunks.len())).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_split_on_headings_and_keep_their_trail() {
        let md = "# Alice\nLikes tea.\n\n## Family\nSister: Bea.\n\n### Birthdays\n3 March\n";
        let chunks = chunk_markdown(md);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].heading, "Alice");
        assert_eq!(chunks[1].heading, "Alice > Family");
        // The deepest chunk carries the whole trail, so "3 March" embeds as a birthday.
        assert_eq!(chunks[2].heading, "Alice > Family > Birthdays");
        assert!(chunks[2].text.contains("3 March"));
    }

    #[test]
    fn a_page_with_no_headings_is_still_one_chunk() {
        let chunks = chunk_markdown("Just some loose notes.\nNo headings at all.");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, "");
        assert!(chunks[0].text.contains("loose notes"));
    }

    #[test]
    fn empty_sections_are_dropped() {
        // A heading with nothing under it embeds to noise, so it must not become a chunk.
        let chunks = chunk_markdown("# Empty\n\n# Real\nContent here.\n");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, "Real");
    }

    #[test]
    fn hashes_and_bare_hashes_are_not_confused_for_headings() {
        // "#tag" and "####" are not headings — only "# " through "###### " are.
        let chunks = chunk_markdown("#nothashtag stays body\n####\nstill body\n");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, "");
    }

    #[test]
    fn long_sections_are_split_but_stay_under_the_cap() {
        let para = "x".repeat(600);
        let md = format!("# Big\n{para}\n\n{para}\n\n{para}\n");
        let chunks = chunk_markdown(&md);
        assert!(chunks.len() > 1, "should have split");
        assert!(chunks.iter().all(|c| c.text.chars().count() <= MAX_CHUNK_CHARS));
        assert!(chunks.iter().all(|c| c.heading == "Big"), "every part keeps its heading");
    }

    #[test]
    fn one_paragraph_larger_than_the_cap_is_hard_split() {
        let md = format!("# Huge\n{}\n", "y".repeat(MAX_CHUNK_CHARS * 2 + 50));
        let chunks = chunk_markdown(&md);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| c.text.chars().count() <= MAX_CHUNK_CHARS));
    }

    #[test]
    fn cosine_is_one_for_identical_and_zero_for_orthogonal() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_degrades_rather_than_panics_on_bad_input() {
        // A truncated or corrupt index entry should cost one result, not crash the search.
        assert_eq!(cosine(&[1.0, 2.0], &[1.0]), 0.0);
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn embed_models_are_preferred_by_quality_not_order() {
        let installed = vec![
            "llama3:latest".to_string(),
            "all-minilm:latest".to_string(),
            "nomic-embed-text:latest".to_string(),
        ];
        assert_eq!(pick_embed_model(&installed).as_deref(), Some("nomic-embed-text:latest"));
    }

    #[test]
    fn an_unlisted_embedding_model_is_still_found() {
        let installed = vec!["llama3:latest".into(), "some-other-embed:v2".into()];
        assert_eq!(pick_embed_model(&installed).as_deref(), Some("some-other-embed:v2"));
    }

    #[test]
    fn no_embedding_model_means_no_semantic_search() {
        // Chat models must not be used as embedders: they return a vector and retrieve badly.
        let installed = vec!["llama3:latest".into(), "qwen3:27b".into()];
        assert_eq!(pick_embed_model(&installed), None);
    }

    #[test]
    fn ranking_keeps_the_best_chunk_per_page_and_drops_noise() {
        let query = vec![1.0, 0.0];
        let a_chunks = vec![
            Chunk { heading: "A1".into(), text: "weak".into() },
            Chunk { heading: "A2".into(), text: "strong".into() },
        ];
        let a_vecs = vec![vec![0.5, 0.86], vec![1.0, 0.0]];          // ~0.5 and 1.0
        let b_chunks = vec![Chunk { heading: "B".into(), text: "unrelated".into() }];
        let b_vecs = vec![vec![0.0, 1.0]];                            // 0.0 — below MIN_SCORE
        let pages = vec![
            ("a.md".to_string(), a_chunks.as_slice(), a_vecs.as_slice()),
            ("b.md".to_string(), b_chunks.as_slice(), b_vecs.as_slice()),
        ];
        let hits = rank(&query, &pages, 10);
        assert_eq!(hits.len(), 1, "unrelated page must be filtered out, not ranked last");
        assert_eq!(hits[0].path, "a.md");
        assert_eq!(hits[0].heading, "A2", "the stronger chunk represents the page");
    }

    #[test]
    fn ranking_honours_the_limit_and_orders_by_score() {
        let query = vec![1.0, 0.0];
        let mk = |v: Vec<f32>| (vec![Chunk { heading: "h".into(), text: "t".into() }], vec![v]);
        let (c1, v1) = mk(vec![1.0, 0.0]);      // 1.00
        let (c2, v2) = mk(vec![0.9, 0.44]);     // ~0.90
        let (c3, v3) = mk(vec![0.8, 0.6]);      // ~0.80
        let pages = vec![
            ("low.md".to_string(), c3.as_slice(), v3.as_slice()),
            ("high.md".to_string(), c1.as_slice(), v1.as_slice()),
            ("mid.md".to_string(), c2.as_slice(), v2.as_slice()),
        ];
        let hits = rank(&query, &pages, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].path, "high.md");
        assert_eq!(hits[1].path, "mid.md");
    }

    /// Pins the Ollama `/api/embed` contract: the batch request shape and the response
    /// field name. Reading the code cannot tell you whether this matches the real server,
    /// and a silent mismatch would make semantic search quietly return nothing.
    #[tokio::test]
    async fn embed_batch_sends_a_batch_and_parses_the_response() {
        use wiremock::matchers::{body_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .and(body_json(serde_json::json!({
                "model": "nomic-embed-text",
                "input": ["first", "second"]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "nomic-embed-text",
                "embeddings": [[0.1, 0.2], [0.3, 0.4]]
            })))
            .mount(&server)
            .await;

        let backend = Backend::ollama(server.uri());
        let out = embed_batch(&backend, "nomic-embed-text", &["first".into(), "second".into()])
            .await
            .expect("embed should succeed");
        assert_eq!(out, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    }

    #[tokio::test]
    async fn embed_batch_reports_a_failing_server_rather_than_returning_empty() {
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let backend = Backend::ollama(server.uri());
        let err = embed_batch(&backend, "missing-model", &["x".into()]).await.unwrap_err();
        assert!(err.to_string().contains("404"), "error should name the status: {err}");
    }

    #[tokio::test]
    async fn embedding_nothing_makes_no_request() {
        // Guards against an empty page triggering a pointless round trip on every search.
        let backend = Backend::ollama("http://127.0.0.1:1");  // would fail if contacted
        assert!(embed_batch(&backend, "m", &[]).await.unwrap().is_empty());
    }

    /// End-to-end against the real Ollama and the real wiki. Ignored by default (CI has
    /// neither); run with:
    ///   cargo test --lib wiki_index -- --ignored --nocapture live_semantic_search
    #[tokio::test]
    #[ignore]
    async fn live_semantic_search_finds_pages_by_meaning() {
        let backend = Backend::ollama("http://localhost:11434");
        let query = std::env::var("WIKI_QUERY").unwrap_or_else(|_| "what do I need to do".into());
        let started = std::time::Instant::now();
        let hits = semantic_search(&backend, &query, 5)
            .await
            .expect("semantic search failed")
            .expect("no embedding model installed");
        println!("\nquery: {query:?}  ({} ms)", started.elapsed().as_millis());
        for h in &hits {
            println!("  {:.3}  {}  [{}]\n         {}", h.score, h.path, h.heading, h.snippet);
        }
        assert!(!hits.is_empty(), "expected at least one semantic hit");
    }

    #[test]
    fn centroid_averages_a_page_into_one_point() {
        let c = centroid(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(c, vec![0.5, 0.5]);
    }

    #[test]
    fn centroid_refuses_ragged_or_empty_input() {
        // A corrupt index entry must yield no edge rather than a meaningless one.
        assert!(centroid(&[]).is_empty());
        assert!(centroid(&[vec![1.0, 2.0], vec![3.0]]).is_empty());
    }

    #[test]
    fn content_hash_tracks_edits() {
        assert_eq!(content_hash("same"), content_hash("same"));
        assert_ne!(content_hash("before"), content_hash("after"));
    }
}
