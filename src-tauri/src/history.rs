// Chat history: on-disk persistence of past conversations.
//
// Each conversation is stored as its own JSON file under `<data>/lexichat/
// conversations/{id}.json`, holding both the authoritative Ollama `wire`
// history and the opaque frontend `display` messages. A lightweight
// `index.json` of metadata lets the sidebar list load without parsing every
// (potentially image-heavy) conversation file. Mirrors the load/save idiom in
// `jobs.rs`.

use serde::{Deserialize, Serialize};
use crate::ollama::WireMessage;

/// Lightweight per-conversation metadata — this is what the history list shows.
#[derive(Clone, Serialize, Deserialize)]
pub struct ConversationMeta {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub model: String,
    pub created_at: i64, // unix seconds
    pub updated_at: i64,
    #[serde(default)]
    pub message_count: usize,
    /// Pinned chats sort to the top of the history list regardless of recency. The index is the
    /// authority for this — see `save_one`, which preserves it across a message-triggered save so
    /// a new message never silently unpins a chat. `#[serde(default)]` keeps pre-pinning files
    /// loading as unpinned.
    #[serde(default)]
    pub pinned: bool,
}

/// A full saved conversation: metadata + backend wire history + frontend display.
#[derive(Clone, Serialize, Deserialize)]
pub struct Conversation {
    #[serde(flatten)]
    pub meta: ConversationMeta,
    /// Authoritative message history sent to Ollama — restored verbatim on load.
    #[serde(default)]
    pub wire: Vec<WireMessage>,
    /// Opaque frontend `ChatMessage[]` used only for rendering.
    #[serde(default)]
    pub display: serde_json::Value,
    /// Absolute paths of files the user attached during this chat. Persisted so a reopened
    /// conversation can still edit the photo it was about, rather than reporting it as no
    /// longer attached. Only paths are stored, never the bytes — the files live wherever the
    /// user keeps them, so one may have moved or been deleted by the time the chat is reopened.
    #[serde(default)]
    pub attachments: Vec<String>,
}

fn conversations_dir() -> std::path::PathBuf {
    let dir = crate::dirs_path().join("conversations");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn index_path() -> std::path::PathBuf {
    conversations_dir().join("index.json")
}

fn conversation_path(id: &str) -> std::path::PathBuf {
    conversations_dir().join(format!("{id}.json"))
}

/// Mint a new conversation id. Microsecond timestamp is unique enough given a
/// new conversation is created at most once per "New chat".
pub fn new_id() -> String {
    format!("conv-{}", chrono::Utc::now().timestamp_micros())
}

pub fn now_secs() -> i64 {
    chrono::Utc::now().timestamp()
}

/// Load the metadata index (newest first). Missing/corrupt → empty list.
/// A conversation's on-disk footprint: its saved JSON plus any working files kept alongside it
/// (offloaded tool results and /work/artifacts data). Computed on demand rather than stored in
/// index.json — it is derived state and would go stale the moment a turn wrote anything.
pub fn disk_size(id: &str) -> u64 {
    fn dir_size(p: &std::path::Path) -> u64 {
        let Ok(entries) = std::fs::read_dir(p) else { return 0 };
        entries.flatten().map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path()),
            _ => e.metadata().map(|m| m.len()).unwrap_or(0),
        }).sum()
    }
    let json = conversation_path(id).metadata().map(|m| m.len()).unwrap_or(0);
    json + dir_size(&conversations_dir().join(format!("{id}-files")))
}

pub fn load_index() -> Vec<ConversationMeta> {
    std::fs::read_to_string(index_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_index(index: &[ConversationMeta]) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(index)?;
    std::fs::write(index_path(), json)?;
    Ok(())
}

pub fn load_one(id: &str) -> Option<Conversation> {
    std::fs::read_to_string(conversation_path(id))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

/// Write the conversation file and upsert its meta into the index (newest first).
pub fn save_one(conv: &Conversation) -> anyhow::Result<()> {
    let mut index = load_index();

    // Organizer metadata (pinned, and later folder) is set through its own command, not carried
    // in the frontend's save payload — so a message-triggered save arrives with pinned = false.
    // Preserve the index's value into the record we write, or a new message would unpin the chat.
    let mut conv = conv.clone();
    if let Some(prev) = index.iter().find(|m| m.id == conv.meta.id) {
        conv.meta.pinned = prev.pinned;
    }

    let json = serde_json::to_string_pretty(&conv)?;
    std::fs::write(conversation_path(&conv.meta.id), json)?;

    index.retain(|m| m.id != conv.meta.id);
    index.push(conv.meta.clone());
    index.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    save_index(&index)
}

/// Pin or unpin a chat. Index-only: pinned state lives in the index (the authority), so this is a
/// cheap metadata write, not a rewrite of a possibly-large conversation file. `save_one` carries
/// the value back into the file on the next message.
pub fn set_pinned(id: &str, pinned: bool) -> anyhow::Result<()> {
    let mut index = load_index();
    if let Some(m) = index.iter_mut().find(|m| m.id == id) {
        m.pinned = pinned;
        save_index(&index)?;
    }
    Ok(())
}

/// A conversation whose title or message text contains the query, with a short snippet around the
/// first hit for the history list to show under the title.
#[derive(Clone, Serialize)]
pub struct SearchHit {
    pub id: String,
    pub snippet: String,
}

/// A window of `text` centred on the first case-insensitive occurrence of `q` (already lowercased),
/// with ellipses and collapsed whitespace, so a match deep in a long message reads as one tidy line.
fn make_snippet(text: &str, q: &str) -> String {
    let lower = text.to_lowercase();
    let chars: Vec<char> = text.chars().collect();
    // Map the byte position of the match to a char index. Lowercasing can shift lengths for a few
    // unicode chars, so this is best-effort for display — a slight offset only nudges the window.
    let cpos = lower.find(q).map(|b| lower[..b].chars().count()).unwrap_or(0);
    let start = cpos.saturating_sub(40);
    let end = (cpos + q.chars().count() + 60).min(chars.len());
    let mut s = String::new();
    if start > 0 { s.push('…'); }
    s.extend(&chars[start..end]);
    if end < chars.len() { s.push('…'); }
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Full-text search over saved chats: title plus user/assistant message content. Tool results and
/// system text are excluded — they are bulk/noise, not what "search my chats" means.
///
/// A raw-substring pre-filter rejects non-matching files before the (more expensive) JSON parse, so
/// only files that actually contain the query are parsed to build a clean snippet. At a few hundred
/// conversations this is fast enough for a debounced as-you-type search without a separate index.
pub fn search(query: &str, profile_id: Option<&str>) -> Vec<SearchHit> {
    let q = query.trim().to_lowercase();
    if q.is_empty() { return Vec::new(); }

    let mut hits = Vec::new();
    for meta in load_index() {
        if let Some(p) = profile_id {
            if meta.profile_id.as_deref() != Some(p) { continue; }
        }
        let Ok(raw) = std::fs::read_to_string(conversation_path(&meta.id)) else { continue };
        if !raw.to_lowercase().contains(&q) { continue; } // fast reject, no parse

        // Title is the cheapest relevant match.
        if meta.title.to_lowercase().contains(&q) {
            hits.push(SearchHit { id: meta.id.clone(), snippet: make_snippet(&meta.title, &q) });
            continue;
        }
        // Otherwise look in the actual conversation text. A raw match that is only in a tool result
        // or system prompt is treated as noise and skipped.
        if let Ok(conv) = serde_json::from_str::<Conversation>(&raw) {
            if let Some(text) = conv.wire.iter()
                .filter(|m| matches!(m.role.as_str(), "user" | "assistant"))
                .filter_map(|m| m.content.as_deref())
                .find(|c| c.to_lowercase().contains(&q))
            {
                hits.push(SearchHit { id: meta.id, snippet: make_snippet(text, &q) });
            }
        }
    }
    hits
}

pub fn delete_one(id: &str) -> anyhow::Result<()> {
    let _ = std::fs::remove_file(conversation_path(id));
    let mut index = load_index();
    index.retain(|m| m.id != id);
    save_index(&index)
}

pub fn rename(id: &str, title: &str) -> anyhow::Result<()> {
    if let Some(mut conv) = load_one(id) {
        conv.meta.title = title.to_string();
        save_one(&conv)?;
    }
    Ok(())
}

/// Split saved attachment paths into those still on disk and those that have gone.
/// A path is only useful if the file is still there, and a vanished one is worth naming:
/// the model can then say the file has moved instead of silently reinventing it.
pub fn split_attachments(paths: &[String]) -> (Vec<String>, Vec<String>) {
    paths.iter().cloned().partition(|p| std::path::Path::new(p).is_file())
}

/// Title = first line of the first user message, trimmed to ~40 chars.
pub fn derive_title(wire: &[WireMessage]) -> String {
    let raw = wire
        .iter()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_deref())
        .unwrap_or("")
        .trim();
    if raw.is_empty() {
        return "New conversation".to_string();
    }
    let first_line = raw.lines().next().unwrap_or(raw).trim();
    let mut title: String = first_line.chars().take(40).collect();
    if first_line.chars().count() > 40 {
        title.push('…');
    }
    title
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The history list flattens meta and appends a derived size. Guards the wire shape the
    /// sidebar reads — a rename or a missed flatten would silently drop the size indicator.
    #[test]
    fn list_item_serialises_flat_meta_plus_size() {
        #[derive(serde::Serialize)]
        struct Item { #[serde(flatten)] meta: ConversationMeta, size_bytes: u64 }
        let meta = ConversationMeta {
            id: "conv-1".into(), title: "t".into(), profile_id: None,
            model: "m".into(), created_at: 1, updated_at: 2, message_count: 3, pinned: false,
        };
        let v = serde_json::to_value(Item { meta, size_bytes: 4096 }).unwrap();
        assert_eq!(v["id"], "conv-1");            // flattened, not nested under "meta"
        assert_eq!(v["message_count"], 3);
        assert_eq!(v["size_bytes"], 4096);
        assert!(v.get("meta").is_none());
    }

    /// A conversation with no working-files directory must report just its JSON size, not fail.
    #[test]
    fn disk_size_handles_a_missing_files_dir() {
        assert_eq!(disk_size("conv-does-not-exist-at-all"), 0);
    }

    /// An index written before pinning existed has no `pinned` field; it must load as unpinned,
    /// not fail to deserialize (which would blank the whole history list).
    #[test]
    fn pre_pinning_meta_loads_as_unpinned() {
        let old = r#"{"id":"c1","title":"t","profile_id":null,"model":"m",
                      "created_at":1,"updated_at":2,"message_count":3}"#;
        let m: ConversationMeta = serde_json::from_str(old).unwrap();
        assert!(!m.pinned, "a chat from before pinning must default to unpinned");
    }

    #[test]
    fn snippet_windows_around_the_match_and_collapses_whitespace() {
        let text = "The quick brown fox\njumps over   the lazy dog by the river".to_string();
        let s = make_snippet(&text, "lazy");
        assert!(s.contains("lazy"), "snippet must contain the match: {s:?}");
        assert!(!s.contains('\n') && !s.contains("   "), "whitespace collapsed: {s:?}");
    }

    #[test]
    fn snippet_adds_ellipses_only_when_it_actually_truncates() {
        // Match at the very start of a short string: no leading ellipsis, no trailing one.
        assert_eq!(make_snippet("hello world", "hello"), "hello world");
        // Match deep in a long string: both ends elided.
        let long = "x ".repeat(80) + "needle " + &"y ".repeat(80);
        let s = make_snippet(&long, "needle");
        assert!(s.starts_with('…') && s.ends_with('…'), "both ends elided: {s:?}");
    }

    /// Attachments are paths, not bytes, so a file can vanish between saving a chat and
    /// reopening it. The split is what lets the reopened chat use what survives and name
    /// what didn't, instead of handing the model a path that fails on read.
    #[test]
    fn split_attachments_separates_present_from_vanished() {
        let here = std::env::current_exe().unwrap().to_string_lossy().to_string();
        let gone = "/definitely/not/a/real/path/photo.jpg".to_string();
        let (present, missing) = split_attachments(&[here.clone(), gone.clone()]);
        assert_eq!(present, vec![here]);
        assert_eq!(missing, vec![gone]);
    }

    /// A conversation saved before attachments were tracked must still load.
    #[test]
    fn conversation_without_attachments_field_still_deserialises() {
        let json = r#"{"id":"c1","title":"t","model":"m","created_at":1,"updated_at":2,
                       "message_count":0,"wire":[],"display":[]}"#;
        let conv: Conversation = serde_json::from_str(json).unwrap();
        assert!(conv.attachments.is_empty());
    }
}
