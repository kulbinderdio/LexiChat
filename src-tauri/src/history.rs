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
    let json = serde_json::to_string_pretty(conv)?;
    std::fs::write(conversation_path(&conv.meta.id), json)?;

    let mut index = load_index();
    index.retain(|m| m.id != conv.meta.id);
    index.push(conv.meta.clone());
    index.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    save_index(&index)
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
            model: "m".into(), created_at: 1, updated_at: 2, message_count: 3,
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
}
