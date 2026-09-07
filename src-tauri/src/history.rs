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
